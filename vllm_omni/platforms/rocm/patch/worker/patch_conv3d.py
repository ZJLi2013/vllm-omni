# SPDX-License-Identifier: Apache-2.0

"""Route the Wan VAE's causal 3D convolutions to the FlyDSL implicit-GEMM kernel on ROCm.

On gfx950 MIOpen wraps every 3D convolution in NCDHW<->NDHWC transposes, which cost
17% of the Wan VAE encoder. FlyDSL's `conv3d_implicit` caches the packed weight and
writes NCDHW straight out of its epilogue, and is 1.2x faster on the encoder's own
shapes; the convolutions it cannot take fall back to MIOpen untouched.
"""

import torch
import torch.nn as nn
from vllm.logger import init_logger

import vllm_omni.diffusion.registry as _registry_mod

logger = init_logger(__name__)

# `conv3d_implicit` loads LDG_VEC=8 bf16 channels per thread and asserts
# `c % LDG_VEC == 0`. The encoder's first layer is RGB (C=3) and stays on MIOpen.
_CHANNEL_MULTIPLE = 8

_original_conv_forward = nn.Conv3d._conv_forward


@torch.library.custom_op("vllm_omni::flydsl_conv3d", mutates_args=())
def flydsl_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
) -> torch.Tensor:
    """FlyDSL implicit-GEMM 3D convolution, zero padding, dense, dilation 1.

    Registered as a custom op because the VAE encoder is compiled with
    ``fullgraph=True``; dynamo cannot trace into the FlyDSL launcher.
    """
    from kernels.conv.conv3d_implicit import conv3d_implicit

    return conv3d_implicit(x, weight, bias, stride=tuple(stride), padding=0)


@flydsl_conv3d.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
) -> torch.Tensor:
    n = x.shape[0]
    k, _, kt, kh, kw = weight.shape
    st, sh, sw = stride
    out = (
        n,
        k,
        (x.shape[2] - kt) // st + 1,
        (x.shape[3] - kh) // sh + 1,
        (x.shape[4] - kw) // sw + 1,
    )
    return x.new_empty(out)


def _takes_flydsl(module: nn.Conv3d, x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.shape[1] % _CHANNEL_MULTIPLE == 0
        and module.groups == 1
        and module.padding_mode == "zeros"
        and all(p == 0 for p in module.padding)
        and all(d == 1 for d in module.dilation)
    )


def _flydsl_conv_forward(self, x, weight, bias):
    if _takes_flydsl(self, x, weight):
        return torch.ops.vllm_omni.flydsl_conv3d(x, weight, bias, list(self.stride))
    return _original_conv_forward(self, x, weight, bias)


def _prewarm_weights(convs: list[nn.Conv3d]) -> None:
    """Pack every weight before capture, from whatever the parameters hold now.

    Two reasons this cannot be left to the first call. It has to run *before*
    CUDAGraph capture: ``_prep_weight`` caches the packed weight, so left alone
    the first use happens inside the recording, the packed copies land in the
    graph's memory pool, and cudagraph_trees rejects the graph over untracked
    live allocations. And it has to run *after* the checkpoint is loaded: that
    cache is keyed on ``id(weight)`` and validated with a weakref, while weight
    loading is an in-place ``param.data.copy_()`` -- same object, live weakref,
    so a pack taken before the load is never invalidated and every convolution
    silently keeps running on the pre-load values. Dropping the cache here makes
    the pack follow the current values whenever this is called again.
    """
    from kernels.conv.conv3d_implicit import _WEIGHT_CACHE, _prep_weight

    _WEIGHT_CACHE.clear()
    for m in convs:
        k, c, kt, kh, kw = m.weight.shape
        _prep_weight(m.weight, k, kt, kh, kw, c)


def _selftest(conv: nn.Conv3d) -> None:
    """One convolution both ways on the loaded weights; log the relative error.

    A packed weight that has drifted from its parameter shows up here as a
    macroscopic error, which is the failure this whole path is prone to.
    """
    k, c, kt, kh, kw = conv.weight.shape
    h, w = kh + 7, kw + 7
    while (kt * h * w) % 8:
        w += 1
    x = torch.randn((1, c, kt, h, w), device=conv.weight.device, dtype=torch.bfloat16)
    got = torch.ops.vllm_omni.flydsl_conv3d(x, conv.weight, conv.bias, list(conv.stride))
    want = _original_conv_forward(conv, x, conv.weight, conv.bias)
    rel = ((got.float() - want.float()).abs().max() / want.float().abs().max().clamp_min(1e-6)).item()
    if rel > 1e-2:
        logger.warning("FlyDSL conv3d selftest: relative error %.3e on %s -- packed "
                       "weights may be stale.", rel, tuple(conv.weight.shape))
    else:
        logger.info("FlyDSL conv3d selftest: relative error %.3e on %s.", rel, tuple(conv.weight.shape))


def _patch_wan_causal_conv3d(vae: nn.Module) -> int:
    """Point every ``WanCausalConv3d`` at FlyDSL. Returns how many are in this VAE.

    ``WanCausalConv3d.forward`` applies the causal padding itself and leaves
    ``self.padding`` at zero, so overriding ``_conv_forward`` -- which sees the
    already-padded tensor -- leaves the causal bookkeeping alone.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    convs = [m for m in vae.modules() if isinstance(m, WanCausalConv3d)]
    if not convs:
        return 0
    if WanCausalConv3d._conv_forward is not _flydsl_conv_forward:
        WanCausalConv3d._conv_forward = _flydsl_conv_forward
    takeable = [m for m in convs if m.in_channels % _CHANNEL_MULTIPLE == 0 and m.groups == 1]
    _prewarm_weights(takeable)
    if takeable:
        try:
            _selftest(min(takeable, key=lambda m: m.weight.numel()))
        except Exception:
            logger.warning("FlyDSL conv3d selftest did not run.", exc_info=True)
    return len(convs)


_original_initialize_model = _registry_mod.initialize_model


def _patched_initialize_model(od_config):
    model = _original_initialize_model(od_config)

    vae = getattr(model, "vae", None)
    if vae is None:
        return model
    try:
        # The kernels live in the FlyDSL repo tree, not in the PyPI wheel.
        from kernels.conv.conv3d_implicit import conv3d_implicit  # noqa: F401
    except ImportError:
        logger.debug("FlyDSL conv3d not available; VAE convolutions stay on MIOpen.")
        return model

    def _enable(when: str) -> None:
        try:
            count = _patch_wan_causal_conv3d(vae)
            if count:
                logger.info("FlyDSL conv3d is enabled for %d Wan VAE convolutions (%s).", count, when)
        except Exception:
            logger.warning("Failed to enable FlyDSL conv3d for VAE.", exc_info=True)

    _enable("at init")

    # Weights arrive after this returns, in place, so the packed copies taken
    # above describe the pre-load values. Repack once the real ones are in.
    load_weights = getattr(model, "load_weights", None)
    if callable(load_weights):

        def _load_weights_then_repack(*args, **kwargs):
            out = load_weights(*args, **kwargs)
            _enable("after load_weights")
            return out

        model.load_weights = _load_weights_then_repack
    else:
        logger.warning("No load_weights on the model; FlyDSL packed weights may be stale.")

    return model


_registry_mod.initialize_model = _patched_initialize_model
