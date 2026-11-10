"""DeepEP layout adapter for HCU channel-INT4 W4A16 Triton experts."""
from dataclasses import replace
from functools import lru_cache

import torch

from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()


def _selected(layer):
    if not _is_hcu:
        return False
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
        CompressedTensorsWNA16TritonMoE,
    )
    scheme = getattr(layer, "scheme", None)
    return (
        isinstance(scheme, CompressedTensorsWNA16TritonMoE)
        and scheme.num_bits == 4
        and scheme.strategy == "channel"
        and scheme.group_size == -1
        and scheme.sym
        and scheme.actorder is None
    )


def around_ep_init(original, layer, *args, **kwargs):
    result = original(layer, *args, **kwargs)
    if not _selected(layer):
        return result
    from sglang.srt.layers.moe.moe_runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    config = layer.moe_runner_config
    assert config.activation == "silu" and config.is_gated
    assert not config.no_combine and not config.apply_router_weight_on_input
    assert not config.silu_mul_keep_fp32 and not layer.num_fused_shared_experts
    assert not layer.should_fuse_routed_scaling_factor_in_topk
    # The existing DeepSeek caller applies this factor after DeepEP combine.
    local_config = replace(config, inplace=False, routed_scaling_factor=None)
    layer._hcu_wna16_deepep_runner = MoeRunner(MoeRunnerBackend.TRITON, local_config)
    previous = getattr(layer.dispatcher, "quant_config", None) or {}
    layer.dispatcher.set_quant_config({**previous, "preserve_input_dtype": True})
    return result


@lru_cache(maxsize=None)
def _routing_slots(device, experts, capacity):
    # Immutable graph operands must remain alive until process exit.
    expert_ids = torch.arange(experts, device=device, dtype=torch.int32).view(experts, 1).expand(experts, capacity)
    positions = torch.arange(capacity, device=device, dtype=torch.int32).view(1, capacity)
    weights = torch.ones((experts * capacity, 1), device=device, dtype=torch.float32)
    return expert_ids, positions, weights


def _run_standard(layer, hidden_states, topk_ids, topk_weights):
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    assert hidden_states.dtype == torch.bfloat16
    info = layer.scheme.get_triton_quant_info(layer)
    assert info.use_int4_w4a16
    assert info.block_shape == [0, -1]
    # Repeated channel scales are exact group-128 views. A positive group
    # size selects the existing INT4 kernel instead of the plain GEMM path.
    scales = []
    for weight, scale in ((info.w13_weight, info.w13_scale), (info.w2_weight, info.w2_scale)):
        assert weight.dtype == torch.uint8 and scale.ndim == 3
        assert scale.shape[:2] == weight.shape[:2] and scale.shape[-1] == 1
        assert scale.device == weight.device
        k = weight.shape[-1] * 2
        assert k % 128 == 0
        scales.append(scale.expand(*scale.shape[:2], k // 128))
    info = replace(info, w13_scale=scales[0], w2_scale=scales[1], block_shape=[0, 128])
    dispatch = StandardDispatchOutput(hidden_states, None, StandardTopKOutput(topk_weights, topk_ids, None))
    result = layer._hcu_wna16_deepep_runner.run(dispatch, info)
    assert result.hidden_states.shape == hidden_states.shape
    assert result.hidden_states.dtype == hidden_states.dtype
    return result.hidden_states


def around_run_moe_core(original, layer, dispatch_output, *args, **kwargs):
    if not _selected(layer):
        return original(layer, dispatch_output, *args, **kwargs)
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )

    assert getattr(layer, "down_gemm_overlap_args", None) is None
    assert getattr(layer, "meta_overlap_args", None) is None
    assert dispatch_output.hidden_states_scale is None
    x = dispatch_output.hidden_states
    assert x.dtype == torch.bfloat16
    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        assert x.ndim == 2
        output = _run_standard(layer, x, dispatch_output.topk_ids, dispatch_output.topk_weights)
        wrapper = DeepEPNormalCombineInput
    elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        assert x.ndim == 3
        experts, capacity, hidden = x.shape
        assert experts == layer.num_local_experts and dispatch_output.masked_m.shape == (experts,)
        ids, positions, weights = _routing_slots(x.device, experts, capacity)
        valid = positions < dispatch_output.masked_m.view(experts, 1)
        local_ids = torch.where(valid, ids, -1).reshape(-1, 1)
        inputs = torch.where(valid[..., None], x, 0).reshape(-1, hidden)
        output = _run_standard(layer, inputs, local_ids, weights).view(experts, capacity, hidden)
        output = torch.where(valid[..., None], output, 0)
        wrapper = DeepEPLLCombineInput
    else:
        raise ValueError("HCU channel INT4 W4A16 requires DeepEP normal or low-latency dispatch")
    return wrapper(hidden_states=output, topk_ids=dispatch_output.topk_ids, topk_weights=dispatch_output.topk_weights)


def register_hooks():
    if not _is_hcu:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register("sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.__init__", around_ep_init, HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core", around_run_moe_core, HookType.AROUND)
