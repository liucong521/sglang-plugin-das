from hcu_sglang.utils.detection import is_hcu


def around_get_quant_method(original, config, layer, prefix):
    if not is_hcu() or config.quant_format != "pack-quantized":
        return original(config, layer, prefix)
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    if not isinstance(layer, DeepEPMoE) or not get_moe_a2a_backend().is_deepep():
        return original(config, layer, prefix)
    config._add_fused_moe_to_target_scheme_map()
    try:
        schemes = [
            config.get_scheme_dict(layer, prefix + name)
            for name in (".0.w1", ".0.w3", ".0.w2")
        ]
    except ValueError:
        return original(config, layer, prefix)
    scheme = schemes[0]
    if scheme is None or not all(item == scheme for item in schemes):
        return original(config, layer, prefix)
    weight = scheme.get("weights")
    activation = scheme.get("input_activations")
    if (
        weight is None
        or activation is None
        or not config._is_dynamic_token_w4a8(weight, activation)
        or weight.strategy != "channel"
        or weight.group_size != -1
        or not weight.symmetric
        or not activation.symmetric
        or weight.actorder is not None
        or weight.dynamic
    ):
        return original(config, layer, prefix)
    from hcu_sglang.layers.quantization.compressed_w4a8_deepep import (
        HCUCompressedW4A8DeepEPMoEMethod,
    )

    return HCUCompressedW4A8DeepEPMoEMethod(config)


def around_run_moe_core(original, layer, dispatch_output, *args, **kwargs):
    if not is_hcu():
        return original(layer, dispatch_output, *args, **kwargs)
    from hcu_sglang.layers.quantization.compressed_w4a8_deepep import (
        HCUCompressedW4A8DeepEPMoEMethod,
    )

    if not isinstance(layer.quant_method, HCUCompressedW4A8DeepEPMoEMethod):
        return original(layer, dispatch_output, *args, **kwargs)
    assert not args and not kwargs
    assert getattr(layer, "down_gemm_overlap_args", None) is None
    assert getattr(layer, "meta_overlap_args", None) is None
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )
    from hcu_sglang.layers.moe.w4a8_hipc import forward_masked, forward_normal

    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        output = forward_normal(layer, dispatch_output)
        wrapper = DeepEPNormalCombineInput
    elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        output = forward_masked(layer, dispatch_output)
        layer.dispatcher._execute("record_combine_input_ready_event")
        wrapper = DeepEPLLCombineInput
    else:
        raise ValueError("HCU W4A8 requires DeepEP normal or low-latency dispatch")
    return wrapper(hidden_states=output, topk_ids=dispatch_output.topk_ids,
                   topk_weights=dispatch_output.topk_weights)



def around_weight_loader_impl(original, layer, param, loaded_weight,
                              weight_name, shard_id, expert_id):
    if is_hcu():
        from hcu_sglang.layers.quantization.compressed_w4a8_deepep import (
            HCUCompressedW4A8DeepEPMoEMethod,
        )

        method = layer.scheme if layer.scheme is not None else layer.quant_method
        if isinstance(method, HCUCompressedW4A8DeepEPMoEMethod) and "zero" not in weight_name:
            # The native loader's CT transpose allowlist excludes this plugin class.
            loaded_weight = loaded_weight.t().contiguous()
    return original(layer, param, loaded_weight, weight_name, shard_id, expert_id)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
        "CompressedTensorsConfig.get_quant_method",
        around_get_quant_method,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core",
        around_run_moe_core,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE._weight_loader_impl",
        around_weight_loader_impl,
        HookType.AROUND,
    )
