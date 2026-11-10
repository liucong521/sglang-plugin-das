"""HCU speculative-decoding and optional dependency compatibility."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def around_handle_dspark(original: Callable[..., None], server_args: Any) -> None:
    if not is_hcu():
        return original(server_args)
    from hcu_sglang.speculative_dspark import _handle_dspark

    return _handle_dspark(server_args)


def around_inference_only(original: Callable[..., Any], *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    with torch.inference_mode():
        return original(*args, **kwargs)


def around_dspark_dequant_supported(original: Callable[..., bool], linear: Any) -> bool:
    if (
        is_hcu()
        and getattr(getattr(linear, "weight", None), "dtype", None)
        == torch.float8_e4m3fn
        and not hasattr(linear, "weight_scale_inv")
    ):
        return False
    return original(linear)


def _get_dspark_ragged_settings():
    from sglang.srt.environ import envs
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    return (
        read_ragged_verify_mode(),
        RaggedVerifyMode.STATIC,
        envs.SGLANG_PREP_IN_CUDA_GRAPH.get(),
    )


def before_dspark_verify_planner_init(*args, **kwargs) -> None:
    del args, kwargs
    if not is_hcu():
        return None

    ragged_verify_mode, static_mode, prep_in_cuda_graph = _get_dspark_ragged_settings()
    if ragged_verify_mode is not static_mode and not prep_in_cuda_graph:
        raise ValueError(
            f"DSpark ragged-verify mode {ragged_verify_mode.value!r} "
            f"requires SGLANG_PREP_IN_CUDA_GRAPH=1 (the captured-graph prepare "
            f"path). It is currently disabled, which would put per-step "
            f"verify_lens_cpu host reads on the critical path. Set "
            f"SGLANG_PREP_IN_CUDA_GRAPH=1 or run SGLANG_RAGGED_VERIFY_MODE=static."
        )
    return None


def _needs_mhc_hidden_states(model_runner: Any) -> bool:
    model_config = getattr(model_runner, "model_config", None)
    return getattr(model_config, "hc_hidden_size", None) is not None


def around_prepare_for_draft(original: Callable[..., Any], *args, **kwargs):
    result = original(*args, **kwargs)
    forward_batch = result[0]
    draft_model_runner = kwargs.get("draft_model_runner")
    if draft_model_runner is None and len(args) >= 5:
        draft_model_runner = args[4]
    if _needs_mhc_hidden_states(draft_model_runner):
        forward_batch.return_hidden_states_before_norm = True
    return result




def around_tbo_filter_batch(
    original: Callable[..., Any], cls: type, batch: Any, *args, **kwargs
):
    child_batch = original(cls, batch, *args, **kwargs)
    if is_hcu() and getattr(batch, "return_hidden_states_before_norm", False):
        child_batch.return_hidden_states_before_norm = True
    return child_batch


def patch_broken_torchaudio_availability() -> None:
    try:
        import torchaudio  # noqa: F401
    except (ImportError, OSError):
        try:
            import transformers.utils as transformers_utils
            import transformers.utils.import_utils as import_utils
        except ImportError:
            return
        import_utils.is_torchaudio_available = lambda: False
        transformers_utils.is_torchaudio_available = lambda: False


def around_dflash_top_p_renorm_prob(original, probs, top_ps):
    if not is_hcu() or probs.device.type != "cuda":
        return original(probs, top_ps)
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    return top_p_renorm_probs_triton(probs, top_ps)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    patch_broken_torchaudio_availability()
    from hcu_sglang.speculative_sampling import (
        around_can_use_sparse_target_sampling,
        around_sample_tree_target_tokens,
    )

    HookRegistry.register(
        "sglang.srt.speculative.eagle_utils._can_use_sparse_uno_tree_target_sampling",
        around_can_use_sparse_target_sampling,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.uno_utils.sample_uno_tree_target_tokens",
        around_sample_tree_target_tokens,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.arg_groups.speculative_hook._handle_dspark",
        around_handle_dspark,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_draft_sampler."
        "DsparkDraftSampler.stage_sampling_params",
        around_inference_only,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2."
        "DSparkWorkerV2.init_cuda_graphs",
        around_inference_only,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.speculative.dspark.dspark_draft_model."
        "_dequant_supported",
        around_dspark_dequant_supported,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_planner."
        "DSparkVerifyPlanner.__init__",
        before_dspark_verify_planner_init,
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_common.prepare_for_draft",
        around_prepare_for_draft,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.batch_overlap.two_batch_overlap."
        "TboForwardBatchPreparer.filter_batch",
        around_tbo_filter_batch,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dflash_utils._dflash_top_p_renorm_prob",
        around_dflash_top_p_renorm_prob,
        HookType.AROUND,
    )
