# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch.profiler import record_function
from hcu_sglang.layers.moe.deepep_fp8 import (
    _ep_scatter_with_optional_lightop, _ep_gather_with_optional_lightop,
    get_offloader, copy_list_to_gpu_no_ce,
)

def forward_normal(self, dispatch_output):
    from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc
    from lightop import fuse_silu_mul_clamp_quant, fuse_silu_mul_quant

    x = dispatch_output.hidden_states
    x_scale = dispatch_output.hidden_states_scale
    ids = dispatch_output.topk_ids
    weights = dispatch_output.topk_weights
    counts = dispatch_output.num_recv_tokens_per_expert
    if self.moe_runner_config.activation != "silu":
        raise ValueError("HCU W4A8 HIPC currently supports only SiLU")
    if (
        x.dtype != torch.int8
        or x_scale is None
        or x_scale.dtype != torch.float32
    ):
        raise RuntimeError(
            "HCU W4A8 reference normal requires INT8 dispatch with FP32 scales"
        )
    if x_scale.shape != x.shape[:-1] + (1,):
        raise RuntimeError("Unexpected HCU W4A8 normal scale shape")
    if len(counts) != self.w13_weight_scale.shape[0]:
        raise RuntimeError(
            "DeepEP expert counts do not match local W4A8 experts"
        )
    if any(count % 256 != 0 for count in counts):
        raise RuntimeError(
            "HCU W4A8 HIPC normal requires 256-aligned expert counts"
        )
    all_tokens = sum(counts)
    if all_tokens == 0:
        return torch.zeros_like(x, dtype=torch.bfloat16)

    # Reuse source-side INT8 quantization from DeepEP dispatch.
    a1 = torch.empty(
        (all_tokens, x.shape[1]), dtype=torch.int8, device=x.device
    )
    a1_scale = torch.empty(
        (all_tokens, 1), dtype=torch.float32, device=x.device
    )
    if get_offloader().forbid_copy_engine_usage:
        counts_gpu = copy_list_to_gpu_no_ce(counts)
    else:
        counts_gpu = torch.tensor(
            counts, dtype=torch.int32, pin_memory=True, device="cpu"
        ).to(device=x.device, non_blocking=True)
    m_indices, output_index = _ep_scatter_with_optional_lightop(
        x,
        x_scale,
        ids,
        counts_gpu,
        a1,
        a1_scale,
        all_tokens,
        counts_are_aligned=True,
    )
    del x_scale
    gate_up = torch.empty(
        (all_tokens, self.w13_weight_scale.shape[1]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (a1, a1_scale),
        (self.w13_weight, self.w13_weight_scale),
        gate_up,
        m_indices,
    )
    del a1, a1_scale
    with record_function("dsv4_routed_expert_silu_mul_int8_fused_contig"):
        if self.moe_runner_config.swiglu_limit is None:
            a2, a2_scale = fuse_silu_mul_quant(gate_up)
        else:
            a2, a2_scale = fuse_silu_mul_clamp_quant(
                gate_up, float(self.moe_runner_config.swiglu_limit)
            )
    del gate_up
    if a2.shape[-1] != self.w4a8_padded_intermediate_size:
        a2 = F.pad(a2, (0, self.w4a8_padded_intermediate_size - a2.shape[-1]))
    down = torch.empty(
        (all_tokens, self.w2_weight_scale.shape[1]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (a2, a2_scale),
        (self.w2_weight, self.w2_weight_scale),
        down,
        m_indices,
    )
    del a2, a2_scale
    output = torch.empty_like(x, dtype=torch.bfloat16)
    _ep_gather_with_optional_lightop(down, ids, weights, output_index, output)
    return output


def forward_masked(self, dispatch_output):
    from lightop import fuse_silu_mul_clamp_quant_ep, fuse_silu_mul_quant_ep

    x = dispatch_output.hidden_states
    scale = dispatch_output.hidden_states_scale
    masked_m = dispatch_output.masked_m
    if self.moe_runner_config.activation != "silu":
        raise ValueError("HCU W4A8 DeepGEMM currently supports only SiLU")
    if x.dtype != torch.int8 or scale is None or scale.dtype != torch.float32:
        raise RuntimeError(
            "HCU W4A8 DeepGEMM low-latency dispatch requires INT8 "
            "activations with FP32 per-token scales"
        )
    expected_scale_shape = x.shape[:-1] + (1,)
    if scale.shape != expected_scale_shape:
        raise RuntimeError(
            "Unexpected DeepEP low-latency activation scale shape: "
            f"{scale.shape} != {expected_scale_shape}"
        )
    if masked_m.dtype != torch.int32 or masked_m.shape != (x.shape[0],):
        raise RuntimeError(
            "Unexpected DeepEP low-latency masked_m metadata: "
            f"dtype={masked_m.dtype}, shape={masked_m.shape}"
        )
    if x.device != scale.device or x.device != masked_m.device:
        raise RuntimeError(
            "DeepEP low-latency activations, scales, and masked_m must "
            "be on the same device"
        )
    if not x.is_contiguous() or not scale.is_contiguous():
        raise RuntimeError(
            "HCU W4A8 DeepGEMM requires contiguous DeepEP low-latency "
            "activations and scales"
        )
    expected_m = min(x.shape[1], dispatch_output.expected_m)
    grouped_gemm = _masked_gemm
    w13_weight = getattr(self, "w13_weight_low_latency", self.w13_weight)
    w2_weight = getattr(self, "w2_weight_low_latency", self.w2_weight)
    gate_up = torch.empty(
        (x.shape[0], x.shape[1], self.w13_weight_scale.shape[1]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    grouped_gemm(
        x,
        scale,
        w13_weight,
        self.w13_weight_scale,
        gate_up,
        masked_m,
        expected_m,
    )
    swiglu_limit = self.moe_runner_config.swiglu_limit
    with record_function("dsv4_routed_expert_silu_mul_int8_fused_masked"):
        if swiglu_limit is None:
            a2, a2_scale = fuse_silu_mul_quant_ep(
                gate_up, tokens_per_expert=masked_m, expect_m=expected_m
            )
        else:
            a2, a2_scale = fuse_silu_mul_clamp_quant_ep(
                input=gate_up,
                limit=swiglu_limit,
                mask_m=masked_m,
                expect_m=expected_m,
            )
    del gate_up
    padded_size = getattr(
        self, "w4a8_ll_padded_intermediate_size", self.w4a8_padded_intermediate_size
    )
    if a2.shape[-1] != padded_size:
        a2 = F.pad(a2, (0, padded_size - a2.shape[-1]))
    output = torch.empty(
        (x.shape[0], x.shape[1], self.w2_weight_scale.shape[1]),
        dtype=torch.bfloat16,
        device=x.device,
    )
    grouped_gemm(
        a2,
        a2_scale,
        w2_weight,
        self.w2_weight_scale,
        output,
        masked_m,
        expected_m,
    )
    return output


def _masked_gemm(a, a_scale, weight, weight_scale, output, masked_m, expected_m):
    from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc

    m_grouped_w4a8_gemm_nt_masked_hipc(
        (a, a_scale), (weight, weight_scale), output, masked_m, expected_m
    )
    return output
