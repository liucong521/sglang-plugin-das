# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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

from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import (
    HCUW4A8Int8DeepEPMoEMethod,
)


class HCUCompressedW4A8DeepEPMoEMethod(HCUW4A8Int8DeepEPMoEMethod):
    def process_weights_after_loading(self, layer):
        from deepgemm import (
            m_grouped_w4a8_gemm_nt_contiguous_hipc,
            m_grouped_w4a8_gemm_nt_masked_hipc,
            pack_w4a8_moe_hipc_weight,
        )

        if not all(callable(f) for f in (
            m_grouped_w4a8_gemm_nt_contiguous_hipc,
            m_grouped_w4a8_gemm_nt_masked_hipc,
            pack_w4a8_moe_hipc_weight,
        )):
            raise RuntimeError("HCU W4A8 DeepEP requires the HIPC provider")
        w13 = self._convert_packed_weight(layer.w13_weight_packed.data)
        w2 = self._convert_packed_weight(layer.w2_weight_packed.data)
        intermediate_size = w2.shape[-1] * 2
        padded_size = ((intermediate_size + 127) // 128) * 128
        layer.w4a8_intermediate_size = intermediate_size
        layer.w4a8_padded_intermediate_size = padded_size
        layer.w4a8_ll_padded_intermediate_size = padded_size
        if padded_size != intermediate_size:
            w2 = F.pad(w2, (0, (padded_size - intermediate_size) // 2))
        for prefix, weight in (("w13", w13), ("w2", w2)):
            packed = pack_w4a8_moe_hipc_weight(weight)
            layer.register_parameter(
                prefix + "_weight", torch.nn.Parameter(packed, requires_grad=False)
            )
            delattr(layer, prefix + "_weight_packed")
            name = prefix + "_weight_scale"
            scale = self._channel_scale(getattr(layer, name).data)
            setattr(layer, name, torch.nn.Parameter(scale, requires_grad=False))
        layer.dispatcher.set_quant_config({
            "normal_dispatcher_output_dtype": "int8",
            "low_latency_dispatcher_output_dtype": "int8",
            "normal_expert_alignment": 256,
            "hcu_w4a8_int8_dispatch": True,
            "hcu_w4a8_deepep_ll_producer_event": True,
        })
