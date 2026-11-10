# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from array import array
from collections import defaultdict
import torch
from sglang.srt.managers.mm_utils import assert_int64_array, Modality

def pad_mm_tokens(input_ids: array, mm_inputs: MultimodalInputs) -> array:
    """Pad inclusive per-item offsets without mutating the input token array."""
    assert_int64_array(input_ids, "input_ids")
    if not input_ids or not mm_inputs.mm_items:
        return input_ids

    # as_tensor(array) walks and boxes every token. View a bulk copy instead,
    # keeping the original request's unpadded token buffer unchanged.
    padded_input_ids = input_ids[:]
    input_ids_tensor = torch.frombuffer(padded_input_ids, dtype=torch.int64)

    # Replace multimodal tokens using per-item offsets
    items_by_modality = defaultdict(list)
    for item in mm_inputs.mm_items:
        items_by_modality[item.modality].append(item)

    token_id_map = {
        Modality.IMAGE: mm_inputs.im_token_id,
        Modality.AUDIO: mm_inputs.audio_token_id,
        Modality.VIDEO: mm_inputs.video_token_id,
    }

    for modality, items in items_by_modality.items():
        token_id = token_id_map.get(modality)

        if not items or token_id is None:
            continue

        for i, item in enumerate(items):
            for offset in items[i].offsets:
                input_ids_tensor[offset[0] : offset[1] + 1] = item.pad_value

    return padded_input_ids
