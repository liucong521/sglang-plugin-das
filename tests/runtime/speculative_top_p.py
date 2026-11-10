from __future__ import annotations

import os
import unittest

if os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1":
    raise unittest.SkipTest("HCU_PLUGIN_RUN_GPU_TESTS=1 is required")

import torch
from sglang.test.test_utils import CustomTestCase
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.speculative import dflash_utils as N
from hcu_sglang.utils.detection import is_hcu


class SpeculativeTopPTests(CustomTestCase):
    def test_registered_HCU_top_p_matches_threshold_reference(self):
        self.assertTrue(is_hcu())
        target = "sglang.srt.speculative.dflash_utils._dflash_top_p_renorm_prob"
        self.assertIn(target, HookRegistry._patched)
        for vocab in (7, 1025, 129280):
            probs = torch.softmax(torch.randn(6, vocab, device="cuda"), dim=-1)
            top_ps = torch.full((6,), .95, device="cuda")
            cpu = probs.cpu()
            ordered = cpu.sort(dim=-1).values
            cutoff = torch.searchsorted(ordered.cumsum(-1), torch.full((6, 1), .05))
            pivots = ordered.gather(1, cutoff.clamp(max=vocab-1))
            expected = cpu.masked_fill(cpu < pivots, 0)
            expected /= expected.sum(-1, keepdim=True)
            actual = N._dflash_top_p_renorm_prob(probs, top_ps)
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-7)

    def test_graph_replay_reads_changed_probabilities_and_top_p(self):
        probs = torch.softmax(torch.randn(6, 1025, device="cuda"), dim=-1)
        top_ps = torch.full((6,), .95, device="cuda")
        for _ in range(3):
            N._dflash_top_p_renorm_prob(probs, top_ps)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = N._dflash_top_p_renorm_prob(probs, top_ps)
        for p in (.8, .95):
            probs.copy_(torch.softmax(torch.randn_like(probs), dim=-1))
            top_ps.fill_(p)
            expected = N._dflash_top_p_renorm_prob(probs, top_ps)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
