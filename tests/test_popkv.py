from __future__ import annotations

import math
import unittest

import torch

from popkv import (
    METHODS, RetrievalConfig, append_quest, build_quest_cache, append_fier, build_fier_cache,
    _popkv_reference_scores_g32_legacy, _reference_query_encode_g32_legacy,
    popkv_reference_bitwise_scores, popkv_reference_scores, quest_indices, quest_scores,
    reference_key_encode, reference_query_encode, reference_query_encode_g32,
    DecodeDiagnostics, _DecodeStageTimer, _DECODE_STAGE_FIELDS, _append_current,
    _sparse_layer_indices,
)


class PopKVTests(unittest.TestCase):
    def test_exhaustive_stage_timer_closes_to_total(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        timer = _DecodeStageTimer(device)
        for name in _DECODE_STAGE_FIELDS:
            if device.type == "cuda": torch.ones(32, device=device).add_(1)
            else: sum(range(32))
            timer.mark(name)
        diagnostics = DecodeDiagnostics(); timer.finish(diagnostics)
        self.assertAlmostEqual(
            diagnostics.gpu_component_sum_ms + diagnostics.gpu_unattributed_ms,
            diagnostics.gpu_decode_ms, places=5,
        )
        self.assertAlmostEqual(
            diagnostics.candidate_search_ms,
            diagnostics.retrieval_stage_ms + diagnostics.candidate_score_ms +
            diagnostics.candidate_topk_ms, places=7,
        )

    def test_public_methods_are_only_paper_methods(self) -> None:
        self.assertEqual(METHODS, ("full", "quest", "fier", "popkv"))

    def test_sparse_methods_skip_configured_full_layers(self) -> None:
        self.assertEqual(_sparse_layer_indices(4, (0, 1)), (2, 3))

    def test_current_token_is_always_reserved(self) -> None:
        selected = torch.tensor([[4, 7], [2, 9]])
        actual = _append_current(selected, 12)
        self.assertTrue(torch.equal(actual[:, -1], torch.tensor([11, 11])))

    def test_release_configuration_is_explicit(self) -> None:
        RetrievalConfig().validate()
        with self.assertRaises(ValueError):
            RetrievalConfig(group_size=64).validate()
        for size in (16, 32, 64):
            RetrievalConfig(dim_group_size=size).validate()
        with self.assertRaises(ValueError):
            RetrievalConfig(dim_group_size=8).validate()

    def test_quest_cache_matches_direct_page_bounds_across_boundary(self) -> None:
        torch.manual_seed(3)
        keys = torch.randn(8, 31, 16, dtype=torch.bfloat16)
        mapping = torch.arange(32) // 4
        cache = build_quest_cache(keys, page_size=8, reserve_tokens=8)
        for _ in range(4):
            queries = torch.randn(32, 16, dtype=torch.bfloat16)
            current = torch.randn(8, 16, dtype=torch.bfloat16)
            cached = quest_scores(queries, cache, mapping)
            all_keys = keys.float()
            pad = (-all_keys.shape[1]) % 8
            if pad:
                all_keys = torch.cat([all_keys, all_keys[:, -1:].expand(8, pad, 16)], dim=1)
            pages = all_keys.view(8, math.ceil(keys.shape[1] / 8), 8, 16)
            mins, maxs = pages.amin(2).index_select(0, mapping), pages.amax(2).index_select(0, mapping)
            direct = torch.maximum(queries.float()[:, None] * mins, queries.float()[:, None] * maxs).sum(-1)
            self.assertTrue(torch.equal(cached, direct))
            left = quest_indices(cached, num_tokens=keys.shape[1] + 1, page_size=8, budget=16)
            right = quest_indices(direct, num_tokens=keys.shape[1] + 1, page_size=8, budget=16)
            self.assertTrue(torch.equal(left[0], right[0]) and torch.equal(left[1], right[1]))
            self.assertTrue((left[0].eq(keys.shape[1]) & left[1]).any(dim=-1).all())
            append_quest(cache, current[:, None]); keys = torch.cat([keys, current[:, None]], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_quest_cuda_update_matches_direct_active_page_bounds(self) -> None:
        torch.manual_seed(5)
        keys = torch.randn(2, 15, 64, device="cuda", dtype=torch.bfloat16)
        cache = build_quest_cache(keys, page_size=8, reserve_tokens=4)
        for _ in range(3):
            current = torch.randn(2, 1, 64, device="cuda", dtype=torch.bfloat16)
            append_quest(cache, current); keys = torch.cat([keys, current], dim=1)
            page = (keys.shape[1] - 1) // 8
            expected = keys[:, page * 8:].float()
            torch.testing.assert_close(cache.page_min[:, page], expected.amin(dim=1), atol=0, rtol=0)
            torch.testing.assert_close(cache.page_max[:, page], expected.amax(dim=1), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fier_post_update_keeps_active_group_searchable(self) -> None:
        from fier_triton import score_packed_batched, score_tensors
        torch.manual_seed(7)
        keys = torch.randn(2, 31, 64, device="cuda", dtype=torch.bfloat16)
        mapping = torch.arange(8, device="cuda") // 4
        cache = build_fier_cache(keys, group_size=32, reserve_tokens=4)
        for _ in range(3):
            current = torch.randn(2, 1, 64, device="cuda", dtype=torch.bfloat16)
            append_fier(cache, current); keys = torch.cat([keys, current], dim=1)
            query = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
            cached = score_packed_batched(query, cache.packed, cache.group_min, cache.group_max,
                mapping, tokens=keys.shape[1], group_size=32)
            direct = score_tensors(query, keys.contiguous(), head_to_kv=mapping, group_size=32)
            torch.testing.assert_close(cached, direct, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_popkv_cuda_matches_readable_reference(self) -> None:
        from popkv_cuda import pack_keys, score
        torch.manual_seed(11)
        keys = torch.randn(2, 37, 64, device="cuda", dtype=torch.bfloat16)
        queries = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
        mapping = torch.arange(8, device="cuda") // 4
        expected = popkv_reference_scores(queries, keys, head_to_kv=mapping)
        actual = score(queries.contiguous(), pack_keys(keys.contiguous()), head_to_kv=mapping, tokens=37)
        torch.testing.assert_close(actual, expected, atol=0.12, rtol=0.02)

    def test_reference_query_encode_shapes_dtypes_and_invariants(self) -> None:
        cases = (
            torch.ones(2, 3, 32) * 2,
            torch.ones(2, 3, 64) * -2,
            torch.linspace(-4, 4, 2 * 3 * 96).view(2, 3, 96),
            torch.zeros(2, 3, 127),
            torch.tensor([-2.0, -2.0, 0.0, 2.0]).repeat(2 * 3 * 32).view(2, 3, 128),
        )
        for query in cases:
            sign, magnitude, low, delta = reference_query_encode_g32(query)
            expected = (2, 3, math.ceil(query.shape[-1] / 32))
            self.assertEqual(sign.shape, expected)
            self.assertEqual(magnitude.shape, expected)
            self.assertEqual(low.shape, expected)
            self.assertEqual(delta.shape, expected)
            self.assertEqual(sign.dtype, torch.int32)
            self.assertEqual(magnitude.dtype, torch.int32)
            self.assertEqual(low.dtype, torch.float32)
            self.assertEqual(delta.dtype, torch.float32)
            self.assertTrue(torch.isfinite(low).all())
            self.assertTrue(torch.isfinite(delta).all())
            self.assertTrue((delta >= 0).all())

    def test_reference_query_encode_hand_computable_case(self) -> None:
        query = torch.tensor([[[1.0, -1.0, 3.0, -3.0, 0.0]]])
        sign, magnitude, low, delta = reference_query_encode_g32(query)
        self.assertEqual(int(sign.item()), 0b10101)
        self.assertEqual(int(magnitude.item()), 0b01100)
        torch.testing.assert_close(low, torch.tensor([[[2.0 / 3.0]]]))
        torch.testing.assert_close(delta, torch.tensor([[[7.0 / 3.0]]]))

    def test_reference_query_encode_empty_bucket_behavior(self) -> None:
        for value in (2.0, -2.0):
            _, magnitude, low, delta = reference_query_encode_g32(torch.full((1, 1, 32), value))
            self.assertEqual(int(magnitude.item()), -1)  # all 32 bits set
            torch.testing.assert_close(low, torch.tensor([[[2.0]]]))
            torch.testing.assert_close(delta, torch.zeros(1, 1, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_query_encode_matches_reference(self) -> None:
        from popkv_cuda import query_encode
        torch.manual_seed(17)
        for group in (16, 32, 64):
            for dim in (32, 64, 96, 127, 128):
                query = torch.randn(1, 8, dim, device="cuda", dtype=torch.bfloat16)
                expected = reference_query_encode(query, group)
                actual = query_encode(query.contiguous(), dim_group_size=group)
                torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
                torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
                torch.testing.assert_close(actual[2], expected[2], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual[3], expected[3], atol=2e-5, rtol=2e-5)

    def test_reference_layouts_for_all_dimension_groups_and_tails(self) -> None:
        torch.manual_seed(23)
        data_cases = [
            torch.randn(2, 3, 128), torch.ones(2, 3, 128), -torch.ones(2, 3, 128),
            torch.zeros(2, 3, 128), torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]).repeat(2 * 3 * 128 // 5 + 1)[:2 * 3 * 128].view(2, 3, 128),
        ]
        for group in (16, 32, 64):
            for dim in (32, 64, 96, 127, 128):
                for base in data_cases:
                    query = base[..., :dim].contiguous()
                    sign, mag, low, delta = reference_query_encode(query, group)
                    self.assertEqual(sign.shape[-1], math.ceil(dim / 32))
                    self.assertEqual(mag.shape, sign.shape)
                    self.assertEqual(low.shape[-1], math.ceil(dim / group))
                    self.assertEqual(delta.shape, low.shape)
                    self.assertTrue(torch.isfinite(low).all() and torch.isfinite(delta).all())
                keys = torch.randn(2, 37, dim)
                ksign, kmag, klow, kdelta = reference_key_encode(keys, dim_group_size=group)
                self.assertEqual(ksign.shape, (2, 37, math.ceil(dim / 32)))
                self.assertEqual(kmag.shape, ksign.shape)
                self.assertEqual(klow.shape, (2, 37, math.ceil(dim / group)))
                self.assertEqual(kdelta.shape, klow.shape)

    def test_g32_generalized_reference_matches_frozen_release_oracle(self) -> None:
        torch.manual_seed(29)
        for dim in (32, 64, 96, 127, 128):
            query = torch.randn(2, 8, dim)
            generalized = reference_query_encode(query, 32)
            legacy = _reference_query_encode_g32_legacy(query)
            for left, right in zip(generalized, legacy):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            keys = torch.randn(2, 37, dim)
            mapping = torch.arange(8) // 4
            score = popkv_reference_scores(query, keys, head_to_kv=mapping, dim_group_size=32)
            legacy_score = _popkv_reference_scores_g32_legacy(query, keys, head_to_kv=mapping)
            torch.testing.assert_close(score, legacy_score, atol=2e-6, rtol=2e-6)

    def test_popcount_decomposition_matches_direct_reconstruction(self) -> None:
        query = torch.tensor([[[3., -1., 0., -3., 1., -2., 2., 0.]]]).repeat(1, 1, 8)
        keys = torch.stack((query[0], -query[0]), dim=1)
        mapping = torch.tensor([0])
        for group in (16, 32, 64):
            direct = popkv_reference_scores(
                query, keys, head_to_kv=mapping, dim_group_size=group,
            )
            bitwise = popkv_reference_bitwise_scores(
                query, keys, head_to_kv=mapping, dim_group_size=group,
            )
            torch.testing.assert_close(bitwise, direct, atol=2e-6, rtol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_key_encoding_and_scores_for_all_dimension_groups(self) -> None:
        from popkv_cuda import pack_keys, score
        torch.manual_seed(31)
        mapping = torch.arange(8, device="cuda") // 4
        for group in (16, 32, 64):
            for dim in (32, 64, 96, 127, 128):
                keys = torch.randn(2, 37, dim, device="cuda", dtype=torch.bfloat16)
                query = torch.randn(1, 8, dim, device="cuda", dtype=torch.bfloat16)
                packed = pack_keys(keys.contiguous(), dim_group_size=group)
                expected_packed = reference_key_encode(keys, dim_group_size=group)
                torch.testing.assert_close(packed[0], expected_packed[0], atol=0, rtol=0)
                torch.testing.assert_close(packed[1], expected_packed[1], atol=0, rtol=0)
                torch.testing.assert_close(packed[2].float(), expected_packed[2], atol=2e-3, rtol=2e-3)
                torch.testing.assert_close(packed[3].float(), expected_packed[3], atol=2e-3, rtol=2e-3)
                actual = score(query.contiguous(), packed, head_to_kv=mapping, tokens=37, dim_group_size=group)
                expected = popkv_reference_scores(query, keys, head_to_kv=mapping, dim_group_size=group)
                torch.testing.assert_close(actual, expected, atol=0.15, rtol=0.025)


if __name__ == "__main__": unittest.main()
