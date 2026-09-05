from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F


def quest_score_reference(queries, page_min, page_max, mapping, pages):
    mins = page_min[:, :pages].index_select(0, mapping)
    maxs = page_max[:, :pages].index_select(0, mapping)
    query = queries.float()[:, None]
    return torch.maximum(query * mins, query * maxs).sum(dim=-1)


def quest_indices_reference(scores, *, num_tokens, page_size, budget):
    page_budget = min(int(scores.shape[-1]), max(1, math.ceil(budget / page_size)))
    pages = torch.topk(scores, k=page_budget, dim=-1, sorted=False).indices
    offsets = torch.arange(page_size, device=scores.device)
    indices = (pages[:, :, None] * page_size + offsets).flatten(1)
    valid = indices.lt(num_tokens)
    current = num_tokens - 1
    has_current = (indices.eq(current) & valid).any(dim=-1)
    indices[:, -1] = torch.where(has_current, indices[:, -1], indices.new_tensor(current))
    valid[:, -1] = torch.where(has_current, valid[:, -1], valid.new_tensor(True))
    return indices, valid


def materialized_attention(queries, keys, values, current_keys, current_values,
                           indices, mapping, valid=None):
    prefix_tokens = int(keys.shape[1])
    safe = indices.clamp_max(prefix_tokens - 1)
    kv_heads = mapping[:, None].expand_as(safe)
    selected_keys = keys[kv_heads, safe]
    selected_values = values[kv_heads, safe]
    current_mask = indices.eq(prefix_tokens).unsqueeze(-1)
    selected_keys = torch.where(
        current_mask, current_keys.index_select(0, mapping)[:, None], selected_keys,
    )
    selected_values = torch.where(
        current_mask, current_values.index_select(0, mapping)[:, None], selected_values,
    )
    return F.scaled_dot_product_attention(
        queries[None, :, None], selected_keys[None], selected_values[None],
        attn_mask=None if valid is None else valid[None, :, None],
        dropout_p=0.0, is_causal=False,
    )[0, :, 0]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparseKernelTests(unittest.TestCase):
    def test_quest_score_matches_reference_with_capacity_tail(self) -> None:
        from sparse_kernels import quest_scores

        torch.manual_seed(101)
        for dim in (64, 128):
            for kv_heads in (1, 2, 8):
                queries = torch.randn(8, dim, device="cuda", dtype=torch.bfloat16)
                mins = torch.randn(kv_heads, 23, dim, device="cuda", dtype=torch.float32)
                maxs = mins + torch.rand_like(mins)
                mapping = torch.arange(8, device="cuda") // (8 // kv_heads)
                expected = quest_score_reference(queries, mins, maxs, mapping, pages=17)
                actual = quest_scores(queries, mins, maxs, mapping, pages=17)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)

    def test_quest_selection_matches_reference_at_page_boundaries(self) -> None:
        from sparse_kernels import quest_indices

        torch.manual_seed(102)
        scores = torch.randn(8, 17, device="cuda")
        for num_tokens in (127, 128, 131):
            expected = quest_indices_reference(
                scores, num_tokens=num_tokens, page_size=8, budget=24,
            )
            actual = quest_indices(
                scores, num_tokens=num_tokens, page_size=8, budget=24,
            )
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_indexed_attention_matches_materialized_reference(self) -> None:
        from sparse_kernels import indexed_attention

        torch.manual_seed(103)
        q_heads, kv_heads, tokens, dim = 8, 2, 37, 64
        queries = torch.randn(q_heads, dim, device="cuda", dtype=torch.bfloat16)
        key_storage = torch.randn(
            kv_heads, tokens + 11, dim, device="cuda", dtype=torch.bfloat16,
        )
        value_storage = torch.randn_like(key_storage)
        keys, values = key_storage[:, :tokens], value_storage[:, :tokens]
        current_keys = torch.randn(kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        current_values = torch.randn_like(current_keys)
        mapping = torch.arange(q_heads, device="cuda") // 4
        indices = torch.randint(tokens, (q_heads, 19), device="cuda")
        indices[:, -1] = tokens

        expected = materialized_attention(
            queries, keys, values, current_keys, current_values, indices, mapping,
        )
        actual = indexed_attention(
            queries, keys, values, current_keys, current_values, indices, mapping,
        )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

        valid = torch.ones_like(indices, dtype=torch.bool)
        valid[:, -4:-1] = False
        expected = materialized_attention(
            queries, keys, values, current_keys, current_values, indices, mapping, valid,
        )
        actual = indexed_attention(
            queries, keys, values, current_keys, current_values, indices, mapping, valid,
        )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


    def test_split_indexed_attention_matches_materialized_reference(self) -> None:
        from sparse_kernels import indexed_attention_split

        torch.manual_seed(104)
        q_heads, kv_heads, tokens, dim = 8, 2, 32768, 128
        queries = torch.randn(q_heads, dim, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(kv_heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        current_keys = torch.randn(kv_heads, dim, device="cuda", dtype=torch.bfloat16)
        current_values = torch.randn_like(current_keys)
        mapping = torch.arange(q_heads, device="cuda") // 4
        indices = torch.randint(tokens, (q_heads, 4096), device="cuda")
        indices[:, -1] = tokens
        valid = torch.ones_like(indices, dtype=torch.bool)
        valid[:, -7:-1] = False
        expected = materialized_attention(
            queries, keys, values, current_keys, current_values, indices, mapping, valid,
        )
        actual = indexed_attention_split(
            queries, keys, values, current_keys, current_values, indices, mapping, valid,
            split_tokens=256,
        )
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
