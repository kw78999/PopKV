from __future__ import annotations

import unittest

from experiments.benchmarks.benchmark_encoding import DEFAULT_CONTEXTS, OCCUPANCIES, arguments, benchmark_cases, expected_shapes
from experiments.benchmarks.benchmark_dimension_groups import arguments as group_arguments, cases as group_cases


class EncodingBenchmarkTests(unittest.TestCase):
    def test_cli_defaults_and_shapes(self) -> None:
        args = arguments([])
        self.assertEqual(args.contexts, DEFAULT_CONTEXTS)
        self.assertEqual((args.q_heads, args.kv_heads, args.head_dim, args.layers), (32, 8, 128, 32))
        shapes = expected_shapes(args)
        self.assertEqual(shapes["query"], [1, 32, 128])
        self.assertEqual(shapes["query_bitplane_each"], [1, 32, 4])
        self.assertEqual(shapes["query_representative_each"], [1, 32, 4])
        self.assertEqual(args.token_group_size, 32)
        self.assertEqual(shapes["packed_key_by_context"]["65536"], [8, 65536, 4])
        self.assertEqual(shapes["key_representative_by_context"]["65536"], [8, 65536, 4])

    def test_case_generator_covers_contexts_and_occupancies(self) -> None:
        cases = benchmark_cases(DEFAULT_CONTEXTS, 32)
        prefix = {x.context_length for x in cases if x.operation == "key_prefix_encode"}
        incremental = {x.active_group_occupancy for x in cases if x.operation == "key_incremental_encode"}
        scopes = {x.scope for x in cases}
        self.assertEqual(prefix, set(DEFAULT_CONTEXTS))
        self.assertEqual(incremental, set(OCCUPANCIES))
        self.assertEqual(incremental, set(range(1, 33)))
        self.assertIn("30_sparse_layers", scopes)
        small = benchmark_cases(DEFAULT_CONTEXTS, 32, token_group_size=8)
        small_incremental = {x.active_group_occupancy for x in small if x.operation == "key_incremental_encode"}
        self.assertEqual(small_incremental, set(range(1, 9)))

        self.assertIn("32_layers", scopes)

    def test_group_size_benchmark_cases(self) -> None:
        args = group_arguments([])
        generated = group_cases(args)
        self.assertEqual(len(generated), 16)
        self.assertEqual({x["dim_group_size"] for x in generated}, {16, 32, 64})
        self.assertEqual({x["dim_kernel"] for x in generated},
                         {"g16", "g32", "g64-u32", "g64-u64"})
        self.assertEqual({x["context_length"] for x in generated}, set(DEFAULT_CONTEXTS))


if __name__ == "__main__":
    unittest.main()
