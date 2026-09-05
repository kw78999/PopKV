import unittest

from experiments.estimate_resource_costs import ModelShape, PROFILES, estimate


class ResourceCostTests(unittest.TestCase):
    def test_popcount_is_separate_and_only_popkv_uses_it(self):
        shape = ModelShape()
        rows = {p.label: estimate(p, shape, 32768, 128, 4096, 2)[1] for p in PROFILES}
        self.assertEqual(rows["Full"]["popcount_ops"], 0)
        self.assertEqual(rows["QUEST"]["popcount_ops"], 0)

        self.assertGreater(rows["PopKV-Q"]["popcount_ops"], rows["PopKV-S"]["popcount_ops"])
        self.assertEqual(rows["FIER"]["popcount_ops"], 0)

    def test_public_profiles_include_normal_ablation(self):
        self.assertEqual(
            tuple(profile.label for profile in PROFILES),
            ("Full", "QUEST", "FIER", "PopKV-Normal", "PopKV-Q", "PopKV-S"),
        )

    def test_sparse_metadata_is_additional_to_full_kv(self):
        shape = ModelShape()
        for profile in PROFILES:
            _, row = estimate(profile, shape, 32768, 128, 4096, 2)
            self.assertGreaterEqual(row["total_cache_vs_full_kv"], 1.0)
            self.assertEqual(row["total_cache_bytes"], row["base_kv_bytes"] + row["metadata_bytes"])

    def test_quest_prepack_records_non_flop_work(self):
        shape = ModelShape()
        profile = next(x for x in PROFILES if x.label == "QUEST")
        stages, row = estimate(profile, shape, 32768, 256, 4096, 2)
        self.assertEqual(stages["prepack"].flops, 0)
        self.assertGreater(stages["prepack"].comparisons, 0)
        self.assertGreater(stages["prepack"].conversions, 0)
        self.assertNotIn("metadata_vs_bf16_k_ratio", row)

    def test_all_large_stages_are_reported(self):
        shape = ModelShape()
        stages, _ = estimate(PROFILES[0], shape, 32768, 256, 4096, 2)
        self.assertEqual(
            set(stages),
            {"prepack", "frontend", "qkv_rope", "retrieval_score", "topk_selection",
             "gather", "attention", "output_projection", "mlp", "final_norm_lm_head",
             "token_selection", "kv_update", "retrieval_update"},
        )

    def test_online_cost_excludes_prepack(self):
        shape = ModelShape()
        for profile in PROFILES:
            _, row = estimate(profile, shape, 32768, 128, 4096, 2)
            self.assertEqual(row["total_flops"], row["online_flops"] + row["prepack_flops"])


if __name__ == "__main__":
    unittest.main()
