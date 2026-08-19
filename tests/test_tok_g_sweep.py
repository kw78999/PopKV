from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.sweep_token_groups import arguments, duration, token_groups, write_summary, read_completed


class TokenGroupSweepTests(unittest.TestCase):
    def test_default_plan_has_exactly_250_cases_and_two_blocks(self) -> None:
        args = arguments(["--dry-run"])
        groups = token_groups(args)
        self.assertEqual(len(groups), 250)
        self.assertEqual(groups[0], 2)
        self.assertEqual(groups[-1], 500)
        self.assertEqual(args.generate_tokens, 500)
        self.assertEqual(args.num_blocks, 2)

    def test_progress_duration_format(self) -> None:
        self.assertEqual(duration(0), "00:00:00")
        self.assertEqual(duration(3661), "01:01:01")

    def test_summary_checkpoint_round_trip(self) -> None:
        row = {
            "case": 1, "tok_g": 5, "tokens": 500, "perplexity": 8.5,
            "mean_nll": 2.1, "key_metadata_encoding_ms_total": 3.0,
            "key_metadata_encoding_ms_per_token": 0.006,
            "retrieval_group_sealing_ms_total": 1.0,
            "retrieval_group_sealing_ms_per_token": 0.002,
            "prepack_ms": 10.0, "case_elapsed_seconds": 60.0,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "completed_at_utc": "2026-01-01T00:01:00Z",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            write_summary(path, [row])
            self.assertEqual(read_completed(path), [row])


if __name__ == "__main__":
    unittest.main()
