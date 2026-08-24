from __future__ import annotations

import unittest

from bnhmm.cli import build_parser


class SectorFlowCliTests(unittest.TestCase):
    def test_sector_flow_command_exposes_only_relevant_model_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "sector-flow-walk-forward",
                "--duckdb-path",
                "sector.duckdb",
                "--symbols",
                "A,B,C",
                "--start",
                "2024-01-01",
                "--end",
                "2026-01-01",
                "--output-dir",
                "run",
                "--normalization-lookback-sessions",
                "20",
                "--maximum-lag-minutes",
                "30",
                "--factor-process-variance",
                "0.2",
                "--top-k",
                "10",
                "--protocol",
                "river-topology-v2",
                "--topology-lookback-sessions",
                "60",
            ]
        )

        self.assertEqual(args.command, "sector-flow-walk-forward")
        self.assertEqual(args.maximum_lag_minutes, 30)
        self.assertEqual(args.factor_process_variance, 0.2)
        self.assertEqual(args.protocol, "river-topology-v2")
        self.assertEqual(args.topology_lookback_sessions, 60)
        self.assertFalse(hasattr(args, "epochs"))
        self.assertFalse(hasattr(args, "d_model"))


if __name__ == "__main__":
    unittest.main()
