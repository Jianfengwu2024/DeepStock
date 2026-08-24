from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "bnhmm" / "web" / "static"


class SectorFlowStaticContractTests(unittest.TestCase):
    def test_river_workspace_has_four_views_and_stock_inspector(self) -> None:
        html = (ROOT / "river.html").read_text(encoding="utf-8")
        script = (ROOT / "river.js").read_text(encoding="utf-8")

        for identifier in (
            "river-heatmap",
            "topology-view",
            "migration-view",
            "structure-view",
            "stock-inspector",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn("loadTopology", script)
        self.assertIn("loadSlice", script)
        self.assertIn("renderStockInspector", script)

    def test_sector_flow_result_links_selected_job_to_river_workspace(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="open-river-view"', html)
        self.assertIn("/jobs/${encodeURIComponent(job.id)}/river", script)

    def test_sector_flow_configuration_and_results_have_stable_hooks(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('value="sector-flow-bayes"', html)
        for field in (
            "normalization_minimum_observations",
            "maximum_lag_minutes",
            "minimum_attribution_rows",
            "factor_process_variance",
            "sector_flow_ranking_window",
            "sector_flow_evaluation_horizon_minutes",
            "sector_flow_top_k",
            "sector_flow_monte_carlo_samples",
        ):
            self.assertIn(f'name="{field}"', html)
        for identifier in (
            "sector-flow-results",
            "sector-flow-water-state",
            "sector-flow-ranking-table",
            "sector-flow-baseline-table",
            "sector-flow-result-status",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn('role="status"', html)
        self.assertIn("renderSectorFlowResult", script)
        self.assertIn('"sector-flow-bayes"', script)


if __name__ == "__main__":
    unittest.main()
