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
            "sector_index_symbol",
            "regime_context_sessions",
            "regime_hidden_channels",
            "regime_latent_states",
            "regime_neutral_sigma",
            "regime_confidence_threshold",
            "regime_maximum_bull_variance",
            "regime_epochs",
            "regime_batch_size",
            "regime_learning_rate",
            "regime_kl_weight",
            "regime_state_loss_weight",
            "regime_mc_samples",
            "sector_flow_top_k",
        ):
            self.assertIn(f'name="{field}"', html)
        for identifier in (
            "sector-flow-results",
            "sector-flow-water-state",
            "sector-flow-ranking-table",
            "sector-flow-baseline-table",
            "sector-flow-result-status",
            "sector-regime-results",
            "sector-regime-probabilities",
            "sector-regime-baselines",
            "sector-regime-policy",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn('role="status"', html)
        self.assertIn("renderSectorFlowResult", script)
        self.assertIn('"sector-flow-bayes"', script)
        self.assertIn("renderSectorRegimeResult", script)

    def test_river_workspace_has_next_session_forecast_view(self) -> None:
        html = (ROOT / "river.html").read_text(encoding="utf-8")
        script = (ROOT / "river.js").read_text(encoding="utf-8")

        self.assertIn('id="forecast-view"', html)
        self.assertIn('id="forecast-canvas"', html)
        self.assertIn("loadForecast", script)
        self.assertIn("renderForecast", script)


if __name__ == "__main__":
    unittest.main()
