from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "bnhmm" / "web" / "static"


class SectorFlowStaticContractTests(unittest.TestCase):
    def test_strategy_selector_contains_only_bnhmm_and_bayesian_fk(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")

        self.assertEqual(html.count('name="strategy"'), 2)
        self.assertIn('value="bnhmm"', html)
        self.assertIn('value="boundary-flow-fk"', html)
        for removed in (
            "holo-bnhmm",
            "holo-repair",
            "stock-transformer",
            "fk-stock",
            "sector-flow-bayes",
            "percolation-branching",
        ):
            self.assertNotIn(removed, html)

    def test_mobile_header_keeps_actions_compact(self) -> None:
        styles = (ROOT / "styles.css").read_text(encoding="utf-8")

        mobile = styles.split("@media (max-width: 760px)", 1)[1]
        self.assertIn(".workspace-header > div:first-child { min-width: 0; flex: 1; }", mobile)
        self.assertIn(
            "#job-subtitle { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
            mobile,
        )
        self.assertIn(".gpu-monitor { display: none; }", mobile)

    def test_history_cards_cannot_overflow_the_history_column(self) -> None:
        styles = (ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn(
            ".job-list { display: grid; grid-template-columns: minmax(0, 1fr); gap: 5px; }",
            styles,
        )
        self.assertIn("min-width: 0; max-width: 100%;", styles)
        self.assertIn("overflow: hidden;", styles)

    def test_removed_branch_window_comparison_is_not_exposed(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        for identifier in ("run-window-comparison", "comparison-list", "comparison-view"):
            self.assertNotIn(f'id="{identifier}"', html)
        self.assertNotIn("submitWindowComparison", script)
        self.assertNotIn("boundary_cache_lookbacks", script)

    def test_history_management_actions_stay_inside_history_panel(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        workspace_header = html.split(
            '<header class="workspace-header">', 1
        )[1].split("</header>", 1)[0]
        history_panel = html.split('<section class="run-history"', 1)[1].split(
            "</section>", 1
        )[0]

        self.assertNotIn('id="edit-history-button"', workspace_header)
        self.assertIn('id="edit-history-button"', history_panel)
        self.assertIn('id="batch-delete-button"', history_panel)
        self.assertLess(
            history_panel.index('id="edit-history-button"'),
            history_panel.index('id="history-batch-actions"'),
        )
        self.assertLess(
            history_panel.index('id="batch-delete-button"'),
            history_panel.index('id="history-batch-actions"'),
        )
        self.assertNotIn('id="delete-button"', html)
        self.assertNotIn("function deleteSelectedJob(", script)

    def test_history_cards_prefer_the_completed_summary_symbol_count(self) -> None:
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn(
            "job.summary?.symbols ?? job.request.symbols.length",
            script,
        )

    def test_strategy_switch_excludes_hidden_configuration_controls(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("function syncStrategyControlAvailability()", script)
        self.assertIn("function namedFormControls(form, name)", script)
        self.assertIn("function preferredFormControl(form, name)", script)
        self.assertIn('form.querySelectorAll("[name]")', script)
        self.assertIn('element.closest(".hidden")', script)
        self.assertIn("const control = preferredFormControl(form, name);", script)
        self.assertIn('id="bnhmm-fields"', html)
        self.assertIn('id="boundary-fields"', html)

    def test_boundary_flow_configuration_and_results_have_stable_hooks(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('/static/app.js?v=0.29.1', html)
        self.assertIn('/static/styles.css?v=0.29.1', html)
        self.assertIn('value="boundary-flow-fk"', html)
        self.assertIn('id="boundary-data-build"', html)
        self.assertIn('id="boundary-data-status"', html)
        self.assertIn("窗口前一交易日收盘，60日内不重置", html)
        for field in (
            "boundary_fk_paths",
            "boundary_drift_half_life_sessions",
            "boundary_minimum_direction_probability",
            "boundary_minimum_signal_to_noise",
            "boundary_minimum_survival_probability",
            "boundary_relative_strength_lookback_sessions",
            "boundary_maximum_drawdown",
        ):
            self.assertIn(f'name="{field}"', html)
        self.assertNotIn('name="topology_lookback_sessions"', html)
        self.assertNotIn('name="convection_lookback_sessions"', html)
        self.assertIn('api("/api/boundary-data/build"', script)
        for tab in (
            "策略评估",
            "样本外预测",
            "模型后验",
            "收益分析",
            "残差分析",
            "运行摘要",
        ):
            self.assertIn(tab, html)

    def test_fk_dataset_switch_uses_catalog_dates_and_nested_cache_status(self) -> None:
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('const start = (dataset.data_start || dataset.start || "").slice(0, 10);', script)
        self.assertIn('const end = (dataset.data_end || dataset.end || "").slice(0, 10);', script)
        self.assertIn('const variants = result.cache?.variants || result.variants || [];', script)
        self.assertIn('if (kind === "boundary-flow-fk") applyDataset();', script)
        self.assertIn('start: $(\'[name="start"]\').value,', script)
        self.assertIn('end: $(\'[name="end"]\').value,', script)

    def test_fk_policy_is_described_without_obsolete_strategy_variants(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertEqual(html.count('value="boundary-flow-fk"'), 1)
        self.assertIn("看多：全仓买入最强前 1/2", html)
        self.assertIn("看平：每次减总资产的 1/4", html)
        self.assertIn("看空：立即全部清仓", html)
        self.assertNotIn('value="path-efficiency-direct"', html)
        self.assertNotIn('"path-efficiency-direct"', script)









if __name__ == "__main__":
    unittest.main()
