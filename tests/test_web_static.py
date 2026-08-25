from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "bnhmm" / "web" / "static"


class SectorFlowStaticContractTests(unittest.TestCase):
    def test_window_comparison_has_submit_entry_and_dedicated_result_view(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        for identifier in (
            "run-window-comparison",
            "comparison-list",
            "comparison-view",
            "comparison-summary-metrics",
            "comparison-window-table",
            "comparison-equity-canvas",
            "delete-comparison-button",
        ):
            self.assertIn(f'id="{identifier}"', html)
        for label in (
            "运行七窗口对比",
            "窗口对比",
            "成本后收益",
            "最大回撤",
            "方向正确率",
            "均衡正确率",
            "平均仓位",
            "平均换手",
        ):
            self.assertIn(label, html)
        self.assertIn('api("/api/comparisons"', script)
        self.assertIn("submitWindowComparison", script)
        self.assertIn("comparison.windows.length", script)
        self.assertIn("120: \"#", script)
        self.assertNotIn("comparison.completed_count}/6", script)
        self.assertNotIn("ready}/6", script)
        self.assertIn("boundary_cache_lookbacks?.length", script)
        self.assertIn('<option value="120">120 日</option>', html)
        self.assertIn("构造 / 刷新七套数据", html)
        self.assertIn("renderComparison", script)
        self.assertIn("drawComparisonEquity", script)
        self.assertIn("openComparisonChild", script)
        self.assertIn("async function openComparisonChild", script)
        self.assertIn("deleteSelectedComparison", script)
        self.assertIn("收益较好，但预测优势仍弱", script)

    def test_history_management_actions_stay_inside_history_panel(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        workspace_header = html.split(
            '<header class="workspace-header">', 1
        )[1].split("</header>", 1)[0]
        history_panel = html.split('<nav class="run-history"', 1)[1].split(
            "</nav>", 1
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

    def test_strategy_switch_excludes_hidden_configuration_controls(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn("function syncStrategyControlAvailability()", script)
        self.assertIn("function namedFormControls(form, name)", script)
        self.assertIn("function preferredFormControl(form, name)", script)
        self.assertIn("for (const field of namedFormControls(form, name))", script)
        self.assertIn('form.querySelectorAll("[name]")', script)
        self.assertIn('element.closest(".hidden")', script)
        self.assertIn("if (element.disabled) continue;", script)
        self.assertIn(
            'class="non-boundary-flow-only">交易成本 BP<input '
            'name="transaction_cost_bps"',
            html,
        )

    def test_boundary_flow_configuration_and_results_have_stable_hooks(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('/static/app.js?v=0.26.1', html)
        self.assertIn('/static/styles.css?v=0.26.1', html)
        self.assertIn('value="boundary-flow-fk"', html)
        self.assertIn('name="boundary_fk_paths"', html)
        self.assertIn('id="boundary-data-build"', html)
        self.assertIn('id="boundary-data-status"', html)
        self.assertIn('<option value="90">90 日</option>', html)
        self.assertIn('api("/api/boundary-data/build"', script)
        for field in (
            "boundary_drift_half_life_sessions",
            "boundary_minimum_direction_probability",
            "boundary_minimum_signal_to_noise",
            "boundary_minimum_survival_probability",
        ):
            self.assertIn(f'name="{field}"', html)
        for field in (
            "boundary_branch_count",
            "boundary_viscosity_window",
            "boundary_min_raw_bullish_probability",
            "boundary_min_survival_bullish_probability",
            "boundary_min_trend_survival_probability",
            "boundary_max_bullish_probability_std",
            "boundary_max_thickness_risk_quantile",
        ):
            self.assertNotIn(f'name="{field}"', html)
        for identifier in (
            "boundary-flow-results",
            "boundary-probabilities",
            "boundary-dynamics",
            "boundary-gate-reason",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn("renderBoundaryFlowResult", script)
        expected_tabs = (
            ("evaluation", "策略评估"),
            ("predictions", "样本外预测"),
            ("posterior", "模型后验"),
            ("returns", "收益分析"),
            ("residuals", "残差分析"),
            ("summary", "运行摘要"),
        )
        for panel, label in expected_tabs:
            self.assertIn(f'data-boundary-tab="{panel}"', html)
            self.assertIn(f'data-boundary-panel="{panel}"', html)
            self.assertIn(label, html)
        self.assertLess(
            html.index('id="boundary-result-tabs"'),
            html.index('data-boundary-panel="evaluation"'),
        )
        for identifier in (
            "boundary-evaluation-summary",
            "boundary-prediction-table",
            "boundary-return-canvas",
            "boundary-residual-canvas",
            "boundary-run-summary",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn("switchBoundaryResultTab", script)
        self.assertIn("paintBoundaryReturnAnalysis", script)
        self.assertIn("paintBoundaryResidualAnalysis", script)
        self.assertNotIn('id="boundary-table-head"', html)
        self.assertNotIn('id="boundary-stage-table"', html)
        self.assertNotIn('id="boundary-ablation-table"', html)
        self.assertNotIn("boundary-policy-ablation.json", script)
        self.assertNotIn("ablationLabels", script)
        self.assertIn("纯路径效率", html)
        self.assertIn("metrics.average_exposure", script)
        self.assertIn("metrics.average_turnover", script)
        self.assertIn('"boundary-flow-fk"', script)
        self.assertIn("路径效率 FK", html)
        for artifact in (
            "path-efficiency-fk-report.md",
            "boundary-policy-evaluation.json",
            "feynman-kac-predictions.json",
        ):
            self.assertIn(artifact, script)
        for field in (
            "branches",
            "state_probabilities",
            "expected_drift",
            "expected_effective_potential",
            "bullish_drift_probability",
            "drift_signal_to_noise",
            "survival_confidence",
            "selected_symbols",
            "branch_target_exposures",
        ):
            self.assertIn(field, script)
        self.assertNotIn("旧衰减漂移", script)

    def test_path_efficiency_fk_is_the_only_two_branch_strategy(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertEqual(html.count('value="boundary-flow-fk"'), 1)
        self.assertIn("路径效率 FK", html)
        self.assertIn("看多：满仓", html)
        self.assertIn("看平：每次减仓 25%", html)
        self.assertIn("连续四次看平清仓", html)
        self.assertIn("看空：立即清仓", html)
        self.assertNotIn('value="path-efficiency-direct"', html)
        self.assertNotIn('"path-efficiency-direct"', script)
        self.assertNotIn("path-efficiency-evaluation.json", script)
        self.assertNotIn("path-efficiency-predictions.json", script)

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
