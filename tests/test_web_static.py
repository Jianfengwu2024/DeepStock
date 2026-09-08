from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "bnhmm" / "web" / "static"


class SectorFlowStaticContractTests(unittest.TestCase):
    def test_mobile_header_keeps_actions_compact(self) -> None:
        styles = (ROOT / "styles.css").read_text(encoding="utf-8")

        mobile = styles.split("@media (max-width: 820px)", 1)[1]
        self.assertIn(".header-title { min-width: 0; flex: 1; }", mobile)
        self.assertIn(
            ".header-title p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
            mobile,
        )
        self.assertIn(".header-link span { display: none; }", mobile)

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
        self.assertIn("for (const field of namedFormControls(form, name))", script)
        self.assertIn('form.querySelectorAll("[name]")', script)
        self.assertIn('element.closest(".hidden")', script)
        self.assertIn("if (element.disabled) continue;", script)
        self.assertIn(
            'class="non-boundary-flow-only">交易成本 BP<input '
            'name="transaction_cost_bps"',
            html,
        )
        self.assertIn("佣金 2 BP + 滑点 3 BP", html)

    def test_boundary_flow_configuration_and_results_have_stable_hooks(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertIn('/static/app.js?v=0.28.0', html)
        self.assertIn('/static/styles.css?v=0.28.2', html)
        self.assertIn('value="boundary-flow-fk"', html)
        self.assertIn('name="boundary_fk_paths"', html)
        self.assertIn('name="boundary_maximum_drawdown"', html)
        self.assertIn("贝叶斯基础漂移截距", html)
        self.assertIn("全板块最强前 1/2", html)
        self.assertIn("所有成分股压缩为一个每日板块状态", html)
        self.assertIn("预测下一交易日板块涨跌方向", html)
        self.assertIn('id="boundary-data-build"', html)
        self.assertIn('id="boundary-data-status"', html)
        boundary_config = html.split('<details class="form-section boundary-flow-only', 1)[1].split("</details>", 1)[0]
        self.assertNotIn('name="topology_lookback_sessions"', boundary_config)
        self.assertNotIn('name="convection_lookback_sessions"', html)
        self.assertNotIn('<select name="topology_lookback_sessions">', html)
        self.assertIn('api("/api/boundary-data/build"', script)
        for field in (
            "boundary_drift_half_life_sessions",
            "boundary_minimum_direction_probability",
            "boundary_minimum_signal_to_noise",
            "boundary_minimum_survival_probability",
            "boundary_relative_strength_lookback_sessions",
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
        self.assertIn("function selectBoundaryStages(stages)", script)
        self.assertIn('item.model_name === "SectorFK-PathOnly"', script)
        self.assertIn("function boundaryModelLabel(modelName)", script)
        self.assertIn('document.querySelector("#boundary-return-legend").innerHTML', script)
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
            "boundary-return-legend",
            "boundary-residual-canvas",
            "boundary-run-summary",
        ):
            self.assertIn(f'id="{identifier}"', html)
        self.assertIn("switchBoundaryResultTab", script)
        self.assertIn("paintBoundaryReturnAnalysis", script)
        self.assertIn("drawBoundaryComparison", script)
        self.assertIn("baselinePredictions", script)
        self.assertIn("一日反向流量外势 FK 净值", html)
        self.assertIn("纯路径效率基线", html)
        self.assertIn('class="run-identity"', html)
        self.assertIn("paintBoundaryResidualAnalysis", script)
        self.assertNotIn("无截距贝叶斯方向分类", html)
        self.assertIn("带方向路径效率 + 板块成分股同步率", html)
        self.assertIn("忽略小于 0.1 倍历史波动率的模糊涨跌", html)
        self.assertNotIn('id="boundary-table-head"', html)
        self.assertNotIn('id="boundary-stage-table"', html)
        self.assertNotIn('id="boundary-ablation-table"', html)
        self.assertNotIn("boundary-policy-ablation.json", script)
        self.assertNotIn("ablationLabels", script)
        self.assertIn("纯路径效率", html)
        self.assertIn("metrics.average_exposure", script)
        self.assertIn("metrics.average_turnover", script)
        self.assertIn("metrics.drift_direction_accuracy", script)
        self.assertIn("metrics.drift_balanced_accuracy", script)
        self.assertIn("directionReportUrl", script)
        self.assertIn('id="boundary-direction-comparison"', html)
        self.assertIn("metrics.direction_signal_coverage", script)
        self.assertIn("metrics.direction_signal_accuracy", script)
        self.assertIn('"boundary-flow-fk"', script)
        self.assertIn("路径效率 FK", html)
        for artifact in (
            "path-efficiency-fk-report.md",
            "path-efficiency-convection-potential-report.md",
            "boundary-policy-evaluation.json",
            "convection-potential-evaluation.json",
            "feynman-kac-predictions.json",
        ):
            self.assertIn(artifact, script)
        self.assertIn("PathEfficiencyFK-60D-OneDayPotential", script)
        self.assertIn("最新 β0", script)
        self.assertIn("potential_mean?.[0]", script)
        self.assertIn("boundaryImprovementVerdict", script)
        self.assertIn("returnDelta > 0", script)
        self.assertIn("未通过改进标准", script)
        self.assertIn("方向预测结论", script)
        self.assertIn("纯路径效率尚未超过多数方向基线", script)
        self.assertIn("paintSectorDirectionComparison", script)
        self.assertIn("板块峰值回撤止损已触发", script)
        for field in ("expected_effective_potential", "survival_confidence"):
            self.assertIn(field, script)
        self.assertIn("背景势 β0 + 一日反向流量集中度 C1", html)
        self.assertIn("convection_potential_features", script)
        self.assertNotIn("convection_lookback_sessions", script)
        self.assertNotIn("convection_exposure_cap", script)
        self.assertNotIn("convection_gate_reason", script)
        self.assertNotIn("对流强度", script)
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

    def test_path_efficiency_fk_uses_one_whole_sector_state(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")

        self.assertEqual(html.count('value="boundary-flow-fk"'), 1)
        self.assertIn("路径效率 FK", html)
        self.assertIn("看多：全仓买入最强前 1/2", html)
        self.assertIn("看平：每次减仓总资产的 1/4", html)
        self.assertIn("连续四次看平清仓", html)
        self.assertIn("看空：全部清仓", html)
        self.assertIn("const sector = latest.sector", script)
        self.assertIn("latest.branches?.[0]", script)
        self.assertNotIn('value="path-efficiency-direct"', html)
        self.assertNotIn('"path-efficiency-direct"', script)
        self.assertNotIn("path-efficiency-evaluation.json", script)
        self.assertNotIn("path-efficiency-predictions.json", script)

    def test_percolation_branching_has_a_dedicated_configuration_and_result_workspace(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "styles.css").read_text(encoding="utf-8")

        self.assertIn('value="percolation-branching"', html)
        self.assertIn("渗流分枝", html)
        for field in (
            "percolation_activation_threshold",
            "percolation_null_permutations",
            "percolation_posterior_paths",
            "percolation_phase_probability_threshold",
            "transaction_cost_bps",
            "seed",
        ):
            self.assertIn(f'name="{field}"', html)
        self.assertIn('name="maximum_lag_minutes" value="3"', html)
        self.assertIn('name="topology_lookback_sessions" value="60"', html)
        self.assertIn("固定 1 至 3 分钟传播", html)
        self.assertIn("固定 60 日月度骨架", html)

        expected_tabs = (
            ("evaluation", "策略评估"),
            ("prediction", "相变预测"),
            ("critical", "临界后验"),
            ("network", "传播网络"),
            ("returns", "收益分析"),
            ("calibration", "校准诊断"),
            ("summary", "运行摘要"),
        )
        for panel, label in expected_tabs:
            self.assertIn(f'data-percolation-tab="{panel}"', html)
            self.assertIn(f'data-percolation-panel="{panel}"', html)
            self.assertIn(label, html)
        for identifier in (
            "percolation-results",
            "percolation-phase-probabilities",
            "percolation-prediction-table",
            "percolation-critical-summary",
            "percolation-network-sources",
            "percolation-equity-canvas",
            "percolation-confusion-matrix",
            "percolation-calibration",
            "percolation-run-summary",
            "percolation-downloads",
        ):
            self.assertIn(f'id="{identifier}"', html)

        self.assertIn('"percolation-branching": [', script)
        self.assertIn('"percolation-branching": "PercolationBranching"', script)
        self.assertIn("const percolation = strategy === \"percolation-branching\"", script)
        self.assertIn("renderPercolationResult", script)
        self.assertIn("switchPercolationResultTab", script)
        self.assertIn("paintPercolationResult", script)
        self.assertIn("drawPercolationEquity", script)
        for artifact in (
            "percolation-evaluation.json",
            "percolation-baselines.json",
            "percolation-predictions.json",
            "percolation-posteriors.json",
            "percolation-equity.json",
            "percolation-network-summary.json",
            "percolation-report-zh.md",
        ):
            self.assertIn(artifact, script)
        self.assertIn('"fk-stock", "boundary-flow-fk", "percolation-branching"', script)
        self.assertIn(".percolation-result-tabs", styles)
        self.assertIn(".percolation-phase-grid", styles)

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
