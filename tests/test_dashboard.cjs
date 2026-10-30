const assert = require('node:assert/strict');
const test = require('node:test');
const dashboard = require('../src/bnhmm/web/static/dashboard.js');

test('cumulative equity uses realized returns and turnover costs, not reset internal equity', () => {
  const predictions = [
    {signal_time:'2026-01-01',target_time:'2026-01-02',gross_strategy_return:.10,turnover:1,sector_equity:99,actual_weights:[.4,.6],sector:{actual_return:.02}},
    {signal_time:'2026-01-02',target_time:'2026-02-02',gross_strategy_return:-.05,turnover:.5,sector_equity:1,actual_weights:[],sector:{actual_return:-.01}},
  ];
  const rows = dashboard.equitySeries(predictions,10);
  assert.equal(rows[0].net,1.099);
  assert.ok(Math.abs(rows[1].net-1.099*.9495)<1e-12);
  assert.ok(Math.abs(rows[1].drawdown+.0505)<1e-12);
  assert.equal(rows[0].exposure,1);
  assert.equal(rows[1].exposure,0);
  assert.equal(rows[0].date,'2026-01-02');
  const months = dashboard.monthlyReturns(rows);
  assert.deepEqual(months.map(m=>m.month),['2026-01','2026-02']);
  assert.ok(Math.abs(months.reduce((product,m)=>product*(1+m.value),1)-rows[1].net)<1e-12);
});

test('missing cost or missing returns are not presented as zero performance', () => {
  const values = [{signal_time:'2026-01-01',gross_strategy_return:.1,turnover:1,sector:{}}];
  assert.equal(dashboard.equitySeries(values,undefined).length,0);
  assert.equal(dashboard.pct(null),'—');
  assert.equal(dashboard.pct(0),'0.00%');
});

test('months without scoring signals are gaps, and zero-hit scored months remain zero', () => {
  const job = {request:{transaction_cost_bps:5,symbols:['000001']}};
  const evaluation = {models:[{metrics:{},monthly_direction_accuracy:[{month:'2026-01',signal_count:0,accuracy:0,majority_baseline:.5},{month:'2026-02',signal_count:3,accuracy:0,majority_baseline:1}]}]};
  const html = dashboard.overview(job,evaluation,[],{});
  assert.equal(html.includes('26-01 · 多空命中率'),false);
  assert.equal(html.includes('26-02 · 多空命中率：0.00%'),true);
});

test('chart rendering escapes artifact text and does not create nonfinite SVG coordinates', () => {
  const html = dashboard.lineChart(['<script>alert(1)</script>','day2'],[{name:'test',color:'#000000',values:[0,null]}]);
  assert.equal(html.includes('<script>'),false);
  assert.equal(/NaN|Infinity/.test(html),false);
  assert.ok(html.includes('role="img"'));
  const bar = dashboard.barChart(['a'],[{name:'empty',color:'#000',values:[null]}]);
  assert.ok(bar.includes('暂无可评分样本'));
});

test('prediction filters match final strategy direction and preserve metadata links', () => {
  const predictions = [{signal_time:'2026-01-01T15',sector:{direction:'bull',state_probabilities:[.1,.2,.7],actual_return:.01},target_exposure:1,actual_symbols:['000001']}];
  assert.ok(dashboard.predictionTable(predictions,'bull',0).includes('data-signal-index="0"'));
  assert.ok(dashboard.predictionTable(predictions,'bear',0).includes('暂无匹配信号'));
});

test('BNHMM views consume saved evaluation series without mixing symbols into portfolio returns', () => {
  const predictions = [
    {symbol:'000001',signal_time:'2026-01-01',probability_mean:[.1,.2,.3,.4],target:3,forward_return:.02,expected_return:.01},
    {symbol:'000002',signal_time:'2026-01-01',probability_mean:[.4,.3,.2,.1],target:0,forward_return:-.02,expected_return:-.01},
  ];
  const grouped=dashboard.bnhmmDailyProbabilities(predictions);
  assert.equal(grouped.length,1);
  assert.equal(grouped[0].count,2);
  grouped[0].probabilities.forEach(value=>assert.ok(Math.abs(value-.25)<1e-12));
  const evaluation={observation_count:2,accuracy:.5,balanced_accuracy:.25,class_support:[1,0,0,1],predicted_class_counts:[1,0,0,1],confusion_matrix:[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],calibration:[{mean_confidence:.4,accuracy:.5}],backtest:{dates:['2026-01-02'],equity:[1.01],benchmark:[1],drawdown:[0],total_return:.01,benchmark_return:0,max_drawdown:0},series:{signal_times:['2026-01-01'],return_residuals:[.01]},residual_autocorrelation:[0],return_residual:{rmse:.01,mae:.01,correlation:0}};
  const job={request:{strategy:'bnhmm',symbols:['000001','000002']}};
  const views=dashboard.bnhmmPanels(job,{evaluation,summary:{symbols:['000001','000002']},posterior:{}},predictions);
  assert.ok(views.evaluation.includes('分类混淆矩阵'));
  assert.ok(views.evaluation.includes('1.00%'));
  assert.ok(views.returns.includes('策略与参考净值'));
  assert.ok(views.predictions.includes('data-bnhmm-symbol'));
  assert.equal(/NaN|Infinity/.test(Object.values(views).join('')),false);
});
