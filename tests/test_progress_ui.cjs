const test=require('node:test');
const assert=require('node:assert/strict');
const ui=require('../src/bnhmm/web/static/progress-ui.js');
test('running monitor shows measured counts and time',()=>{
 const p=ui.model({status:'running',progress:{phase:'walk_forward',completed_units:25,total_units:100,unit:'交易日',fraction:.25,elapsed_seconds:20,eta_seconds:60,current_fold:3,total_folds:13,current_epoch:2,total_epochs:10,updated_at:'2026-10-26T00:00:00Z'}},Date.parse('2026-10-26T00:00:05Z'));
 assert.equal(p.percent,'25.0%');assert.equal(p.completed,'25 / 100 交易日');assert.equal(p.remaining,'75 交易日');assert.equal(p.eta,'1 分 0 秒');assert.ok(p.detail.includes('第 3 / 13 折'));assert.equal(p.stale,false);
});
test('unknown progress is indeterminate, failed runs do not become complete',()=>{
 const running=ui.model({status:'running',progress:{phase:'starting',eta_seconds:null}});
 assert.equal(running.percent,'统计中');assert.equal(running.eta,'估算中');
 const failure=ui.model({status:'failed',progress:{phase:'walk_forward',fraction:.4}});
 assert.equal(failure.percent,'40.0%');assert.equal(failure.eta,'已停止');
});
test('stored backtest progress survives report stage and stale updates are labelled',()=>{
 const m=ui.model({status:'running',progress:{phase:'evaluation',backtest_fraction:1,backtest_completed_units:242,backtest_total_units:242,backtest_unit:'交易日',updated_at:'2026-10-26T00:00:00Z'}},Date.parse('2026-10-26T00:01:00Z'));
 assert.equal(m.title,'汇总评估与报告');assert.equal(m.percent,'100.0%');assert.equal(m.remaining,'0 交易日');assert.equal(m.stale,true);
 assert.equal(m.eta,'等待更新');assert.ok(m.etaNote.includes('暂停显示旧估算'));
});
test('terminal monitor does not retain an old activity or success message',()=>{
 const m=ui.model({status:'failed',error:'模型进程退出，请重试',progress:{phase:'walk_forward',fraction:.4,message:'正在训练',activity:'调参训练',current_epoch:2}});
 assert.equal(m.title,'运行失败');assert.equal(m.message,'模型进程退出，请重试');assert.equal(m.detail,'');assert.equal(m.percent,'40.0%');
});
