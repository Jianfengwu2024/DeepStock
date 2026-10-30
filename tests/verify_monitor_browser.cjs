// Live completed-job check plus an isolated replay of recorded real progress.
// API replay never starts a calculation or changes the service's stored jobs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const base = 'http://127.0.0.1:8765';
  const output = path.resolve('local-state/validation');
  const evidence = JSON.parse(fs.readFileSync(path.join(output, 'live-progress-verification.json'), 'utf8'));
  const job = await (await fetch(base + '/api/jobs/' + evidence.job_id)).json();
  assert.equal(job.status, 'completed');
  const browser = await chromium.launch({channel:'msedge', headless:true});
  try {
    const errors = [];
    const page = await browser.newPage({viewport:{width:1540,height:1080}});
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base + '/#training', {waitUntil:'domcontentloaded'});
    await page.locator('#progress-band:not(.hidden)').waitFor();
    assert.equal(await page.locator('#monitor-percent').textContent(), '100.0%');
    assert.equal(await page.locator('#monitor-completed').textContent(), '242 / 242 交易日');
    assert.equal(await page.locator('#monitor-remaining').textContent(), '0 交易日');
    assert.equal(await page.locator('#monitor-eta').textContent(), '已完成');
    assert.equal(await page.locator('#monitor-detail').textContent(), '');
    await page.locator('#progress-band').screenshot({path:path.join(output,'task-monitor-completed.png')});
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => document.querySelector('#monitor-percent')?.textContent === '100.0%');
    await page.locator('[data-page="data"]').click();
    await page.waitForFunction(() => document.querySelector('#strategy-datasets')?.textContent.includes('已保存，可直接运行'));
    const saved = await page.locator('#strategy-datasets').textContent();
    assert.ok(saved.includes('179 只'));
    assert.ok(saved.includes('2025-10-23'));
    assert.ok(saved.includes('sector-rolling-anchor-60d-v3'));
    await page.locator('.dataset-section').filter({has:page.locator('#strategy-datasets')}).screenshot({path:path.join(output,'strategy-datasets-saved.png')});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.close();

    const replay = await browser.newPage({viewport:{width:1540,height:1080}});
    replay.on('pageerror', error => errors.push(error.message));
    const snapshot = evidence.progress_snapshots.find(p => p.completed_units === 27);
    assert.ok(snapshot);
    let replayJob = {...job, status:'running', finished_at:null, artifacts:{}, progress:{...snapshot,
      unit:'交易日', backtest_unit:'交易日', updated_at:new Date().toISOString()}};
    await replay.route('**/api/jobs/' + job.id, route => route.fulfill({contentType:'application/json',body:JSON.stringify(replayJob)}));
    await replay.goto(base + '/#training', {waitUntil:'domcontentloaded'});
    await replay.waitForFunction(() => document.querySelector('#monitor-percent')?.textContent === '11.2%');
    assert.equal(await replay.locator('#monitor-completed').textContent(), '27 / 242 交易日');
    assert.equal(await replay.locator('#monitor-remaining').textContent(), '215 交易日');
    assert.ok((await replay.locator('#monitor-detail').textContent()).includes('第 3 / 13 折'));
    assert.ok((await replay.locator('#monitor-eta-note').textContent()).includes('按当前阶段'));
    await replay.locator('#progress-band').screenshot({path:path.join(output,'task-monitor-real-progress-replay.png')});
    const render = async value => {
      replayJob = value;
      await replay.evaluate(jobValue => window.DeepStockProgress.render(jobValue), replayJob);
    };
    await render({...replayJob, progress:{phase:'starting',updated_at:new Date().toISOString(),eta_seconds:null}});
    assert.equal(await replay.locator('#monitor-eta').textContent(), '估算中');
    assert.equal(await replay.locator('#progress-value').getAttribute('value'), null);
    await render({...replayJob, progress:{...snapshot,unit:'交易日',updated_at:new Date(Date.now()-61000).toISOString()}});
    assert.equal(await replay.locator('#monitor-eta').textContent(), '等待更新');
    assert.ok((await replay.locator('#progress-band').getAttribute('class')).includes('monitor-stale'));
    await render({...replayJob,status:'failed',error:'浏览器验证：进程失败提示',progress:{...snapshot,message:'正在训练'}});
    assert.equal(await replay.locator('#progress-label').textContent(), '运行失败');
    assert.equal(await replay.locator('#monitor-percent').textContent(), '11.2%');
    assert.equal(await replay.locator('#monitor-eta').textContent(), '已停止');
    assert.equal(await replay.locator('#monitor-message').textContent(), '浏览器验证：进程失败提示');
    await render({...job, status:'running',finished_at:null,progress:{...snapshot,unit:'交易日',updated_at:new Date().toISOString()}});
    await replay.setViewportSize({width:390,height:844});
    assert.equal(await replay.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await replay.locator('#progress-band').screenshot({path:path.join(output,'task-monitor-mobile-replay.png')});
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed:true,checks:['live completed counts','reload final progress','saved strategy datasets','recorded real progress replay','unknown ETA','stale ETA suppressed','failure retains partial progress','mobile overflow'],screenshots:output}));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
