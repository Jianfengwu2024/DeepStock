(function (root) {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const phases = {
    queued: '任务排队', starting: '初始化', preparing_profiles: '准备训练样本',
    loading_strategy_dataset: '复用策略数据集', walk_forward: '滚动训练 / 回测',
    evaluation: '汇总评估与报告', completed: '运行完成', failed: '运行失败',
    cancelled: '已取消', interrupted: '运行中断',
    loading_member_bars: '读取本地分钟数据', building_minute_panel: '构建分钟矩阵',
    normalizing_minute_panel: '因果标准化', building_execution_constraints: '生成成交约束',
    building_sector_daily_states: '生成滚动状态与标签', saving_fk_cache: '保存策略数据集',
  };
  function duration(seconds) {
    if (!finite(seconds) || seconds < 0) return '估算中';
    const value = Math.round(seconds);
    if (value < 60) return value + ' 秒';
    if (value < 3600) return Math.floor(value / 60) + ' 分 ' + value % 60 + ' 秒';
    return Math.floor(value / 3600) + ' 小时 ' + Math.floor(value % 3600 / 60) + ' 分';
  }
  function model(job, now = Date.now()) {
    const p = job.progress || {}, terminal = ['completed', 'failed', 'cancelled', 'interrupted'].includes(job.status);
    const fraction = job.status === 'completed' ? 1 : finite(p.backtest_fraction) ? p.backtest_fraction : finite(p.fraction) ? p.fraction : null;
    const done = finite(p.backtest_completed_units) ? p.backtest_completed_units : p.completed_units;
    const total = finite(p.backtest_total_units) ? p.backtest_total_units : p.total_units;
    const unit = p.backtest_unit || p.unit || '单位';
    const validCount = finite(done) && finite(total) && total > 0;
    const elapsed = finite(p.elapsed_seconds) ? p.elapsed_seconds : job.started_at ? Math.max(0, (Date.parse(job.finished_at || new Date(now).toISOString()) - Date.parse(job.started_at)) / 1000) : 0;
    const age = p.updated_at ? Math.max(0, (now - Date.parse(p.updated_at)) / 1000) : null;
    const stale = !terminal && job.status === 'running' && age != null && age > 30;
    const terminalMessages = {completed:'计算与报告已完成', failed:'运行失败，请查看日志', cancelled:'任务已取消', interrupted:'任务在完成前中断，请查看日志'};
    const detail = [p.activity, p.current_symbol ? '股票 ' + p.current_symbol : '',
      finite(p.current_fold) ? '第 ' + p.current_fold + ' / ' + (p.total_folds || p.current_fold) + ' 折' : '',
      finite(p.current_epoch) ? '第 ' + p.current_epoch + ' / ' + (p.total_epochs || p.current_epoch) + ' 轮' : '',
      finite(p.batch_index) ? (p.batch_stage || '批次') + ' ' + p.batch_index + ' / ' + p.total_batches : '',
      p.signal_date || p.fold_month || ''].filter(Boolean).join(' · ');
    return {
      phase: terminal ? job.status : p.phase || job.status,
      title: phases[terminal ? job.status : p.phase || job.status] || p.phase || '运行中',
      message: terminal ? job.error || terminalMessages[job.status] : p.message || (job.status === 'queued' ? '等待本机计算资源，前序任务结束后启动' : validCount ? '计算进行中，按实际完成工作量更新' : '等待计算进程上报真实进度'),
      fraction: fraction == null ? null : Math.max(0, Math.min(1, fraction)),
      percent: fraction == null ? '统计中' : (Math.max(0, Math.min(1, fraction)) * 100).toFixed(1) + '%',
      completed: validCount ? done.toLocaleString('zh-CN') + ' / ' + total.toLocaleString('zh-CN') + ' ' + unit : '尚未确定',
      remaining: validCount ? Math.max(0, total - done).toLocaleString('zh-CN') + ' ' + unit : '尚未确定',
      elapsed: duration(elapsed),
      eta: job.status === 'completed' ? '已完成' : terminal ? '已停止' : stale ? '等待更新' : duration(p.eta_seconds),
      etaNote: terminal ? '最终状态' : stale ? '进度暂未更新，暂停显示旧估算；恢复上报后重新计算' : finite(p.eta_seconds) ? '按当前阶段已完成速度估算，可能波动' : '完成首个工作单元后估算，不使用固定倒计时',
      detail: terminal ? '' : detail, stale,
      freshness: terminal ? '已保存最终进度' : age == null ? '等待首次进度上报' : age > 30 ? Math.round(age) + ' 秒未更新，当前步骤可能耗时较长' : '最近 ' + Math.round(age) + ' 秒更新 · 每 3 秒刷新',
      isBacktest: finite(p.backtest_fraction) || p.phase === 'walk_forward' || job.status === 'completed',
    };
  }
  function render(job) {
    const m = model(job), byId = id => document.getElementById(id);
    const band = byId('progress-band');
    band.classList.remove('hidden');
    band.dataset.phase = m.phase;
    band.classList.toggle('monitor-stale', m.stale);
    const fields = {'progress-label':m.title,'monitor-message':m.message,'monitor-percent':m.percent,
      'monitor-completed':m.completed,'monitor-remaining':m.remaining,'elapsed-time':m.elapsed,
      'monitor-eta':m.eta,'monitor-eta-note':m.etaNote,'monitor-detail':m.detail,'monitor-freshness':m.freshness,
      'monitor-percent-label':m.isBacktest?'回测进度':'当前阶段进度'};
    Object.entries(fields).forEach(([id, value]) => { if (byId(id)) byId(id).textContent = value; });
    const bar = byId('progress-value');
    if (m.fraction == null) bar.removeAttribute('value'); else bar.value = m.fraction * 100;
    band.setAttribute('aria-label', '任务状态监控：' + m.title);
  }
  const api = {duration, model, render};
  root.DeepStockProgress = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
