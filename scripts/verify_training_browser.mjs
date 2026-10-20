// Real browser + workers over synthetic isolated fixtures; not accuracy evidence.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const verification = JSON.parse(await fs.readFile(path.join(root, 'reports/training_verification.json'), 'utf8'));
const catalog = path.join(root, verification.browser_root);
const output = path.join(root, 'artifacts/training_checks/browser');
const reportPath = path.join(root, 'reports/training_browser.json');
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8502';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1300 }, acceptDownloads: true });
const errors = [], transports = [];
page.on('pageerror', error => errors.push(error.message));
const button = name => page.getByRole('button', { name, exact: true });
async function idle() {
  await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 60000 });
  await page.waitForTimeout(300);
  assert.equal(await page.getByTestId('stException').count(), 0);
}
async function choose(label, value) {
  const control = page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) });
  if ((await control.innerText()).includes(value)) return;
  await control.getByRole('combobox').click();
  await page.getByRole('option').filter({ hasText: value }).click();
  await page.waitForTimeout(200);
}
function jobs() {
  const script = `import json,sqlite3,sys
with sqlite3.connect(sys.argv[1]) as c:
 c.row_factory=sqlite3.Row
 rows=[dict(r) for r in c.execute('SELECT * FROM training_jobs ORDER BY created_at DESC,id')]
 for r in rows:r['progress']=json.loads(r.pop('progress_json'))
 print(json.dumps(rows))`;
  const result = spawnSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', script, path.join(catalog, 'catalog.sqlite3')], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
async function waitJob(id, predicate, timeout = 120000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    const job = jobs().find(row => row.id === id);
    assert(job.status !== 'failed' && job.status !== 'interrupted', job.error);
    if (predicate(job)) return job;
    await page.waitForTimeout(500);
  }
  throw new Error('Job wait timed out: ' + id);
}
async function download(filename) {
  const pending = page.waitForEvent('download');
  pending.catch(() => {});
  await button('下载实验文件').click();
  const result = await pending;
  const destination = path.join(output, filename);
  const failure = await result.failure();
  if (failure) {
    // The local IDM can take over native downloads. Verify the exact HTTP
    // artifact without changing IDM/system settings; report this limitation.
    assert.equal(process.env.TRAINING_ALLOW_DOWNLOAD_FALLBACK, '1', failure);
    assert.equal(new URL(result.url()).origin, new URL(base).origin);
    const response = await page.request.get(result.url());
    assert.equal(response.status(), 200);
    await fs.writeFile(destination, await response.body());
    transports.push({ filename, native_failure: failure, transport: 'real HTTP 200 artifact fetched via Playwright request; native IDM integration unchanged' });
  } else {
    await result.saveAs(destination);
    transports.push({ filename, transport: 'native browser download' });
  }
  return { path: destination, sha256: crypto.createHash('sha256').update(await fs.readFile(destination)).digest('hex') };
}
async function waitPlot() {
  const img = page.getByTestId('stImage').locator('img').first();
  await img.scrollIntoViewIfNeeded();
  await img.evaluate(img => img.decode());
  assert(await img.evaluate(img => img.naturalWidth > 100));
  assert((await img.boundingBox()).width > 400, 'Saved plot must be visibly readable');
}
async function newTrain(epochs) {
  await page.getByRole('spinbutton', { name: '训练轮数', exact: true }).fill(String(epochs));
  await choose('训练图像尺寸', '160');
  await page.getByRole('spinbutton', { name: '训练批大小', exact: true }).fill('2');
  await page.getByRole('spinbutton', { name: '早停耐心轮数（0表示关闭）', exact: true }).fill('0');
  await button('开始训练').click();
  await page.getByText(/任务状态：/).waitFor();
  await idle();
  return jobs()[0];
}
try {
  await page.goto(new URL('/Training', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: 'YOLO模型训练与评估', exact: true }).waitFor({ timeout: 60000 });
  await idle();
  if (!historyOnly) {
    const actor = page.getByRole('textbox', { name: '实验执行者姓名或编号', exact: true });
    await actor.fill('浏览器自动化验证'); await actor.press('Tab'); await idle();
    const note = page.getByRole('textbox', { name: '实验说明', exact: true });
    await note.fill('合成数据功能验证，不是正式训练或准确率评测。'); await note.press('Tab'); await idle();
    const train = await newTrain(2);
    await waitJob(train.id, row => row.status === 'completed');
    await page.getByRole('heading', { name: 'val 集评估结果', exact: true }).waitFor({ timeout: 30000 });
    await idle();
    assert(await page.getByTestId('stVegaLiteChart').count() >= 3);
    await choose('选择下载文件', 'output/results.csv'); await idle();
    const csv = await download('results.csv');
    assert((await fs.readFile(csv.path, 'utf8')).includes('train/box_loss'));
    await choose('选择下载文件', 'output/weights/best.pt'); await idle();
    const weights = await download('best.pt');
    const trainResult = JSON.parse(await fs.readFile(path.join(catalog, 'training', train.id, 'result.json'), 'utf8'));
    assert.equal(weights.sha256, trainResult.weights.best.sha256);
    await waitPlot();
    await page.screenshot({ path: path.join(output, 'completed_training.png'), fullPage: true });
    console.log('Browser training, live curves and checkpoint download verified');
    await page.getByText('模型评估', { exact: true }).click(); await idle();
    await choose('评估图像尺寸', '160');
    await choose('评估计算设备', 'CPU');
    await page.getByRole('spinbutton', { name: '评估批大小', exact: true }).fill('2');
    await button('开始评估').click(); await idle();
    const evaluation = jobs()[0];
    assert.equal(evaluation.kind, 'eval');
    await waitJob(evaluation.id, row => row.status === 'completed');
    await page.getByRole('heading', { name: 'test 集评估结果', exact: true }).waitFor({ timeout: 30000 });
    await choose('选择下载文件', 'metrics.json'); await idle();
    const metrics = await download('test_metrics.json');
    const parsed = JSON.parse(await fs.readFile(metrics.path, 'utf8'));
    assert.equal(parsed.split, 'test');
    assert.deepEqual(parsed.per_class.map(row => row.class), ['fire', 'smoke']);
    await waitPlot();
    await page.screenshot({ path: path.join(output, 'test_evaluation.png'), fullPage: true });
    // Leave a real long worker alive for an external service restart. The
    // history-only phase verifies the same PID continues, then cancels it.
    await page.getByText('新建训练', { exact: true }).click(); await idle();
    const survivor = await newTrain(500);
    const running = await waitJob(survivor.id, row => row.status === 'running' && (row.progress.completed_epochs || 0) >= 1);
    assert.deepEqual(errors, []);
    const report = { generated_at: new Date().toISOString(), status: 'awaiting_restart', browser: await browser.version(),
      scope: 'Isolated synthetic fixtures and automated review; not formal accuracy evidence.', catalog_root: verification.browser_root,
      train_job: train.id, test_eval_job: evaluation.id, browser_training_and_live_loss_map_curves: true,
      best_weight_download_sha256: weights.sha256, test_metrics_download_sha256: metrics.sha256,
      explicit_test_evaluation_with_cpu: true, download_transports: transports, page_errors: errors,
      restart_survivor: { id: survivor.id, pid: running.pid, process_started: running.process_started, epochs_before_restart: running.progress.completed_epochs } };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'awaiting_restart', worker: report.restart_survivor }));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    const before = report.restart_survivor;
    const survivor = jobs().find(row => row.id === before.id);
    assert.equal(survivor.pid, before.pid);
    assert.equal(survivor.process_started, before.process_started);
    assert.equal(survivor.status, 'running');
    assert(survivor.progress.completed_epochs >= before.epochs_before_restart);
    await choose('查看任务', before.id); await idle();
    await button('取消当前任务').click();
    await waitJob(before.id, row => row.status === 'cancelled');
    await page.getByText(/任务状态：已取消/).waitFor({ timeout: 30000 });
    await button('刷新任务列表').click(); await idle();
    await choose('查看任务', report.test_eval_job); await idle();
    await page.getByRole('heading', { name: 'test 集评估结果', exact: true }).waitFor();
    await choose('选择下载文件', 'metrics.json'); await idle();
    const metrics = await download('test_metrics_after_restart.json');
    assert.equal(metrics.sha256, report.test_metrics_download_sha256);
    await page.screenshot({ path: path.join(output, 'after_restart.png'), fullPage: true });
    assert.deepEqual(errors, []);
    report.status = 'passed';
    report.worker_survives_service_restart_with_same_pid = true;
    report.running_worker_cancellation_passed = true;
    report.saved_metrics_identical_after_restart = true;
    report.epochs_after_restart_before_cancel = survivor.progress.completed_epochs;
    report.restart_verified_at = new Date().toISOString();
    report.restart_download_transports = transports;
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', background_worker_survived_restart: true, cancellation: true, metrics_preserved: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  throw error;
} finally { await browser.close(); }
