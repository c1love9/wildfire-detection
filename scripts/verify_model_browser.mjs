// Synthetic/public screens only. Physical camera frames are never saved.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const fixture = JSON.parse(await fs.readFile(path.join(root, 'reports/model_integration.json'), 'utf8'));
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8502';
const output = path.join(root, 'artifacts/model_checks/browser');
const reportPath = path.join(root, 'reports/model_browser.json');
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: 'msedge', args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1300 }, acceptDownloads: true });
const errors = [], mediaErrors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('response', response => { if (response.url().includes('/media/') && response.status() >= 400) mediaErrors.push(response.status()); });
const button = name => page.getByRole('button', { name, exact: true });
const modelControl = () => page.getByTestId('stSelectbox').filter({ has: page.getByText('检测模型', { exact: true }) }).getByRole('combobox');
async function idle() {
  await page.waitForTimeout(200); // Let the widget change reach the rerun queue.
  await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 60000 });
  await page.waitForTimeout(600);
  assert.equal(await page.getByTestId('stException').count(), 0);
}
async function choose(label, value) {
  const control = page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) });
  if ((await control.innerText()).includes(value)) return;
  await control.getByRole('combobox').click();
  await page.getByRole('option').filter({ hasText: value }).click();
  await idle();
}
async function expand(label) {
  const control = page.getByTestId('stExpander').filter({ has: page.getByText(label, { exact: true }) });
  if (!await control.locator('details').evaluate(element => element.open)) await control.locator('summary').click();
}
async function number(label, value) {
  const input = page.getByRole('spinbutton', { name: label, exact: true });
  const expander = page.getByTestId('stExpander').filter({ has: page.getByText(label, { exact: true }) });
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      if (await expander.count() && !await expander.locator('details').evaluate(element => element.open)) await expander.locator('summary').click();
      await input.fill(String(value), { timeout: 5000 });
      await input.press('Tab'); await idle();
      if (Math.abs(Number(await input.inputValue()) - value) < 1e-6) return;
    } catch (error) { if (attempt === 2) throw error; await idle(); }
  }
  throw new Error('Number input did not retain value: ' + label);
}
async function waitFrames(minimum, camera = false) {
  const label = camera ? '本段已检测帧' : '当前帧';
  const metric = page.getByTestId('stMetric').filter({ has: page.getByText(label, { exact: true }) }).getByTestId('stMetricValue');
  const started = Date.now();
  while (Date.now() - started < 60000) {
    if (await metric.count() && Number((await metric.innerText()).split('/')[0].replaceAll(',', '').trim()) >= minimum) return;
    if (await page.getByText(/实时检测采样间隔过大/).count()) throw new Error('Camera sample gap exceeded');
    await page.waitForTimeout(100);
  }
  throw new Error('No advancing detection frames');
}
async function decoded() {
  const image = page.getByTestId('stImage').locator('img').first();
  await image.waitFor();
  await image.evaluate(image => image.decode());
  assert(await image.evaluate(image => image.naturalWidth > 0));
}
function events() {
  const script = `import json,sqlite3,sys
with sqlite3.connect(sys.argv[1]) as c:
 c.row_factory=sqlite3.Row
 rows=[dict(r) for r in c.execute('SELECT e.id,r.source,r.metadata_json FROM events e JOIN runs r ON e.run_id=r.id ORDER BY e.started_at DESC')]
 print(json.dumps(rows))`;
  const result = spawnSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', script, path.join(root, fixture.alarm_log)], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
async function history(eventId) {
  await page.goto(new URL('/Alarm_History', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '报警日志', exact: true }).waitFor({ timeout: 60000 });
  await choose('输入来源筛选', 'synthetic_scene.avi');
  await choose('查看事件变化', eventId.slice(0, 12));
  await page.getByText(/本次检测模型：自训/).waitFor();
  await decoded();
  const video = page.locator('video');
  await video.waitFor();
  await page.waitForFunction(() => document.querySelector('video')?.readyState >= 2);
  const duration = await video.evaluate(video => video.duration);
  assert(duration > 0.2);
  await video.evaluate(async video => { video.muted = true; await video.play(); });
  await page.waitForFunction(() => document.querySelector('video').currentTime > 0.1);
  await video.evaluate(video => video.pause());
  assert.equal(await video.evaluate(video => video.error), null);
  const pending = page.waitForEvent('download');
  await button('导出当前筛选 CSV').click();
  const download = await pending;
  const dest = path.join(output, historyOnly ? 'history_after_restart.csv' : 'history.csv');
  await download.saveAs(dest);
  const csv = await fs.readFile(dest, 'utf8');
  assert(csv.includes(fixture.model_sha256) && csv.includes(fixture.training_id));
  return { duration, csv_sha256: crypto.createHash('sha256').update(await fs.readFile(dest)).digest('hex') };
}
try {
  if (!historyOnly) {
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.getByText('Select Model Type', { exact: true }).waitFor({ timeout: 60000 });
    await choose('Select Model Size', fixture.training_id.slice(-8));
    await page.getByText('Upload from Computer', { exact: true }).click(); await idle();
    await page.getByTestId('stFileUploader').locator('input[type=file]').setInputFiles(path.join(root, fixture.browser_image));
    await page.getByTestId('stAlert').filter({ hasText: 'Predicted' }).waitFor();
    await idle(); await decoded();
    await page.screenshot({ path: path.join(output, 'self_trained_image.png'), fullPage: true });
    await choose('Select Model Size', 'fire_n');
    await page.getByText('General', { exact: true }).click(); await idle();
    await page.getByText('Fire Detection', { exact: true }).click(); await idle();
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor();
    await page.getByText('上传视频', { exact: true }).click(); await idle();
    await page.getByTestId('stFileUploader').locator('input[type=file]').setInputFiles(path.join(root, fixture.browser_video)); await idle();
    await choose('检测模型', fixture.training_id.slice(-8));
    const confidence = page.getByTestId('stSlider').filter({ has: page.getByText('置信度阈值', { exact: true }) }).getByRole('slider');
    await confidence.press('Home'); await idle();
    await expand('报警规则');
    for (const [label, value] of [['连续确认帧数', 2], ['火焰报警置信度', 0.05], ['烟雾报警置信度', 0.05],
                                  ['关注持续时间（秒）', 0], ['预警持续时间（秒）', 0.1], ['报警持续时间（秒）', 0.2]]) await number(label, value);
    await expand('报警截图与片段');
    await number('确认前保存（秒）', 0); await number('确认后保存（秒）', 1);
    await button('开始检测').click();
    await waitFrames(5); assert(await modelControl().isDisabled());
    await button('暂停检测').click(); await idle();
    assert(await modelControl().isDisabled());
    await page.getByText(/本次观测模型：自训/).waitFor();
    await button('继续检测').click(); await waitFrames(18);
    await button('停止检测').click(); await idle();
    assert(await modelControl().isEnabled());
    const event = events().find(row => row.source === 'synthetic_scene.avi' && JSON.parse(row.metadata_json).model?.training_id === fixture.training_id);
    assert(event);
    await choose('检测模型', 'fire_n');
    await button('开始检测').click(); await waitFrames(2);
    await page.getByText(/本次观测模型：fire_n/).waitFor();
    await button('停止检测').click(); await idle();
    const playback = await history(event.id);
    await page.screenshot({ path: path.join(output, 'self_trained_history.png'), fullPage: true });
    console.log('Image/video model switching, alarms, provenance CSV and replay passed');
    // Physical camera: suppress events and media; do not take a screenshot.
    const cameraEventsBefore = events().filter(row => row.source === '摄像头 0').length;
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor();
    await idle();
    await page.getByText('摄像头', { exact: true }).click(); await idle();
    await choose('检测模型', fixture.training_id.slice(-8));
    await expand('报警规则');
    await number('火焰报警置信度', 1); await number('烟雾报警置信度', 1);
    await expand('报警截图与片段');
    const checkbox = page.getByRole('checkbox', { name: '保存报警截图和片段', exact: true });
    if (await checkbox.isChecked()) await page.getByTestId('stCheckbox').filter({ has: page.getByText('保存报警截图和片段', { exact: true }) }).locator('label').click();
    await idle();
    assert.equal(await checkbox.isChecked(), false);
    await button('开始检测').click(); await waitFrames(12, true);
    assert(await modelControl().isDisabled()); await decoded();
    await button('暂停检测').click(); await idle();
    assert(await modelControl().isDisabled());
    await button('继续检测').click();
    await page.getByText(/观测段 2/).waitFor(); await waitFrames(5, true);
    await button('停止检测').click(); await idle();
    assert(await modelControl().isEnabled());
    assert.equal(events().filter(row => row.source === '摄像头 0').length, cameraEventsBefore);
    assert.deepEqual(errors, []); assert.deepEqual(mediaErrors, []);
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      scope: 'Synthetic images/video and self-trained functional model. Physical camera frames/alarms/media not saved.',
      model_sha256: fixture.model_sha256, model_key: fixture.model_key, training_id: fixture.training_id,
      image_upload_and_self_trained_selection: true, switch_back_to_upstream_and_general_image_regression: true,
      video_upload_model_switch_pause_resume_stop: true, model_locked_while_running_and_paused: true,
      self_trained_history_event: event.id, history_playback: playback, model_provenance_csv: true,
      physical_camera_self_trained_pause_resume_stop: true, physical_camera_images_and_events_saved: false,
      page_errors: errors, media_errors: mediaErrors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.getByText('Select Model Type', { exact: true }).waitFor({ timeout: 60000 });
    await choose('Select Model Size', fixture.training_id.slice(-8)); await idle();
    const playback = await history(report.self_trained_history_event);
    assert.equal(playback.csv_sha256, report.history_playback.csv_sha256);
    assert.deepEqual(errors, []); assert.deepEqual(mediaErrors, []);
    await page.screenshot({ path: path.join(output, 'after_restart.png'), fullPage: true });
    report.model_selection_and_history_survive_restart = true;
    report.restart_verified_at = new Date().toISOString();
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', same_model_and_history_after_restart: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure_masked.png'), fullPage: true, mask: [page.getByTestId('stImage'), page.locator('video')] });
  throw error;
} finally { await browser.close(); }
