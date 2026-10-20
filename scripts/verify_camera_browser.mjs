// Real-camera UI test: disable live recording; only public-fixture media is saved.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/camera_checks/browser');
const reportPath = path.join(root, 'reports/camera_browser.json');
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8501';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--disable-gpu'],
  ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : { channel: 'msedge' }) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1280 } });
const errors = [], mediaErrors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('response', response => { if (response.url().includes('/media/') && response.status() >= 400) mediaErrors.push(response.status()); });
const button = name => page.getByRole('button', { name, exact: true });
const frameMetric = () => page.getByTestId('stMetric').filter({ hasText: '本段已检测帧' }).getByTestId('stMetricValue');
async function frames() { return Number((await frameMetric().innerText()).replaceAll(',', '')); }
async function waitFrames(minimum) {
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (await frameMetric().count() && await frames() >= minimum) return;
    if (await page.getByText(/实时检测采样间隔过大/).count()) throw new Error('Camera observation gap exceeded limit');
    await page.waitForTimeout(100);
  }
  throw new Error('Real camera did not produce enough detected frames');
}
try {
  let pausedFrames;
  if (!historyOnly) {
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
    await page.getByText('摄像头', { exact: true }).click();
    await page.getByRole('spinbutton', { name: '摄像头编号', exact: true }).waitFor();
    await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 30000 });
    await page.waitForTimeout(500); // source change inserts controls before expanders
    const recording = page.getByTestId('stExpander').filter({ has: page.getByText('报警截图与片段', { exact: true }) });
    if (!await recording.locator('details').evaluate(details => details.open)) await recording.locator('summary').click();
    await recording.locator('details[open]').waitFor();
    await page.getByTestId('stCheckbox').locator('label').click();
    await page.waitForFunction(() => document.querySelector('input[aria-label="保存报警截图和片段"]').checked === false);
    await button('开始检测').click();
    await waitFrames(12);
    assert(await page.getByTestId('stImage').locator('img').evaluate(async image => { await image.decode(); return image.naturalWidth > 0; }));
    await button('暂停检测').click();
    await page.getByText(/摄像头 0 · 已暂停/).waitFor();
    await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 30000 });
    pausedFrames = await frames();
    await page.waitForTimeout(700);
    assert.equal(await frames(), pausedFrames);
    await button('继续检测').click();
    await page.getByText(/观测段 2/).waitFor({ timeout: 30000 });
    await waitFrames(5);
    await button('停止检测').click();
    await page.getByText(/摄像头 0 · 已停止/).waitFor();
    assert(await button('开始检测').isEnabled());
  }
  await page.goto(new URL('/Alarm_History', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '报警日志', exact: true }).waitFor({ timeout: 60000 });
  await page.getByRole('combobox').nth(0).click();
  await page.getByRole('option', { name: '模拟摄像头（公开样例验证）', exact: true }).click();
  await page.getByRole('heading', { name: '报警截图与片段', exact: true }).waitFor();
  await page.getByText(/摄像头时间从本段开始观测起计时/).waitFor();
  assert(await page.getByTestId('stImage').locator('img').evaluate(async image => { await image.decode(); return image.naturalWidth > 0; }));
  const video = page.locator('video');
  await video.waitFor();
  await page.waitForFunction(() => { const video=document.querySelector('video'); return video && video.readyState >= 2 && video.duration > 1; });
  const duration = await video.evaluate(video => video.duration);
  assert(Math.abs(duration - 15) < 0.1);
  await video.evaluate(async video => { video.muted = true; await video.play(); });
  await page.waitForFunction(() => document.querySelector('video').currentTime > 0.3);
  await video.evaluate(async video => { video.pause(); await new Promise(resolve => { video.addEventListener('seeked', resolve, { once: true }); video.currentTime = video.duration * 0.75; }); });
  assert.equal(await video.evaluate(video => video.error), null);
  await page.getByRole('heading', { name: '报警截图与片段', exact: true }).evaluate(element => { element.style.scrollMarginTop = '80px'; element.scrollIntoView({ block: 'start' }); });
  await page.screenshot({ path: path.join(output, historyOnly ? 'public_fixture_after_restart.png' : 'public_fixture_playback.png'), fullPage: true });
  assert.equal(await page.getByTestId('stException').count(), 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(mediaErrors, []);
  if (!historyOnly) {
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      physical_camera_frames_before_pause: pausedFrames, physical_camera_pause_resume_stop: true,
      physical_camera_images_saved: false, controlled_public_fixture_history: true,
      camera_observation_time_label: true, controlled_clip_duration: duration, screenshot_decoded: true,
      playback_advances: true, seeking_works: true, page_errors: errors, media_errors: mediaErrors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    report.camera_history_and_media_survive_restart = true;
    report.restart_verified_at = new Date().toISOString();
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', camera_history_and_media_survive_restart: true }));
  }
} catch (error) {
  // Never persist an actual webcam image, including when a UI assertion fails.
  await page.screenshot({ path: path.join(output, 'failure_masked.png'), fullPage: true, mask: [page.getByTestId('stImage'), page.locator('video')] });
  throw error;
} finally {
  await browser.close();
}
