// Optional Playwright + Edge test, against an isolated, initially empty log DB.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/recording_checks/browser');
const reportPath = path.join(root, 'reports/recording_browser.json');
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8501';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--disable-gpu'],
  ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : { channel: 'msedge' }) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1280 } });
const errors = [], mediaErrors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('response', response => {
  if (response.url().includes('/media/') && response.status() >= 400) mediaErrors.push(response.status());
});
const button = name => page.getByRole('button', { name, exact: true });
try {
  if (!historyOnly) {
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
    await button('开始检测').click();
    await page.getByText(/当前报警等级：报警/).waitFor({ timeout: 60000 });
    await button('暂停检测').click();
    await page.getByText(/当前报警等级：报警.*检测已暂停/).waitFor();
    await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 30000 });
    const metric = () => page.getByTestId('stMetric').filter({ hasText: '当前帧' }).getByTestId('stMetricValue');
    const paused = await metric().innerText();
    await page.waitForTimeout(700);
    assert.equal(await metric().innerText(), paused);
    await button('继续检测').click();
    await page.getByText(/检测完成：共处理294帧/).waitFor({ timeout: 60000 });
  }
  await page.goto(new URL('/Alarm_History', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '报警日志', exact: true }).waitFor({ timeout: 60000 });
  await page.getByRole('heading', { name: '报警截图与片段', exact: true }).waitFor({ timeout: 45000 });
  const screenshot = page.getByTestId('stImage').locator('img');
  assert(await screenshot.evaluate(async image => { await image.decode(); return image.naturalWidth > 0 && image.naturalHeight > 0; }));
  const video = page.locator('video');
  await video.waitFor({ timeout: 45000 });
  await page.waitForFunction(() => {
    const video = document.querySelector('video');
    return video && video.readyState >= 2 && Number.isFinite(video.duration) && video.duration > 1;
  });
  const duration = await video.evaluate(video => video.duration);
  assert(Math.abs(duration - 9.8) < 0.1, `Unexpected clip duration ${duration}`);
  await video.scrollIntoViewIfNeeded();
  await video.evaluate(async video => { video.muted = true; video.currentTime = 0; await video.play(); });
  await page.waitForFunction(() => document.querySelector('video').currentTime > 0.3);
  await video.evaluate(async video => {
    video.pause();
    await new Promise(resolve => { video.addEventListener('seeked', resolve, { once: true }); video.currentTime = video.duration * 0.75; });
  });
  assert.equal(await video.evaluate(video => video.error), null);
  assert(await video.evaluate(video => video.currentTime > video.duration / 2));
  await page.getByText(/确认前画面不足.*确认后画面不足/).waitFor();
  await page.getByRole('heading', { name: '报警截图与片段', exact: true }).evaluate(element => {
    element.style.scrollMarginTop = '80px';
    element.scrollIntoView({ block: 'start' });
  });
  await page.screenshot({ path: path.join(output, historyOnly ? 'playback_after_restart.png' : 'playback.png'), fullPage: true });
  assert.equal(await page.getByTestId('stException').count(), 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(mediaErrors, []);
  if (!historyOnly) {
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      real_model_video_detected_frames: 294, pause_resume_preserves_frame: true, screenshot_decoded: true,
      video_loaded: true, video_duration: duration, video_playback_advances: true, seeking_works: true,
      truncated_window_explained: true, page_errors: errors, media_errors: mediaErrors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    report.screenshot_and_playback_survive_restart = true;
    report.restart_verified_at = new Date().toISOString();
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', screenshot_and_playback_survive_restart: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  throw error;
} finally {
  await browser.close();
}
