// Optional browser QA: requires Playwright and Edge; no application dependency.
// PLAYWRIGHT_MODULE may be a file: URL for an existing Playwright installation.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/alarm_checks/browser');
const reportPath = path.join(root, 'reports/alarm_browser.json');
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8501';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--disable-gpu'],
  ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : { channel: 'msedge' }) });
const page = await browser.newPage({ viewport: { width: 1440, height: 1280 }, acceptDownloads: true });
const errors = [], mediaErrors = [];
page.on('pageerror', error => errors.push(error.message));
page.on('response', response => {
  if (response.url().includes('/media/') && response.status() >= 400) mediaErrors.push(response.status());
});
const button = name => page.getByRole('button', { name, exact: true });
async function settled() {
  await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 30000 });
  await page.waitForTimeout(700);
}
async function frameNumber() {
  return Number((await page.getByTestId('stMetric').filter({ hasText: '当前帧' }).getByTestId('stMetricValue').innerText()).match(/^\d+/)[0]);
}
async function imageDecoded() {
  assert(await page.getByTestId('stImage').locator('img').evaluate(async image => {
    await image.decode();
    return image.naturalWidth > 0 && image.src.startsWith('data:image/jpeg;base64,');
  }));
}
async function historyExport() {
  await page.goto(new URL('/Alarm_History', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '报警日志', exact: true }).waitFor({ timeout: 45000 });
  await page.getByRole('combobox').nth(1).click();
  await page.getByRole('option', { name: '报警', exact: true }).click();
  const downloaded = page.waitForEvent('download');
  await button('导出当前筛选 CSV').click();
  const download = await downloaded;
  const destination = path.join(output, historyOnly ? 'history_after_restart.csv' : 'history.csv');
  await download.saveAs(destination);
  const csv = await fs.readFile(destination, 'utf8');
  assert(csv.includes('fire1.mov') && csv.includes('最高等级') && csv.includes('手动停止'));
  assert.equal(csv.trim().split(/\r?\n/).length, 2, 'Expected one event, not one log per frame');
  await settled();
  await page.screenshot({ path: path.join(output, historyOnly ? 'history_after_restart.png' : 'history.png'), fullPage: true });
}

try {
  if (!historyOnly) {
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
    await page.evaluate(() => document.fonts.ready);
    await button('开始检测').click();
    for (const level of ['关注', '预警', '报警']) {
      await page.getByText(new RegExp(`当前报警等级：${level}`)).waitFor({ timeout: 60000 });
    }
    await button('暂停检测').click();
    await button('继续检测').waitFor();
    await page.getByText(/当前报警等级：报警.*检测已暂停/).waitFor();
    await settled();
    const pausedFrame = await frameNumber();
    await page.waitForTimeout(800);
    assert.equal(await frameNumber(), pausedFrame);
    await imageDecoded();
    await page.getByText(/当前报警等级：报警/).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, 'paused_alarm.png'), fullPage: true });
    await button('继续检测').click();
    await page.waitForFunction(minimum => {
      const metric = [...document.querySelectorAll('[data-testid="stMetric"]')].find(element => element.innerText.includes('当前帧'));
      return metric && Number(metric.querySelector('[data-testid="stMetricValue"]').innerText.match(/^\d+/)[0]) > minimum;
    }, pausedFrame + 1);
    await button('停止检测').click();
    await page.getByText(/当前报警等级：报警.*检测已结束/).waitFor();
    await page.getByRole('combobox').nth(0).click();
    await page.getByRole('option', { name: 'fire3.mp4', exact: true }).click();
    await button('开始检测').click();
    await page.getByText(/检测完成：共处理64帧/).waitFor({ timeout: 60000 });
    await page.getByText(/当前报警等级：正常.*检测已结束/).waitFor();
    assert.equal(await frameNumber(), 64);
    await imageDecoded();
    await historyExport();
    assert.equal(await page.getByTestId('stException').count(), 0);
    assert.deepEqual(errors, []);
    assert.deepEqual(mediaErrors, []);
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      default_rules_reach_all_three_levels: true, paused_frame: pausedFrame, pause_keeps_alarm_and_frame: true,
      resume_advances: true, stop_preserves_last_alarm: true, new_video_resets_evidence: true,
      short_detections_at_eof_do_not_confirm: true, persistent_history_filter: true,
      csv_download_has_one_event: true, annotated_images_decoded: true, page_errors: errors, failed_media_requests: mediaErrors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    await historyExport();
    assert.equal(await page.getByTestId('stException').count(), 0);
    assert.deepEqual(errors, []);
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    report.history_survives_server_restart = true;
    report.restart_verified_at = new Date().toISOString();
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', history_survives_server_restart: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  throw error;
} finally {
  await browser.close();
}
