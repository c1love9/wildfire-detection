// Uses public fixtures in an isolated DB; never opens a physical camera.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/monitoring_checks/browser');
const reportPath = path.join(root, 'reports/monitoring_browser.json');
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8502';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: 'msedge',
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1280 } });
const errors = [], mediaErrors = [], styles = [], tiles = [];
page.on('pageerror', error => errors.push(error.message));
page.on('response', response => {
  const url = response.url();
  if (url.includes('/media/') && response.status() >= 400) mediaErrors.push(response.status());
  if (url.includes('basemaps.cartocdn.com') && url.includes('style.json')) styles.push(response.status());
  if (url.includes('basemaps.cartocdn.com') && /\.(pbf|mvt)/.test(url)) tiles.push(response.status());
});
const button = name => page.getByRole('button', { name, exact: true });
async function idle(target = page) {
  await target.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 60000 });
  await target.waitForTimeout(500);
}
async function choose(label, option) {
  await page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) }).getByRole('combobox').click();
  await page.getByRole('option', { name: option, exact: true }).click();
  await idle();
}
async function mapReady(target = page) {
  const chart = target.getByTestId('stDeckGlJsonChart');
  await chart.waitFor({ timeout: 30000 });
  await chart.scrollIntoViewIfNeeded();
  await chart.locator('canvas').first().waitFor();
  await target.waitForTimeout(1500);
  assert(await chart.locator('canvas').first().evaluate(canvas => canvas.width > 0 && canvas.height > 0));
  return chart;
}
async function replay() {
  await page.getByRole('heading', { name: '报警截图与片段', exact: true }).waitFor();
  assert(await page.getByTestId('stImage').locator('img').evaluate(async image => { await image.decode(); return image.naturalWidth > 0; }));
  const video = page.locator('video');
  await video.waitFor();
  await page.waitForFunction(() => { const video = document.querySelector('video'); return video?.readyState >= 2 && video.duration > 1; });
  await video.evaluate(async video => { video.muted = true; await video.play(); });
  await page.waitForFunction(() => document.querySelector('video').currentTime > 0.3);
  await video.evaluate(video => video.pause());
  assert.equal(await video.evaluate(video => video.error), null);
}
try {
  let mapPanned = false;
  if (!historyOnly) {
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
    await idle();
    const expander = page.getByTestId('stExpander').filter({ has: page.getByText('监控点位置与视角（模拟）', { exact: true }) });
    await expander.locator('summary').click();
    await choose('模拟监控点', '演示监控点 B');
    assert.equal(await page.getByRole('spinbutton', { name: '模拟经度（WGS84）', exact: true }).inputValue(), '121.475200');
    const heading = page.getByRole('spinbutton', { name: '模拟水平角（度）', exact: true });
    await heading.fill('90');
    await heading.press('Enter');
    await idle();
    const pitch = page.getByRole('spinbutton', { name: '模拟俯仰角（度）', exact: true });
    await pitch.fill('-30');
    await pitch.press('Enter');
    await idle();
    assert(await page.getByTestId('stMetric').filter({ hasText: '水平角' }).getByText('90°', { exact: true }).count());
    const chart = await mapReady();
    const before = await chart.screenshot();
    const bounds = await chart.boundingBox();
    await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(bounds.x + bounds.width / 2 + 100, bounds.y + bounds.height / 2 + 30, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(700);
    mapPanned = !before.equals(await chart.screenshot());
    assert(mapPanned, 'Dragging should change the map image');
    await page.screenshot({ path: path.join(output, 'live_simulated_pose.png'), fullPage: true });
    await button('开始检测').click();
    await page.getByText(/当前报警等级：/).first().waitFor({ timeout: 60000 });
    assert(await heading.isDisabled());
    await page.getByText(/检测完成：共处理294帧/).waitFor({ timeout: 90000 });
    await idle();
    assert(await button('开始检测').isEnabled());
  }
  await page.goto(new URL('/Alarm_History', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '报警日志', exact: true }).waitFor({ timeout: 60000 });
  await idle();
  for (const [source, lon] of [['公开样例 · 监控点 A', '121.473700'], ['模拟摄像头 · 监控点 B', '121.475200']]) {
    await choose('输入来源筛选', source);
    await page.getByText(new RegExp(`模拟坐标 WGS84 · 经度 ${lon}`)).waitFor();
    await mapReady();
    await replay();
  }
  await mapReady();
  await page.screenshot({ path: path.join(output, historyOnly ? 'history_after_restart.png' : 'history_point_b.png'), fullPage: true });
  const offline = page.getByTestId('stCheckbox').filter({ hasText: '显示在线地图底图' });
  await offline.locator('label').click();
  await idle();
  await mapReady();
  assert.equal(await offline.locator('input').isChecked(), false);
  // A separate context starts with no cached tiles and blocks map providers.
  // Use the idle detection preview so no media player navigates mid-request.
  const offlineContext = await browser.newContext({ viewport: { width: 1440, height: 1280 } });
  const offlinePage = await offlineContext.newPage();
  offlinePage.on('pageerror', error => errors.push(error.message));
  await offlinePage.route(/https:\/\/[^/]*(cartocdn|mapbox)[^/]*\//, route => route.abort());
  await offlinePage.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
  await offlinePage.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
  await idle(offlinePage);
  const offlineFresh = offlinePage.getByTestId('stCheckbox').filter({ hasText: '显示在线地图底图' });
  await offlineFresh.locator('label').click();
  await idle(offlinePage);
  await mapReady(offlinePage);
  assert.equal(await offlinePage.getByTestId('stException').count(), 0);
  await offlinePage.screenshot({ path: path.join(output, 'without_online_basemap.png'), fullPage: true });
  await offlineContext.close();
  assert.equal(await page.getByTestId('stException').count(), 0);
  assert.deepEqual(errors, []);
  assert.deepEqual(mediaErrors, []);
  assert(styles.includes(200), `No successful map style: ${styles}`);
  assert(tiles.some(status => status === 200 || status === 304), `No successful map tiles: ${tiles}`);
  if (!historyOnly) {
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      simulated_point_switch: true, manual_heading_and_pitch: true, controls_locked_during_detection: true,
      real_model_video_frames: 294, map_canvas_visible: true, map_pan_changes_render: mapPanned,
      online_style_loaded: true, online_tiles_loaded: true, basemap_toggle_works: true,
      source_free_map_with_map_services_blocked: true,
      historical_points_match_archived_coordinates: true, file_and_camera_event_media_replay: true,
      page_errors: errors, media_errors: mediaErrors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    report.archived_map_and_media_survive_restart = true;
    report.history_first_startup_and_detection_page_preview = true;
    report.restart_verified_at = new Date().toISOString();
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', archived_map_and_media_survive_restart: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  throw error;
} finally {
  await browser.close();
}
