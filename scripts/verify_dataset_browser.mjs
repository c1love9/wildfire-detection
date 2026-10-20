// Synthetic/public inputs only, against an isolated dataset catalog.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/dataset_checks/browser');
const reportPath = path.join(root, 'reports/dataset_browser.json');
const verification = JSON.parse(await fs.readFile(path.join(root, 'reports/dataset_verification.json'), 'utf8'));
const sources = path.join(root, path.dirname(verification.root), 'sources');
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8502';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: 'msedge', args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1280 }, acceptDownloads: true });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const button = label => page.getByRole('button', { name: label, exact: true });
const tab = label => page.getByRole('tab', { name: label, exact: true });
async function idle() {
  await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 60000 });
  await page.waitForTimeout(400);
}
async function select(label, value) {
  const control = page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) });
  if ((await control.innerText()).replace(/\s+/g, ' ').includes(value.replace(/\s+/g, ' '))) return;
  await control.getByRole('combobox').click();
  await page.getByRole('option', { name: value, exact: true }).click();
}
async function batch(index, origin = '自采') {
  await tab('导入与采集').click();
  const expander = page.getByTestId('stExpander').filter({ has: page.getByText('新建采集批次', { exact: true }) });
  if (!await expander.locator('details').evaluate(element => element.open)) await expander.locator('summary').click();
  await page.getByRole('textbox', { name: '批次名称', exact: true }).fill('浏览器合成样例 ' + index);
  await page.getByRole('textbox', { name: '采集组编号', exact: true }).fill('browser_fixture_' + index);
  await page.getByRole('textbox', { name: '来源说明', exact: true }).fill('Synthetic/public verification fixture; not real self-collected internship data');
  await select('素材来源', origin);
  await button('创建批次').click();
  await page.getByText('已创建批次。', { exact: true }).waitFor();
  await idle();
}
async function upload(paths) {
  await page.getByTestId('stFileUploader').locator('input[type=file]').setInputFiles(paths);
  await idle();
  await button('导入上传文件').click();
  await idle();
}
async function metric(label) {
  return Number((await page.getByTestId('stMetric').filter({ has: page.getByText(label, { exact: true }) }).getByTestId('stMetricValue').innerText()).replaceAll(',', ''));
}
async function download(label, filename) {
  const pending = page.waitForEvent('download');
  await button(label).click();
  const result = await pending;
  const destination = path.join(output, filename);
  await result.saveAs(destination);
  return fs.readFile(destination, 'utf8');
}
try {
  await page.goto(new URL('/Dataset', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '数据采集与管理', exact: true }).waitFor({ timeout: 60000 });
  await idle();
  if (!historyOnly) {
    for (let index = 0; index < 3; index++) {
      await batch(index);
      await upload([path.join(sources, `${index * 2}.png`), path.join(sources, `${index * 2 + 1}.png`)]);
      if (index === 0) { await button('导入上传文件').click(); await idle(); }
      await tab('统计与划分').click();
      assert.equal(await metric('去重图片'), (index + 1) * 2);
    }
    await button('生成划分预览').click();
    await idle();
    await button('保存本次划分版本').click();
    await page.getByText(/已保存 split_/).waitFor();
    await idle();
  } else {
    await tab('统计与划分').click();
  }
  const manifestText = await download('下载 manifest.json', historyOnly ? 'manifest_after_restart.json' : 'manifest.json');
  const manifest = JSON.parse(manifestText);
  assert.equal(manifest.assets.length, 6);
  assert.deepEqual(manifest.image_counts, { train: 2, val: 2, test: 2 });
  const groups = new Map();
  for (const asset of manifest.assets) for (const group of asset.groups) {
    if (!groups.has(group)) groups.set(group, new Set());
    groups.get(group).add(asset.split);
  }
  assert([...groups.values()].every(splits => splits.size === 1));
  const lists = [];
  for (const name of ['train', 'val', 'test']) lists.push((await download('下载 ' + name + '.txt', `${name}.txt`)).trim().split('\n'));
  assert.equal(new Set(lists.flat()).size, 6);
  const manifestHash = crypto.createHash('sha256').update(manifestText).digest('hex');
  if (!historyOnly) {
    await batch('public_video', '演示/测试');
    const video = path.join(root, 'demo-videos/fire1.mov');
    await upload([video]);
    await tab('素材与抽帧').click();
    const videoId = crypto.createHash('sha256').update(await fs.readFile(video)).digest('hex');
    await idle();
    await select('选择归档素材', `视频 · ${videoId.slice(0, 16)}`);
    await idle();
    const maximum = page.getByRole('spinbutton', { name: '单次抽帧上限', exact: true });
    await maximum.fill('3');
    await maximum.press('Enter');
    await idle();
    await button('从所选视频抽帧').click();
    await page.getByText(/抽取 3 帧/).waitFor();
    await idle();
    await tab('统计与划分').click();
    assert.equal(await metric('去重视频'), 1);
    assert(await button('保存本次划分版本').isDisabled());
    await page.getByText(/这是历史快照/).waitFor();
    const stats = JSON.parse(await download('导出数量与场景统计 JSON', 'statistics.json'));
    assert.equal(stats.own_fire_videos, 0);
    assert.equal(stats.eligible_assets, 6);
    assert.equal(stats.excluded_demo_or_test, 4);
    const csv = await download('导出来源清单 CSV', 'sources.csv');
    assert(csv.includes('source_sha256') && csv.includes('fire1.mov') && csv.includes('0.png'));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    assert.equal(manifestHash, report.saved_manifest_sha256);
    await page.getByText(/这是历史快照/).waitFor();
  }
  assert.equal(await page.getByTestId('stException').count(), 0);
  await page.screenshot({ path: path.join(output, historyOnly ? 'after_restart.png' : 'statistics_and_frozen_split.png'), fullPage: true });
  // Smoke check navigation without saving any alarms or camera frames.
  await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
  await idle();
  assert.equal(await page.getByTestId('stException').count(), 0);
  assert.deepEqual(errors, []);
  if (!historyOnly) {
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      scope: 'Synthetic/public fixtures in an isolated catalog; no real internship data collected.',
      catalog_root: verification.browser_root,
      multi_image_upload_and_batch_creation: true, duplicate_upload_count_stable: true,
      public_video_upload_and_frame_extraction: true, demo_excluded_from_formal_counts: true,
      source_csv_and_statistics_download: true, split_manifest_and_lists_download: true,
      grouped_images_disjoint: true, split_images: manifest.image_counts,
      new_data_marks_version_historical_and_disables_old_preview: true,
      saved_manifest_sha256: manifestHash, navigation_to_detection: true, page_errors: errors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    report.catalog_counts_and_frozen_manifest_survive_restart = true;
    report.restart_verified_at = new Date().toISOString();
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', catalog_and_split_survive_restart: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  throw error;
} finally { await browser.close(); }
