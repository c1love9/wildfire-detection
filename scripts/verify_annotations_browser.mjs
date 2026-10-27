// Public/synthetic fixtures only. Automated edits are not independent human GT.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/annotation_checks/browser');
const reportPath = path.join(root, 'reports/annotation_browser.json');
const fixture = JSON.parse(await fs.readFile(path.join(root, 'reports/annotation_verification.json'), 'utf8'));
const base = process.env.ALARM_APP_URL || 'http://127.0.0.1:8502';
const historyOnly = process.argv.includes('--history-only');
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: 'msedge', args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1400 }, acceptDownloads: true });
const errors = [];
const downloadTransports = [];
page.on('pageerror', error => errors.push(error.message));
const button = name => page.getByRole('button', { name, exact: true });
const frame = () => page.frameLocator('iframe[title*="box_editor"]');
async function idle() {
  await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 120000 });
  await page.waitForTimeout(450);
  assert.equal(await page.getByTestId('stException').count(), 0);
}
async function mode(name) {
  await page.getByText(name, { exact: true }).click();
  await idle();
}
async function choose(label, prefix) {
  const control = page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) });
  if ((await control.innerText()).includes(prefix)) return;
  await control.getByRole('combobox').click();
  await page.getByRole('option').filter({ hasText: prefix }).click();
  await idle();
}
async function note(text) {
  const input = page.getByRole('textbox', { name: '补充说明（选填）', exact: true });
  await input.fill(text);
  await input.press('Tab');
  await idle();
}
async function reviewAction(text) {
  const input = page.getByTestId('stMultiSelect').filter({ has: page.getByText('复核处理（可多选）', { exact: true }) }).getByRole('combobox');
  await input.click();
  await page.getByRole('option', { name: text, exact: true }).click();
  await input.press('Escape');
  await idle();
}
async function actor(text) {
  const input = page.getByRole('textbox', { name: '执行者/复核者姓名或编号', exact: true });
  await input.fill(text);
  await input.press('Tab');
  await idle();
}
async function count() {
  return Number((await frame().locator('#count').innerText()).split(' ')[0]);
}
async function expectCount(expected) {
  await frame().locator('#count').filter({ hasText: `${expected} 个框` }).waitFor();
  await page.getByText(`当前画布 ${expected} 个框；下方保存后才写入修改记录。`, { exact: true }).waitFor();
  await idle();
}
async function draw(from, to) {
  await frame().getByRole('button', { name: '画框', exact: true }).click();
  const rect = await frame().locator('#canvas').boundingBox();
  await page.mouse.move(rect.x + rect.width * from[0], rect.y + rect.height * from[1]);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width * to[0], rect.y + rect.height * to[1], { steps: 6 });
  await page.mouse.up();
  await idle();
}
async function download(filename) {
  const pending = page.waitForEvent('download');
  pending.catch(() => {});
  await button('下载所选导出 ZIP').click();
  let result = await pending;
  const destination = path.join(output, filename);
  let blobUrl;
  const failure = await result.failure();
  if (failure) {
    // This machine's IDM intercepts native ZIP navigation (204/canceled).
    // An explicit test-only option fetches the SAME server response in the
    // browser and saves its Blob, without changing app or system settings.
    assert(process.env.ANNOTATION_BLOB_DOWNLOAD_FALLBACK === '1', `Native download failed: ${failure}`);
    const url = result.url();
    assert.equal(new URL(url).origin, new URL(base).origin);
    // IDM also remembers intercepted URLs. Forward the real HTTP response
    // through Playwright's request client, preserving status/headers/bytes.
    await page.route(url, async route => {
      const response = await route.fetch();
      assert.equal(response.status(), 200);
      await route.fulfill({ response });
    });
    const replacement = page.waitForEvent('download');
    replacement.catch(() => {});
    blobUrl = await page.evaluate(async ({ url, filename }) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`ZIP HTTP ${response.status}`);
      const blob = await response.blob();
      if (!blob.size) throw new Error('Empty ZIP response');
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = filename;
      document.body.append(a); a.click(); a.remove();
      return a.href;
    }, { url, filename });
    result = await replacement;
    downloadTransports.push({ filename, native_failure: failure, transport: 'real HTTP 200 forwarded unchanged -> browser fetch/Blob download; IDM integration left unchanged' });
  } else downloadTransports.push({ filename, transport: 'native browser download' });
  await result.saveAs(destination);
  if (blobUrl) await page.evaluate(url => URL.revokeObjectURL(url), blobUrl);
  return { path: destination, sha256: crypto.createHash('sha256').update(await fs.readFile(destination)).digest('hex') };
}
function inspectZip(filename, training) {
  const script = `import json,sys,zipfile,math
with zipfile.ZipFile(sys.argv[1]) as z:
 m=json.loads(z.read('manifest.json')); a=json.loads(z.read('audit.json'))
 assert m['classes']=={'0':'fire','1':'smoke'}
 labels=[n for n in z.namelist() if n.startswith('labels/') and n.endswith('.txt')]
 assert len(labels)==3
 for name in labels:
  for line in z.read(name).decode().splitlines():
   p=line.split(); assert len(p)==5 and p[0] in ('0','1')
   v=list(map(float,p[1:])); assert all(math.isfinite(x) and 0<=x<=1 for x in v) and v[2]>0 and v[3]>0
 first=next(r for r in a if r['asset']['id']==sys.argv[2]); blank=next(r for r in a if r['asset']['id']==sys.argv[3])
 assert len(first['history'])==2 and first['history'][0]['decision']=='draft' and first['review']['decision']=='approved'
 diff=first['history'][0]['diff']; assert all(len(diff[k])>=1 for k in ['added','removed','changed'])
 assert first['history'][0]['reviewer']!=first['history'][1]['reviewer']
 assert '补充漏标目标' in first['history'][0]['note'] and '删除误检或重复框' in first['history'][0]['note']
 unchanged=next(r for r in a if r['asset']['id'] not in (sys.argv[2],sys.argv[3]))
 assert unchanged['review']['note']=='核对无误，保留现有框'
 assert blank['review']['negative_confirmed'] and not blank['review']['boxes']
 assert blank['review']['note']=='已检查整张图，确认没有 fire 或 smoke'
 assert z.read(next(n for n in labels if sys.argv[3] in n))==b''
 assert first['proposals'][0]['metadata']['runtime']['model_id']=='IDEA-Research/grounding-dino-tiny'
 if sys.argv[4]=='true':
  assert len([n for n in z.namelist() if n.startswith('images/') and n.endswith('.png')])==3
  assert '0: fire' in z.read('data.yaml').decode() and '1: smoke' in z.read('data.yaml').decode()
  s=json.loads(z.read('source_split.json')); assert s['image_counts']=={'train':1,'val':1,'test':1}
 print(json.dumps({'labels':len(labels),'draft_diff':{k:len(v) for k,v in diff.items()},'explicit_negative':True,'training':sys.argv[4]=='true'}))`;
  const result = spawnSync(path.join(root, '.venv/Scripts/python.exe'), ['-c', script, filename, fixture.browser_asset_ids[0], fixture.browser_asset_ids[2], String(training)], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
try {
  await page.goto(new URL('/Annotations', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '自动预标注与人工复核', exact: true }).waitFor({ timeout: 90000 });
  await idle();
  if (!historyOnly) {
    await actor('浏览器自动化测试A');
    await page.getByText('原方案：Tiny独立查询', { exact: true }).click();
    await idle();
    await button('开始自动预标注').click();
    await page.getByText('预标注任务结束，请进入人工复核；候选框未自动批准。', { exact: true }).waitFor({ timeout: 120000 });
    await idle();
    await mode('人工复核');
    console.log('Real browser prelabel complete; checking canvas edits');
    await choose('复核图片', 'public_scene_0.jpeg');
    await frame().locator('#canvas').waitFor();
    const initial = await count();
    assert(initial > 0);
    await draw([0.05, 0.05], [0.2, 0.2]);
    await expectCount(initial + 1);
    await frame().getByRole('combobox', { name: '框类别' }).selectOption('smoke');
    await idle();
    assert.equal(await frame().getByRole('combobox', { name: '框类别' }).inputValue(), 'smoke');
    const beforeResize = Number(await frame().getByRole('spinbutton', { name: 'x2', exact: true }).inputValue());
    // Resize the new box using a real pointer drag at its lower-right corner.
    await frame().getByRole('button', { name: '选择/移动', exact: true }).click();
    const rect = await frame().locator('#canvas').boundingBox();
    await page.mouse.move(rect.x + rect.width * 0.2, rect.y + rect.height * 0.2);
    await page.mouse.down();
    await page.mouse.move(rect.x + rect.width * 0.22, rect.y + rect.height * 0.22, { steps: 5 });
    await page.mouse.up();
    await idle();
    assert(Number(await frame().getByRole('spinbutton', { name: 'x2', exact: true }).inputValue()) > beforeResize + 5);
    await frame().locator('#list button').first().click();
    await frame().getByRole('button', { name: '删除选中框', exact: true }).click();
    await expectCount(initial);
    // Change an original proposal box so the recorded diff includes a move.
    await frame().locator('#list button').first().click();
    const x1 = frame().getByRole('spinbutton', { name: 'x1', exact: true });
    await x1.fill(String(Number(await x1.inputValue()) + 2));
    await x1.press('Tab');
    await idle();
    await draw([0.3, 0.05], [0.4, 0.15]);
    await expectCount(initial + 1);
    await frame().getByRole('button', { name: '撤销', exact: true }).click();
    await expectCount(initial);
    await page.screenshot({ path: path.join(output, 'canvas_edits.png'), fullPage: true });
    await reviewAction('补充漏标目标');
    await reviewAction('删除误检或重复框');
    await reviewAction('修改 fire/smoke 类别');
    await reviewAction('调整框的位置或大小');
    await note('自动化功能验证：补框、改类、删框与调整；不是正式人工参考标签。');
    await button('保存草稿').click();
    await page.getByText('已保存草稿，尚未通过复核。', { exact: true }).waitFor();
    await idle();
    await actor('浏览器自动化测试B');
    await note('自动化复核通过，仅用于验证版本链，不是独立人工GT。');
    await button('确认复核通过').click();
    await page.getByText('已保存人工复核版本。', { exact: true }).waitFor();
    await idle();
    await choose('复核图片', 'public_scene_1.jpeg');
    assert.equal(await page.getByRole('textbox', { name: '补充说明（选填）', exact: true }).inputValue(), '');
    assert(await button('确认复核通过').isDisabled(), 'Review actions must not carry over to another image');
    await reviewAction('核对无误，保留现有框');
    await button('确认复核通过').click();
    await idle();
    console.log('Canvas edits, draft and approval saved; checking explicit negative');
    await choose('复核图片', 'blank_fixture.png');
    if (await count()) {
      await frame().getByRole('button', { name: '清空所有框', exact: true }).click();
      await expectCount(0);
    }
    assert(await button('确认复核通过').isDisabled());
    await page.getByText('我已检查整张图，确认没有 fire 或 smoke', { exact: true }).click();
    await idle();
    await button('确认复核通过').click();
    await idle();
    await mode('YOLO导出');
    await button('导出已复核标签 ZIP').click();
    await idle();
    const labels = await download('labels.zip');
    const labelCheck = inspectZip(labels.path, false);
    await button('生成YOLO训练数据集').click();
    await idle();
    await choose('已保存导出', '训练数据');
    const training = await download('training.zip');
    const trainingCheck = inspectZip(training.path, true);
    await page.screenshot({ path: path.join(output, 'reviewed_yolo_export.png'), fullPage: true });
    assert.deepEqual(errors, []);
    const report = { generated_at: new Date().toISOString(), status: 'passed', browser: await browser.version(),
      scope: 'Public/synthetic images in an isolated catalog; automated browser actors are not human GT.',
      catalog_root: fixture.browser_root, real_model_batch_prelabel: true, original_proposal_boxes: initial,
      canvas_draw_class_change_pointer_resize_delete_numeric_move_undo: true,
      draft_and_approval_append_only: true, empty_approval_requires_explicit_confirmation: true,
      review_action_options_generate_notes: true, unchanged_and_negative_review_without_typing: true,
      review_actions_do_not_carry_between_images: true,
      labels_zip: labelCheck, training_zip: trainingCheck,
      labels_zip_sha256: labels.sha256, training_zip_sha256: training.sha256, download_transports: downloadTransports, page_errors: errors };
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report));
  } else {
    await mode('人工复核');
    await choose('复核图片', 'public_scene_0.jpeg');
    await page.getByRole('heading', { name: '人工修改记录', exact: true }).waitFor();
    assert(await count() > 0);
    await mode('YOLO导出');
    await choose('已保存导出', '训练数据');
    const downloaded = await download('training_after_restart.zip');
    const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));
    assert.equal(downloaded.sha256, report.training_zip_sha256);
    inspectZip(downloaded.path, true);
    await page.screenshot({ path: path.join(output, 'after_restart.png'), fullPage: true });
    await page.goto(new URL('/Demo_Videos', base).href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: '视频火焰与烟雾检测', exact: true }).waitFor({ timeout: 60000 });
    await idle();
    assert.deepEqual(errors, []);
    report.annotations_and_identical_export_survive_restart = true;
    report.navigation_to_detection = true;
    report.restart_verified_at = new Date().toISOString();
    report.restart_download_transports = downloadTransports;
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', persistent_reviews_and_identical_export: true }));
  }
} catch (error) {
  await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
  throw error;
} finally { await browser.close(); }
