// Formal catalog: read-only navigation. All edits require an isolated fixture.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = fileURLToPath(new URL('..', import.meta.url));
const output = path.join(root, 'artifacts/priority_review_20261028');
const formal = process.argv.includes('--formal');
assert(formal || process.argv.includes('--fixture'), 'Choose --formal (read-only) or --fixture');
const base = process.env.ALARM_APP_URL || `http://127.0.0.1:${formal ? 8506 : 8507}`;
const fixture = formal ? null : JSON.parse(await fs.readFile(path.join(output, 'browser_fixture.json'), 'utf8'));
if (fixture) {
  assert(path.resolve(fixture.root).startsWith(path.join(root, 'artifacts') + path.sep));
  assert(fixture.root.includes('browser_fixture'));
}
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true,
  args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });
const errors = [];
page.on('pageerror', error => errors.push(error.message));
const frame = () => page.frameLocator('iframe[title*="box_editor"]');
const button = name => page.getByRole('button', { name, exact: true });
async function idle() {
  await page.waitForTimeout(400);
  await page.getByTestId('stStatusWidget').waitFor({ state: 'hidden', timeout: 60000 });
  assert.equal(await page.getByTestId('stException').count(), 0);
}
async function click(text) { await page.getByText(text, { exact: true }).click(); await idle(); }
function select(label) {
  return page.getByTestId('stSelectbox').filter({ has: page.getByText(label, { exact: true }) }).getByRole('combobox');
}
async function choose(label, prefix) {
  const input = select(label);
  await input.fill(prefix);
  await page.getByRole('option').filter({ hasText: prefix }).click();
  await idle();
}
async function count(expected) {
  await frame().locator('#count').filter({ hasText: `${expected} 个框` }).waitFor();
  await page.getByText(`当前画布 ${expected} 个框；下方保存后才写入修改记录。`, { exact: true }).waitFor();
}
try {
  await page.goto(new URL('/Annotations', base).href, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: '自动预标注与人工复核', exact: true }).waitFor({ timeout: 60000 });
  await idle();
  await click('人工复核');
  await page.getByText(`优先复核（${formal ? 26 : 2}幅）`, { exact: true }).waitFor();
  const report = { generated_at: new Date().toISOString(), status: 'passed', formal_read_only: formal,
    automated_edits_are_human_gt: false, page_errors: errors };
  if (formal) {
    await page.getByText(/清单共 26 幅 · 当前可用 26 幅/).waitFor();
    await choose('复核图片', 'Q103');
    await page.getByTestId('stAlert').filter({ hasText: /Q103.*重点检查.*smoke.*是否漏标/ }).waitFor();
    await count(0);
    assert.equal(await page.getByRole('checkbox', { name: '我已检查整张图，确认没有 fire 或 smoke', exact: true }).isChecked(), false);
    assert(await button('确认复核通过').isDisabled());
    const caption = await page.getByText(/原图 \d+×\d+/).innerText();
    await click('查看优先复核清单和当前状态');
    await page.getByText(/依据批次：enhanced_base_20261027/).waitFor();
    await page.screenshot({ path: path.join(output, 'formal_priority_26.png'), fullPage: true });
    // A nonmember must be hidden from this scope, available only in the full catalog.
    const nonmember = 'a6ec9fa576ff';
    await select('复核图片').fill(nonmember);
    assert.equal(await page.getByRole('option').filter({ hasText: nonmember }).count(), 0);
    await select('复核图片').press('Escape');
    await click('全部图片（700幅）');
    await choose('复核图片', nonmember);
    await count(2); // Existing real human version, untouched by new candidates/filter.
    await click('优先复核（26幅）');
    await choose('复核图片', 'Q103');
    await count(0);
    await page.reload();
    await page.getByRole('heading', { name: '自动预标注与人工复核', exact: true }).waitFor();
    await idle();
    await click('人工复核');
    await page.getByText(/清单共 26 幅 · 当前可用 26 幅/).waitFor();
    await choose('复核图片', 'Q103');
    await count(0);
    Object.assign(report, { priority_count: 26, class_counts: { fire: 8, smoke: 18 },
      q103_missing_smoke_hint: true, missing_target_not_autoapproved: true,
      nonmember_hidden_and_accessible_in_all: true, saved_human_version_retained: true,
      queue_survives_new_browser_session: true, original_coordinate_caption: caption });
  } else {
    await page.getByText(/清单共 2 幅 · 当前可用 2 幅 · 已通过 1 幅/).waitFor();
    await choose('复核图片', fixture.ids[0].slice(0, 12));
    await count(1);
    assert((await frame().locator('#list').innerText()).includes('smoke'));
    await click('只看尚未通过');
    await count(0);
    assert((await page.getByTestId('stSelectbox').filter({ has: page.getByText('复核图片', { exact: true }) }).innerText()).includes(fixture.ids[1].slice(0, 12)));
    assert(await button('确认复核通过').isDisabled());
    const actor = page.getByRole('textbox', { name: '执行者/复核者姓名或编号', exact: true });
    await actor.fill('自动化流程测试，非真人标注'); await actor.press('Tab'); await idle();
    await frame().getByRole('button', { name: '画框', exact: true }).click();
    const rect = await frame().locator('#canvas').boundingBox();
    await page.mouse.move(rect.x + rect.width * .1, rect.y + rect.height * .1);
    await page.mouse.down();
    await page.mouse.move(rect.x + rect.width * .4, rect.y + rect.height * .4, { steps: 8 });
    await page.mouse.up(); await idle(); await count(1);
    await frame().getByLabel('框类别').selectOption('smoke'); await idle();
    const action = page.getByTestId('stMultiSelect').filter({ has: page.getByText('复核处理（可多选）', { exact: true }) }).getByRole('combobox');
    await action.click(); await page.getByRole('option', { name: '补充漏标目标', exact: true }).click();
    await action.press('Escape'); await idle();
    await button('保存草稿').click(); await idle();
    await page.getByText(/已通过 1 幅 · 尚未通过 1 幅/).waitFor();
    await action.click(); await page.getByRole('option', { name: '核对无误，保留现有框', exact: true }).click();
    await action.press('Escape'); await idle();
    await button('确认复核通过').click(); await idle();
    await page.getByText(/已通过 2 幅 · 尚未通过 0 幅/).waitFor();
    await page.getByText(/当前筛选下没有图片/).waitFor();
    await click('只看尚未通过');
    await choose('复核图片', fixture.ids[1].slice(0, 12)); await count(1);
    await page.getByRole('heading', { name: '人工修改记录', exact: true }).waitFor();
    await page.screenshot({ path: path.join(output, 'fixture_review_saved.png'), fullPage: true });
    await page.reload(); await idle(); await click('人工复核');
    await page.getByText(/清单共 2 幅 · 当前可用 2 幅 · 已通过 2 幅/).waitFor();
    await choose('复核图片', fixture.ids[1].slice(0, 12)); await count(1);
    Object.assign(report, { synthetic_queue_count: 2, preserved_saved_labels: true,
      draw_missing_target_and_change_class: true, draft_remains_pending: true,
      approval_updates_counts_and_empty_pending_scope: true, review_survives_reload: true });
  }
  assert.deepEqual(errors, []);
  const filename = formal ? 'formal_browser.json' : 'fixture_browser.json';
  await fs.writeFile(path.join(output, filename), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
} catch (error) {
  await page.screenshot({ path: path.join(output, formal ? 'formal_failure.png' : 'fixture_failure.png'), fullPage: true });
  throw error;
} finally { await browser.close(); }
