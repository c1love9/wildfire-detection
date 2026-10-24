import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const fixture=JSON.parse(await fs.readFile(path.join(root,'artifacts/annotation_submit_checks/fixture_private.json'),'utf8'));
const {chromium}=await import('file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({headless:true,channel:'msedge'});const errors=[];
async function idle(p){await p.waitForTimeout(250);await p.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:60000});assert.equal(await p.getByTestId('stException').count(),0);}
async function login(port,role='A'){
 const context=await browser.newContext({viewport:{width:1440,height:1100}});const p=await context.newPage();p.on('pageerror',e=>errors.push(e.message));
 await p.goto(`http://127.0.0.1:${port}/Quality`);await p.getByRole('heading',{name:'标注质量与面积误差',exact:true}).waitFor();
 await p.getByText('独立标注',{exact:true}).click();await idle(p);
 const input=p.getByRole('textbox',{name:'独立标注访问码',exact:true});await input.fill(fixture.tokens[role]);await input.press('Tab');await idle(p);
 return {context,p};
}
async function draw(p){
 const frame=p.frames().find(f=>f.url().includes('box_editor'));assert(frame);const canvas=frame.locator('#canvas');await canvas.scrollIntoViewIfNeeded();
 const r=await canvas.boundingBox();await p.mouse.move(r.x+r.width*.15,r.y+r.height*.15);await p.mouse.down();
 await p.mouse.move(r.x+r.width*.7,r.y+r.height*.7,{steps:7});await p.mouse.up();await idle(p);
 assert.equal(await frame.locator('#count').textContent(),'1 个框');
}
async function select(p,number){
 const select=p.getByTestId('stSelectbox').filter({has:p.getByText('本人的待标注图片',{exact:true})}).getByRole('combobox');
 await select.click();await p.getByRole('option').filter({hasText:'Q'+String(number).padStart(3,'0')}).click();await idle(p);
}
try{
 // Old source is a retained copy. All actions use isolated synthetic data.
 {
  const {context,p}=await login(8508);
  await p.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle(p);await draw(p);
  assert(await p.getByRole('button',{name:'提交并锁定本图独立标注',exact:true}).isDisabled());
  await select(p,2);await select(p,1);
  assert(await p.getByRole('button',{name:'提交并锁定本图独立标注',exact:true}).isEnabled());
  await context.close();console.log('Reproduced: old page blocked active timer, switching away/back paused it and enabled submit.');
 }
 {
  const {context,p}=await login(8509);const submit=()=>p.getByRole('button',{name:'提交并锁定本图独立标注',exact:true});
  assert(await submit().isDisabled());
  await p.getByText('独立检查后确认整图无fire/smoke',{exact:true}).click();await idle(p);
  assert(await submit().isDisabled());await p.getByText(/暂不能提交：请先点击/).waitFor();
  await p.getByText('独立检查后确认整图无fire/smoke',{exact:true}).click();await idle(p);
  await p.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle(p);await draw(p);
  assert(await submit().isEnabled());
  await p.screenshot({path:path.join(root,'artifacts/annotation_submit_checks/submit_while_timing.png'),fullPage:true});
  await submit().click();await idle(p);await p.getByText('本人已提交 1 / 2 幅',{exact:true}).waitFor();
  await p.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle(p);
  assert(await submit().isDisabled());await p.getByText(/请画出目标框/).waitFor();
  await p.getByText('独立检查后确认整图无fire/smoke',{exact:true}).click();await idle(p);
  assert(await submit().isEnabled());await submit().click();await idle(p);
  await p.getByText('本人已提交 2 / 2 幅',{exact:true}).waitFor();
  await p.getByText('本人的全部图片已提交。请退出，等待另一位标注者独立完成。',{exact:true}).waitFor();
  await context.close();
 }
 {
  const {context,p}=await login(8509,'B');await p.getByText('本人已提交 0 / 2 幅',{exact:true}).waitFor();
  const frame=p.frames().find(f=>f.url().includes('box_editor'));assert.equal(await frame.locator('#count').textContent(),'0 个框');await context.close();
 }
 assert.deepEqual(errors,[]);
 const report={status:'passed',verified_at:new Date().toISOString(),scope:'Old/new browser reproduction on isolated synthetic two-image task; no real user task submitted or altered',
  old_active_timer_disabled_submit:true,old_switch_away_and_back_enabled_submit:true,
  active_timer_direct_submit_without_pause_or_navigation:true,empty_image_confirmation_still_required:true,
  never_started_timer_still_blocked_with_reason:true,next_image_progress:true,other_role_stays_blank:true,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/blind_submit_browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}catch(error){let message=String(error?.stack||error);for(const token of Object.values(fixture.tokens))message=message.replaceAll(token,'[REDACTED]');console.error(message);process.exitCode=1;}
finally{await browser.close();}
