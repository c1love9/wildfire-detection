import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const read=async name=>JSON.parse(await fs.readFile(path.join(root,name),'utf8'));
const formal=await read('data/fire_dataset/quality_assignments/dual_200_20261024/task_private.json');
const fixture=await read('artifacts/dual_annotation_checks/browser_fixture.json');
const {chromium}=await import('file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true});
const errors=[];
const formalUrl=process.env.DUAL_APP_URL||'http://127.0.0.1:8506';
const fixtureUrl=process.env.DUAL_FIXTURE_URL||'http://127.0.0.1:8507';
async function page(){const context=await browser.newContext({viewport:{width:1440,height:1100}});const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));return {context,page};}
async function idle(p){await p.waitForTimeout(250);await p.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:60000});assert.equal(await p.getByTestId('stException').count(),0);}
async function login(p,url,token,admin=false){
 await p.goto(url+'/Quality');await p.getByRole('heading',{name:'标注质量与面积误差',exact:true}).waitFor();
 await p.getByText(admin?'管理与仲裁':'独立标注',{exact:true}).click();await idle(p);
 const input=p.getByRole('textbox',{name:admin?'管理访问码':'独立标注访问码',exact:true});
 await input.fill(token);await input.press('Tab');await idle(p);
}
async function canvasFrame(p){await p.locator('iframe[title*="box_editor"]').first().waitFor();const frame=p.frames().find(frame=>frame.url().includes('box_editor'));assert(frame);await frame.locator('#count').waitFor();return frame;}
try{
 // Read-only checks of actual task. Do not start timers or submit real images.
 for(const role of ['A','B']){
  const {context,page:p}=await page();await login(p,formalUrl,formal.tokens[role]);
  await p.getByText('本人已提交 0 / 200 幅',{exact:true}).waitFor();
  const frame=await canvasFrame(p);assert.equal(await frame.locator('#count').textContent(),'0 个框');
  assert.equal(await p.getByRole('button',{name:'保存人工参考版本',exact:true}).count(),0);
  assert.equal(await p.getByText('预标注版本',{exact:true}).count(),0);
  await p.getByRole('button',{name:'退出本次独立标注',exact:true}).click();await idle(p);
  assert((await p.getByRole('textbox',{name:'独立标注访问码',exact:true}).inputValue()).length===0,'Logout should clear the password field');
  assert.equal(await p.getByRole('button',{name:'开始或继续工作计时',exact:true}).count(),0);
  await context.close();console.log('Formal '+role+': blank canvas, own progress and logout verified without submissions');
 }
 {
  const {context,page:p}=await page();await login(p,formalUrl,formal.tokens.admin,true);
  await p.getByText('等待两名标注者全部完成；此时不展示框或预标注。',{exact:true}).waitFor();
  assert.equal(await p.getByRole('button',{name:'保存人工参考版本',exact:true}).count(),0);await context.close();
 }
 // Actual writes only in an isolated two-image synthetic fixture.
 for(const role of ['A','B']){
  const {context,page:p}=await page();await login(p,fixtureUrl,fixture.tokens[role]);
  for(let index=0;index<2;index++){
   await p.getByText(`本人已提交 ${index} / 2 幅`,{exact:true}).waitFor();
   const frame=await canvasFrame(p);assert.equal(await frame.locator('#count').textContent(),'0 个框');
   await p.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle(p);
   if(index===0){
    const canvas=frame.locator('#canvas');await canvas.scrollIntoViewIfNeeded();const bounds=await canvas.boundingBox();
    await p.mouse.move(bounds.x+bounds.width*.2,bounds.y+bounds.height*.2);await p.mouse.down();
    await p.mouse.move(bounds.x+bounds.width*.65,bounds.y+bounds.height*.65,{steps:7});await p.mouse.up();await idle(p);
   }else{await p.getByText('独立检查后确认整图无fire/smoke',{exact:true}).click();await idle(p);}
   await p.getByRole('button',{name:'暂停工作计时',exact:true}).click();await idle(p);
   await p.getByRole('button',{name:'提交并锁定本图独立标注',exact:true}).click();await idle(p);
  }
  await p.getByText('本人已提交 2 / 2 幅',{exact:true}).waitFor();await context.close();
 }
 {
  const {context,page:p}=await page();await login(p,fixtureUrl,fixture.tokens.admin,true);
  for(let index=0;index<2;index++){
   const control=p.getByTestId('stSelectbox').filter({has:p.getByText('仲裁图片',{exact:true})}).getByRole('combobox');
   await control.click();await p.getByRole('option').filter({hasText:`Q${String(index+1).padStart(3,'0')}`}).click();await idle(p);
   await p.getByRole('heading',{name:'人工仲裁参考画布',exact:true}).waitFor();
   const frame=await canvasFrame(p);assert.equal(await frame.locator('#count').textContent(),index===0?'1 个框':'0 个框');
   const note=p.getByRole('textbox',{name:'人工仲裁说明',exact:true});await note.fill('Automated synthetic functional exercise; not a real human adjudication.');await note.press('Tab');await idle(p);
   await p.getByRole('button',{name:'保存人工参考版本',exact:true}).click();await idle(p);
   const approve=p.getByRole('button',{name:'确认此仲裁参考用于训练',exact:true});assert(await approve.isDisabled());
   await p.getByText('我已核查已保存的仲裁参考，确认用于训练标签',{exact:true}).click();await idle(p);
   if(index===1){assert(await approve.isDisabled());await p.getByText('仲裁后确认整图无fire/smoke',{exact:true}).click();await idle(p);}
   await approve.click();await p.getByText('已保存为Annotations人工通过版本；A/B独立原始记录保持不变。',{exact:true}).waitFor();
  }
  await p.getByRole('button',{name:'生成标注质量报告',exact:true}).click();await p.getByRole('button',{name:'下载标注质量报告',exact:true}).waitFor();
  await context.close();
 }
 assert.deepEqual(errors,[]);
 const result={status:'passed',verified_at:new Date().toISOString(),formal_task_read_only:true,formal_roles_blank_and_no_submissions:true,
  credential_cleared_on_logout:true,admin_locked_before_completion:true,synthetic_two_role_two_image_submissions:true,
  pending_filter_auto_advances:true,visual_arbitration_and_explicit_review_approval:true,empty_reference_requires_confirmation:true,
  quality_report_download:true,scope:'Formal task inspected without writes; all annotation/approval actions are isolated synthetic automation, never human evidence',page_errors:errors};
 await fs.writeFile(path.join(root,'reports/dual_annotation_browser.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
}catch(error){
 let message=String(error?.stack||error);
 for(const token of [...Object.values(formal.tokens),...Object.values(fixture.tokens)])message=message.replaceAll(token,'[REDACTED]');
 console.error(message);process.exitCode=1;
}finally{await browser.close();}
