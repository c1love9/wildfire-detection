import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const read=async name=>JSON.parse(await fs.readFile(path.join(root,name),'utf8'));
const fixture=await read('artifacts/dual_annotation_checks/browser_fixture.json');
const formal=process.env.DUAL_SKIP_FORMAL==='1'?null:await read('data/fire_dataset/quality_assignments/dual_200_20261024/task_private.json');
const {chromium}=await import('file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true});
const errors=[];
const externalDownloads=[];
const fixtureUrl=process.env.DUAL_FIXTURE_URL||'http://127.0.0.1:8507';
const formalUrl=process.env.DUAL_APP_URL||'http://127.0.0.1:8506';
const downloads=path.join(root,'artifacts/dual_annotation_checks/browser_downloads');
await fs.mkdir(downloads,{recursive:true});
async function page(){const context=await browser.newContext({viewport:{width:1440,height:1100},acceptDownloads:true});const p=await context.newPage();p.on('pageerror',e=>errors.push(e.message));return {context,p};}
async function idle(p){await p.waitForTimeout(350);await p.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:60000});assert.equal(await p.getByTestId('stException').count(),0);}
async function login(p,url,token,admin=false){
 await p.goto(url+'/Quality');await p.getByRole('heading',{name:'标注质量与面积误差',exact:true}).waitFor();
 assert.equal(await p.getByText('管理与仲裁',{exact:true}).count(),0);
 await p.getByText(admin?'报告与导出':'独立标注',{exact:true}).click();await idle(p);
 const input=p.getByRole('textbox',{name:admin?'管理访问码':'独立标注访问码',exact:true});
 await input.fill(token);await input.press('Tab');await idle(p);
}
async function noArbitration(p){
 for(const name of ['保存人工参考版本','确认此仲裁参考用于训练'])assert.equal(await p.getByRole('button',{name,exact:true}).count(),0);
 assert.equal(await p.getByRole('textbox',{name:'人工仲裁说明',exact:true}).count(),0);
}
async function frame(p){await p.locator('iframe[title*="box_editor"]').first().waitFor();const result=p.frames().find(f=>f.url().includes('box_editor'));assert(result);await result.locator('#count').waitFor();return result;}
async function download(p,button,name){
 const waiting=p.waitForEvent('download');await p.getByRole('button',{name:button,exact:true}).click();
 const result=await waiting;const failure=await result.failure();
 if(failure==='canceled' && name.endsWith('.zip') && process.env.DUAL_EXTERNAL_DOWNLOAD_MANAGER){
  // An installed download manager may take over binary browser downloads.
  // Record the limitation, then verify the exact generated attachment endpoint.
  const response=await p.context().request.get(result.url());assert.equal(response.status(),200);
  const bytes=await response.body();assert.deepEqual(Array.from(bytes.slice(0,4)),[80,75,3,4]);
  assert(response.headers()['content-disposition']?.includes('.zip'));
  await fs.writeFile(path.join(downloads,name),bytes);
  externalDownloads.push({file:name,native_browser_failure:failure,attachment_http_status:response.status(),
   transfer:'Exact generated attachment fetched via HTTP after UI click; not a successful native browser download',
   detected_download_manager:process.env.DUAL_EXTERNAL_DOWNLOAD_MANAGER});
 }else{assert.equal(failure,null);await result.saveAs(path.join(downloads,name));}
 await idle(p);
}
try{
 // Only isolated synthetic images receive new annotations in this script.
 for(const role of ['A','B']){
  const {context,p}=await page();await login(p,fixtureUrl,fixture.tokens[role]);
  for(let index=0;index<2;index++){
   await p.getByText(`本人已提交 ${index} / 2 幅`,{exact:true}).waitFor();
   const canvasFrame=await frame(p);assert.equal(await canvasFrame.locator('#count').textContent(),'0 个框');await noArbitration(p);
   await p.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle(p);
   if(index===0){
    const canvas=canvasFrame.locator('#canvas');await canvas.scrollIntoViewIfNeeded();const bounds=await canvas.boundingBox();
    const shift=role==='A'?0:.02;
    await p.mouse.move(bounds.x+bounds.width*(.2+shift),bounds.y+bounds.height*.2);await p.mouse.down();
    await p.mouse.move(bounds.x+bounds.width*.7,bounds.y+bounds.height*.75,{steps:8});await p.mouse.up();await idle(p);
   }else{await p.getByText('独立检查后确认整图无fire/smoke',{exact:true}).click();await idle(p);}
   const submit=p.getByRole('button',{name:'提交并锁定本图独立标注',exact:true});await submit.waitFor();assert(await submit.isEnabled());
   await submit.click();await idle(p);
  }
  await p.getByText('本人已提交 2 / 2 幅',{exact:true}).waitFor();
  await p.getByRole('button',{name:'退出本次独立标注',exact:true}).click();await idle(p);
  assert.equal(await p.getByRole('textbox',{name:'独立标注访问码',exact:true}).inputValue(),'');await context.close();
  if(role==='A'){
   const admin=await page();await login(admin.p,fixtureUrl,fixture.tokens.admin,true);
   await admin.p.getByText('等待两名标注者全部完成；此时不展示框或预标注。',{exact:true}).waitFor();
   assert.equal(await admin.p.getByRole('button',{name:'生成导出包',exact:true}).count(),0);
   assert.equal(await admin.p.getByRole('button',{name:'生成标注质量报告',exact:true}).count(),0);await admin.context.close();
  }
 }
 {
  const {context,p}=await page();await login(p,fixtureUrl,fixture.tokens.admin,true);await noArbitration(p);
  await p.getByRole('button',{name:'生成标注质量报告',exact:true}).click();await idle(p);
  await download(p,'下载标注质量报告','synthetic_quality.json');
  const report=JSON.parse(await fs.readFile(path.join(downloads,'synthetic_quality.json'),'utf8'));
  assert.equal(report.images,2);assert.equal(report.independent_submissions.length,4);
  assert.deepEqual(Object.keys(report.preannotation_quality),['A','B']);assert.equal(report.agreement_summary.both_empty_images,1);
  for(const [role,label] of [['both','A/B完整包（含质量报告）'],['A','仅A原始标注'],['B','仅B原始标注']]){
   const control=p.getByTestId('stSelectbox').filter({has:p.getByText('导出内容',{exact:true})}).getByRole('combobox');
   await control.click();await p.getByRole('option',{name:label,exact:true}).click();await idle(p);
   await p.getByRole('button',{name:'生成导出包',exact:true}).click();await idle(p);
   await download(p,'下载标注ZIP',`synthetic_${role}.zip`);
  }
  await p.getByText('查看原图与A/B标注',{exact:true}).click();await idle(p);
  await p.getByText('A独立结果（只读）',{exact:true}).waitFor();await p.getByText('B独立结果（只读）',{exact:true}).waitFor();
  assert.equal(await p.locator('iframe[title*="box_editor"]').count(),0);
  await p.screenshot({path:path.join(downloads,'report_and_export.png'),fullPage:true});await context.close();
 }
 if(formal){
  const {context,p}=await page();await login(p,formalUrl,formal.tokens.admin,true);await noArbitration(p);
  await p.getByRole('button',{name:'生成标注质量报告',exact:true}).click();await idle(p);
  await download(p,'下载标注质量报告','formal_quality.json');
  const report=JSON.parse(await fs.readFile(path.join(downloads,'formal_quality.json'),'utf8'));
  assert.equal(report.images,200);assert.equal(report.independent_submissions.length,400);
  assert.deepEqual(Object.keys(report.preannotation_quality),['A','B']);await context.close();
 }
 assert.deepEqual(errors,[]);
 const result={status:'passed',verified_at:new Date().toISOString(),formal_task_read_only:Boolean(formal),
  synthetic_two_role_two_image_submissions:true,blank_canvases_and_own_progress:true,logout_clears_credential:true,
  no_arbitration_controls:true,admin_locked_before_both_complete:true,submit_while_timer_active:true,
  separate_A_B_and_combined_export_files_verified:true,native_zip_downloads_verified:externalDownloads.length===0,
  external_download_interceptions:externalDownloads,quality_report_two_human_references:true,read_only_comparison:true,
  scope:'All annotation actions use an isolated synthetic task; formal task only read and exported, never modified',page_errors:errors};
 await fs.writeFile(path.join(root,'reports/dual_annotation_browser.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
}catch(error){
 let message=String(error?.stack||error);
 for(const token of [...Object.values(fixture.tokens),...Object.values(formal?.tokens||{})])message=message.replaceAll(token,'[REDACTED]');
 console.error(message);process.exitCode=1;
}finally{await browser.close();}
