import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));const fixture=JSON.parse(await fs.readFile(path.join(root,'artifacts/completion_checks/quality_access.json'),'utf8'));
const {chromium}=await import('file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({headless:true,channel:'msedge',args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const context=await browser.newContext({viewport:{width:1440,height:1200}});const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
const base=process.env.ALARM_APP_URL||'http://127.0.0.1:8502';
async function idle(){await page.waitForTimeout(300);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:60000});await page.waitForTimeout(300);assert.equal(await page.getByTestId('stException').count(),0);}
try{
 for(const role of ['A','B']){
  await page.goto(base+'/Quality');await page.getByRole('heading',{name:'标注质量与面积误差',exact:true}).waitFor();
  await page.getByText('独立标注',{exact:true}).click();await idle();
  await page.getByRole('textbox',{name:'独立标注访问码',exact:true}).fill(fixture.tokens[role]);await page.getByRole('textbox',{name:'独立标注访问码',exact:true}).press('Tab');await idle();
  assert.equal(await page.getByText('人工仲裁参考框JSON',{exact:true}).count(),0);
  await page.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle();
  const frame=page.frames().find(f=>f.url().includes('box_editor'));
  assert(frame);const canvas=frame.locator('#canvas');await canvas.scrollIntoViewIfNeeded();const bounds=await canvas.boundingBox();
  await page.mouse.move(bounds.x+bounds.width*.2,bounds.y+bounds.height*.2);await page.mouse.down();await page.mouse.move(bounds.x+bounds.width*.6,bounds.y+bounds.height*.6,{steps:8});await page.mouse.up();await idle();
  await page.getByRole('button',{name:'暂停工作计时',exact:true}).click();await idle();
  await page.getByRole('button',{name:'提交并锁定本图独立标注',exact:true}).click();await page.getByText('此图已独立提交并锁定。',{exact:true}).waitFor();
 }
 await page.goto(base+'/Quality');await page.getByText('管理与仲裁',{exact:true}).click();await idle();
 await page.getByRole('textbox',{name:'管理访问码',exact:true}).fill(fixture.tokens.admin);await page.getByRole('textbox',{name:'管理访问码',exact:true}).press('Tab');await idle();
 await page.getByRole('textbox',{name:'人工仲裁说明',exact:true}).fill('Automated synthetic UI workflow; not actual independent human annotations.');await page.getByRole('textbox',{name:'人工仲裁说明',exact:true}).press('Tab');await idle();
 await page.getByRole('button',{name:'保存人工参考版本',exact:true}).click();await idle();
 await page.getByRole('button',{name:'生成标注质量报告',exact:true}).click();await page.getByRole('button',{name:'下载标注质量报告',exact:true}).waitFor();
 await page.getByText('面积误差',{exact:true}).click();await idle();
 const imageControl=page.getByTestId('stSelectbox').filter({has:page.getByText('掩码对应图片',{exact:true})}).getByRole('combobox');
 await imageControl.click();await page.getByRole('option').filter({hasText:fixture.asset_id}).click();await idle();
 await page.getByTestId('stFileUploader').filter({has:page.getByText('独立人工参考掩码PNG',{exact:true})}).locator('input[type=file]').setInputFiles(path.join(root,'artifacts/completion_checks/reference.png'));await idle();
 await page.getByTestId('stFileUploader').filter({has:page.getByText('模型预测掩码PNG',{exact:true})}).locator('input[type=file]').setInputFiles(path.join(root,'artifacts/completion_checks/prediction.png'));await idle();
 for(const [name,value] of [['面积评测执行者','Automated synthetic test'],['参考来源与绘制方法','Synthetic known rectangles; functional math only']]){await page.getByRole('textbox',{name,exact:true}).fill(value);await page.getByRole('textbox',{name,exact:true}).press('Tab');await idle();}
 await page.getByTestId('stCheckbox').locator('label').click();await idle();
 await page.getByRole('button',{name:'计算并保存面积误差',exact:true}).click();await page.getByRole('button',{name:'下载面积误差记录',exact:true}).waitFor();
 await page.getByText('工时对照',{exact:true}).click();await idle();
 for(const [name,value] of [['工时对照组编号',fixture.task],['计时人员','Browser simulated actor'],['计时方法与任务说明','Automated fixture, not actual saved labor']])await page.getByRole('textbox',{name,exact:true}).fill(value);
 await page.getByRole('button',{name:'保存实测工时',exact:true}).click();await page.getByText('实测记录已保存。',{exact:true}).waitFor();
 assert.deepEqual(errors,[]);
 const report={status:'passed',scope:'Two automated simulated browser roles on one existing synthetic image; not human independence evidence',verified_at:new Date().toISOString(),separate_blank_canvases:true,activity_timing_pause_submit:true,adjudication_and_report:true,mask_upload_and_error_record:true,worktime_form_submission:true,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/quality_browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{await browser.close();}
