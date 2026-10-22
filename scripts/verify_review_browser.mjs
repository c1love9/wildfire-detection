import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const fixture=JSON.parse(await fs.readFile(path.join(root,'artifacts/code_review/fixture.json'),'utf8'));
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
async function idle(){await page.waitForTimeout(350);await page.getByTestId('stStatusWidget').waitFor({state:'hidden',timeout:120000});await page.waitForTimeout(350);assert.equal(await page.getByTestId('stException').count(),0);}
const base=process.env.PRODUCTION_APP_URL||'http://127.0.0.1:8501';
const isolated=process.env.ALARM_APP_URL||'http://127.0.0.1:8502';
try{
 await page.goto(base);await page.getByText('Upload from Computer',{exact:true}).click();await idle();
 await page.locator('input[type=file]').setInputFiles(path.join(root,'artifacts/code_review/phone.jpg'));await idle();
 const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'Download Prediction',exact:true}).click()]);
 const png=await fs.readFile(await download.path());
 assert.equal(png.readUInt32BE(16),fixture.phone_width);assert.equal(png.readUInt32BE(20),fixture.phone_height);
 await page.locator('input[type=file]').setInputFiles({name:'invalid.jpg',mimeType:'image/jpeg',buffer:Buffer.from('invalid image')});await idle();
 await page.getByText(/图片内容无法解码/).waitFor();
 await page.goto(isolated+'/Quality');await page.getByText('独立标注',{exact:true}).click();await idle();
 const token=page.getByRole('textbox',{name:'独立标注访问码',exact:true});await token.fill(fixture.token);await token.press('Tab');await idle();
 const image=()=>page.getByTestId('stSelectbox').filter({has:page.getByText('本人的待标注图片',{exact:true})}).getByRole('combobox');
 await page.getByRole('button',{name:'开始或继续工作计时',exact:true}).click();await idle();await page.waitForTimeout(1200);
 await image().click();await page.getByRole('option').nth(1).click();await idle();await page.waitForTimeout(1500);
 await image().click();await page.getByRole('option').nth(0).click();await idle();
 const negative=page.getByTestId('stCheckbox').filter({has:page.getByText('独立检查后确认整图无fire/smoke',{exact:true})});
 await negative.locator('label').click();await idle();
 assert(await page.getByRole('button',{name:'提交并锁定本图独立标注',exact:true}).isEnabled(),'Switching images must leave the previous timer paused');
 const seconds=async()=>Number((await page.getByText(/已记录工作计时/).innerText()).match(/计时 ([\d.]+) 秒/)[1]);
 const before=await seconds();await page.waitForTimeout(1800);await negative.locator('label').click();await idle();
 const after=await seconds();assert.equal(after,before,'Time spent on a different image must not continue charging the previous image');
 assert.deepEqual(errors,[]);
 const result={status:'passed',verified_at:new Date().toISOString(),phone_orientation_and_prediction_png:true,corrupt_upload_handled:true,image_switch_pauses_previous_timer:true,paused_timer_seconds:after,official_data_modified:false,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/code_review_browser.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
}finally{await browser.close();}
