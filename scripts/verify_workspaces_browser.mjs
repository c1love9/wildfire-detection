import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('..',import.meta.url));
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'file:///C:/Users/HP/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
const browser=await chromium.launch({headless:true,channel:'msedge',args:['--enable-unsafe-swiftshader','--use-angle=swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
const base=process.env.ALARM_APP_URL||'http://127.0.0.1:8502';const stage=process.argv[2]||'comparison';
try{
 await page.goto(base+'/Comparison');await page.getByRole('heading',{name:'YOLO与开放域模型对比',exact:true}).waitFor({timeout:60000});
 await page.getByRole('button',{name:'运行并排检测',exact:true}).click();
 await page.getByText('并排推理完成；没有独立参考标注，不计算准确率。',{exact:true}).waitFor({timeout:120000});
 for(const img of await page.getByTestId('stImage').locator('img').all())await img.evaluate(e=>e.decode());
 await page.getByText('统一test评测',{exact:true}).click();
 await page.getByText(/待评测：请先选择本人成功训练的模型/).waitFor();
 assert(await page.getByRole('button',{name:'运行统一测试评测',exact:true}).isDisabled());
 if(stage==='experiments'){
   await page.goto(base+'/Experiments');await page.getByRole('heading',{name:'优化对照实验',exact:true}).waitFor();
   await page.getByText('待实验：尚无已复核训练数据。',{exact:true}).waitFor();
   assert(await page.getByRole('button',{name:'保存成对实验配置',exact:true}).isDisabled());
   await page.getByRole('button',{name:'运行样例前10秒策略对照',exact:true}).click();
   await page.getByRole('button',{name:'下载报警对照记录',exact:true}).waitFor({timeout:120000});
 }
 assert.equal(await page.getByTestId('stException').count(),0);
 assert.deepEqual(errors,[]);
 const report={status:'passed',stage,verified_at:new Date().toISOString(),real_side_by_side_inference:true,formal_evaluation_disabled_without_trained_model:true,page_errors:errors};
 await fs.writeFile(path.join(root,'reports/'+stage+'_browser.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
}finally{await browser.close();}
