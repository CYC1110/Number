/* Offline execution of both delivered HTML files with a real Canvas renderer.
 * npm install --prefix /tmp/iclr-viewer-qa @napi-rs/canvas
 * NODE_PATH=/tmp/iclr-viewer-qa/node_modules node tests/viewer_offline.cjs
 * No browser UI is emulated or claimed verified by this test.
 */
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {createCanvas,GlobalFonts}=require('@napi-rs/canvas');
for(const name of ['DejaVuSans.ttf','DejaVuSans-Bold.ttf'])GlobalFonts.registerFromPath('/usr/share/fonts/truetype/dejavu/'+name,'system-ui');
const root=path.resolve(__dirname,'..'),out=path.join(root,'results/viewers/qa');fs.mkdirSync(out,{recursive:true});
async function boot(mode){
 const html=fs.readFileSync(path.join(root,`viewers/${mode}.html`),'utf8'),elements=new Map(),all=[];const downloads=[],blobs=new Map();let next=0;
 function el(tag='div'){const x={tag,value:'',checked:false,disabled:false,style:{},children:[],min:'',max:'',textContent:'',append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;if(tag==='select')this.value=items[0]?.value||'';},remove(){},setAttribute(){},addEventListener(){},click(){if(tag==='a')downloads.push({name:this.download,blob:blobs.get(this.href)});}};all.push(x);return x;}
 const canvas=()=>{const c=createCanvas(500,520);c.getBoundingClientRect=()=>({width:500,height:520});c.setPointerCapture=()=>{};c.addEventListener=()=>{};c.setAttribute=()=>{};c.toBlob=cb=>cb(new Blob([c.toBuffer('image/png')],{type:'image/png'}));return c;};
 for(const m of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){const x=el(m[1]);for(const k of ['value','min','max']){const a=m[2].match(new RegExp(`${k}="([^"]*)"`));if(a)x[k]=a[1];}x.checked=/\bchecked\b/.test(m[2]);elements.set(m[3],x);}
 Object.assign(elements.get('columns'),{value:'auto'});Object.assign(elements.get('mode'),{value:'both'});Object.assign(elements.get('scale'),{value:'normalized'});Object.assign(elements.get('p-bins'),{value:'6'});
 const urls={createObjectURL(b){const u='blob:'+next++;blobs.set(u,b);return u;},revokeObjectURL(u){blobs.delete(u);}};
 class Worker{constructor(url){this.active=true;this.ready=blobs.get(url).text().then(code=>{const self={postMessage:data=>setImmediate(()=>this.active&&this.onmessage?.({data}))};this.ctx=vm.createContext({self,Uint8Array,Uint32Array,Map,Math,Array,Error,String});vm.runInContext(code,this.ctx);});}postMessage(data){this.ready.then(()=>setImmediate(()=>{if(this.active)this.ctx.self.onmessage({data});})).catch(e=>this.onerror?.({message:String(e)}));}terminate(){this.active=false;}}
 const document={getElementById:id=>{assert(elements.has(id),id);return elements.get(id);},createElement:tag=>tag==='canvas'?canvas():el(tag),querySelectorAll:()=>all.filter(x=>['button','select','input'].includes(x.tag))};const window={devicePixelRatio:1,addEventListener(){}};
 const ctx=vm.createContext({document,window,Worker,URL:urls,Blob,console,atob:s=>Buffer.from(s,'base64').toString('binary'),location:{protocol:'file:',hostname:''},setTimeout:(f,ms)=>{const t=setTimeout(f,ms);t.unref();return t;},requestAnimationFrame:fn=>{if(fn.name!=='tick')setImmediate(()=>fn(performance.now()));return 1;}});
 const source=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).join('\n');vm.runInContext(source,ctx,{filename:mode+'.html'});assert(window.viewerReady);const api=window.viewerAPI;
 // All bundled states pass strict validation; origins and common range stay fixed.
 api.render();fs.writeFileSync(path.join(out,`${mode}_default.png`),api.panels[mode==='compare'?1:0].canvas.toBuffer('image/png'));
 const before=api.extent;for(const g of api.groups){const s=api.stateAt(g);assert(Number.isFinite(s.theta));assert(s.means.every(m=>m.every(Number.isFinite)));}assert.equal(api.extent,before);
 for(const [id,value]of [['az-number','-35'],['el-number','30'],['roll-number','15'],['size-number','8'],['alpha-number','.7']]){elements.get(id).value=value;elements.get(id).oninput();}
 assert.equal(api.camera.az,-35);assert.equal(api.camera.roll,15);assert.equal(elements.get('az').value,-35);
 api.panels[0].canvas.onpointerdown({clientX:0,clientY:0,pointerId:1});api.panels[0].canvas.onpointermove({clientX:10,clientY:10});api.panels[0].canvas.onpointerup();assert.equal(api.camera.az,-32);assert.equal(Number(elements.get('az-number').value),-32);
 api.camera.az=-35;api.camera.el=30;api.camera.roll=15;
 if(mode==='epochs'){elements.get('interval').value='7';elements.get('interval').onchange();assert.deepEqual(JSON.parse(JSON.stringify(api.selectedFrames(api.groups[0]).map(s=>s.epoch))),[0,1,7,14,21,28,35,42,49,56,63,70,77,84,90]);api.setEpoch(15.5);api.render();assert(api.panels[0].scene.interpolated);assert.equal(api.extent,before);}
 api.render();for(let i=0;i<api.panels.length;i++)fs.writeFileSync(path.join(out,`${mode}_panel_${i}.png`),api.panels[i].canvas.toBuffer('image/png'));
 elements.get('mode').value='points';elements.get('mode').onchange();api.render();const svg=api.exportSVG();assert.equal((svg.match(/<image /g)||[]).length,api.panels.length);assert(!svg.includes('<circle'));fs.writeFileSync(path.join(out,`${mode}_points.svg`),svg);fs.writeFileSync(path.join(out,`${mode}_points.png`),api.panels[0].canvas.toBuffer('image/png'));
 const cam=JSON.stringify(api.camera);elements.get('svg').onclick();assert.equal(JSON.stringify(api.camera),cam);assert(all.filter(x=>['button','select','input'].includes(x.tag)).every(x=>!x.disabled));api.setEpoch(1);api.render();
 // A valid user bundle adds a new panel; malformed files do not remove existing data.
 const upload=JSON.parse(JSON.stringify({schema:'np-representation/v1',states:[api.states[0]]}));upload.states[0].id='uploaded/run_01/90';upload.states[0].network='uploaded';upload.states[0].run='run_01';upload.states[0].label='<test & network>';const count=api.states.length;
 await api.importFiles([{name:'user.json',size:1,text:async()=>JSON.stringify(upload)}]);assert.equal(api.states.length,count+1);assert(api.panels.some(p=>api.stateAt(api.groups.find(g=>g.key===p.key)).label==='<test & network>'));assert(api.exportSVG().includes('&lt;test &amp; network&gt;'));
 const corrupt=structuredClone(upload);corrupt.states[0].theta=NaN;assert.throws(()=>api.validateBundle(corrupt));
 await api.importFiles([{name:'bad.json',size:1,text:async()=>'{broken'}]);assert.equal(api.states.length,count+1);assert(!elements.get('upload').disabled);
 // Remove the added panel; exercise encoding and restoration on one measured sequence.
 api.panels.at(-1).card.children[0].children[1].onclick();elements.get('mode').value='both';elements.get('mode').onchange();
 if(mode==='epochs'){
  elements.get('duration').value='1';api.setEpoch(17);api.render();const camera=JSON.stringify(api.camera);await api.saveGIF();assert(window.gifExportReport, elements.get('export-status').textContent);assert.equal(window.gifExportReport.frames,50);assert.equal(window.gifExportReport.frameDelayMilliseconds,20);assert.equal(JSON.stringify(api.camera),camera);assert(all.filter(x=>['button','select','input'].includes(x.tag)).every(x=>!x.disabled));assert.equal(window.viewerState.epoch,17);
  const gif=downloads.find(x=>x.name==='np-epochs.gif');fs.writeFileSync(path.join(out,'current_view.gif'),Buffer.from(await gif.blob.arrayBuffer()));
  api.setEpoch(0);api.render();const scene=api.sceneFor(api.stateAt(api.groups[0]),600,520);const saved=api.panels[0].canvas;vm.runInContext('drawScene',ctx)(saved,scene);fs.writeFileSync(path.join(out,'gif_expected_first.png'),saved.toBuffer('image/png'));
 }
 return {mode,initialStates:count,uploadPassed:true,svgRasterClouds:true,cameraSynchronized:true,gif:window.gifExportReport||null,browserUIVerified:false};
}
(async()=>{const report=[];for(const mode of ['compare','epochs'])report.push(await boot(mode));fs.writeFileSync(path.join(out,'offline_report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));})().catch(e=>{console.error(e);process.exit(1);});
