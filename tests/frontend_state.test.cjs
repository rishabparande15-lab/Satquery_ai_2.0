// Pure-JavaScript UI state tests; these do not substitute for browser visual QA.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname,"../src/static/app.js"),"utf8");

function harness(fetcher) {
  const nodes = new Map();
  function element(id) {
    return {id,value:"",textContent:"",hidden:false,disabled:false,files:[],checked:false,children:[],handlers:{},
      append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items;this.textContent=""},removeAttribute(name){delete this[name]},
      addEventListener(name,fn){this.handlers[name]=fn},reset(){nodes.get("query").value="Default query"}};
  }
  const document = {getElementById(id){if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id)},createElement:()=>element("")};
  const context = vm.createContext({document,fetch:fetcher,AbortController,FormData,performance,TypeError,
    setTimeout:(fn,ms)=>setTimeout(fn,Math.min(ms,50)),clearTimeout,setInterval,clearInterval,console});
  vm.runInContext(source,context);
  for (const [id,value] of [["query","optical"],["source","local"],["sample","61_39"]])document.getElementById(id).value=value;
  return {nodes,document,submit:()=>document.getElementById("analysis-form").handlers.submit({preventDefault(){}})};
}
const ok=data=>Promise.resolve({ok:true,json:async()=>data});
const result = {analysis_id:"real-fixture-id",status:"completed",llm_explanation:"Feature extraction fixture",
  device:"cpu",runtime_seconds:0.123,features:{spectral:{dimension:52},deep:{pooled_dimension:768},hybrid:{}},
  interpretation:{status:"ANSWERED",answer:"The derived optical evidence shows moderate vegetation-related spectral evidence.",
    technical_answer:"Technical vegetation evidence.",provenance:{sample_id:"61_39"},claims:[{claim_id:"vegetation",sensor:"OPTICAL",strength:"MODERATE",source_features:["NDVI"],region_ids:["vegetation-region-000"],token_indices:[1,2]}]},
  validation:{errors:[]},warnings:[],evidence:[],execution_trace:[],confidence:{prediction_status:"unavailable",calibration_status:"uncalibrated",accuracy_status:"no accuracy claim"}};
function base(url){return url==="/api/samples"?ok({samples:["61_39"]}):ok({busy:false})}

test("duplicate submissions send one analysis; busy controls recover",async()=>{
  let count=0,complete;
  const h=harness((url)=>url==="/api/analyze"?(count++,new Promise(r=>{complete=r})):base(url));
  const first=h.submit();
  await new Promise(setImmediate);
  await h.submit();
  assert.equal(count,1);
  assert.equal(h.nodes.get("run").disabled,true);
  complete(await ok(result)); await first;
  assert.equal(h.nodes.get("run").disabled,false);
  assert.equal(h.nodes.get("download").hidden,false);
  assert.match(h.nodes.get("download").href,/api\/report/);
});
test("backend error shows authored message and clears prior result",async()=>{
  let fail=false;
  const h=harness(url=>url==="/api/analyze"?(fail?Promise.resolve({ok:false,json:async()=>({error:{message:"Invalid input"}})}):ok(result)):base(url));
  await h.submit(); fail=true; await h.submit();
  assert.equal(h.nodes.get("status").textContent,"Invalid input");
  assert.equal(h.nodes.get("results").hidden,true);
  assert.equal(h.nodes.get("download").hidden,true);
  assert.equal(h.nodes.get("run").disabled,false);
});
test("network failure and timeout recover controls",async()=>{
  for (const timeout of [false,true]) {
    const h=harness((url,opts)=>url!=="/api/analyze"?base(url):timeout?new Promise((_,reject)=>opts.signal.addEventListener("abort",()=>reject({name:"AbortError"}))):Promise.reject(new TypeError("network")));
    await h.submit();
    assert.match(h.nodes.get("status").textContent,timeout?/timed out/:/Cannot reach/);
    assert.equal(h.nodes.get("run").disabled,false);
    assert.equal(h.nodes.get("progress").hidden,true);
  }
});
test("reset clears prior report, downloads and upload controls",async()=>{
  const h=harness(url=>url==="/api/analyze"?ok(result):base(url));
  await h.submit(); h.nodes.get("reset").handlers.click();
  assert.equal(h.nodes.get("results").hidden,true);
  assert.equal(h.nodes.get("download").hidden,true);
  assert.equal(h.nodes.get("upload-fields").hidden,true);
  assert.equal(h.nodes.get("status").textContent,"Ready for a new analysis.");
});
test("empty query and missing upload do not submit analysis",async()=>{
  let count=0;
  const h=harness(url=>{if(url==="/api/analyze")count++;return base(url)});
  h.nodes.get("query").value=" ";await h.submit();
  assert.equal(h.nodes.get("status").textContent,"Enter a research query.");
  h.nodes.get("query").value="optical";h.nodes.get("source").value="upload";await h.submit();
  assert.equal(h.nodes.get("status").textContent,"Select the required GeoTIFF imagery.");
  assert.equal(count,0);
});
test("server busy state prevents new upload or inference",async()=>{
  let requests=0;
  const h=harness(url=>url==="/api/samples"?base(url):url==="/api/health"?ok({busy:true}):(requests++,ok({})));
  await h.submit();
  assert.equal(requests,0);
  assert.match(h.nodes.get("status").textContent,/already running/);
});
test("constrained interpretation renders summary, evidence, sensor, region and provenance",async()=>{
  const h=harness(url=>url==="/api/analyze"?ok(result):base(url));
  await h.submit();
  assert.equal(h.nodes.get("answer").textContent,result.interpretation.answer);
  assert.match(h.nodes.get("interpretation-evidence").textContent,/NDVI/);
  assert.match(h.nodes.get("interpretation-sensors").textContent,/OPTICAL/);
  assert.equal(h.nodes.get("interpretation-regions").children.length,1);
  assert.match(h.nodes.get("interpretation-provenance").textContent,/61_39/);
});
