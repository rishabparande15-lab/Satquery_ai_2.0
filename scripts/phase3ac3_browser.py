"""Fresh single-session Chrome lifecycle regression for the frozen four routes."""
from __future__ import annotations
import json, os, tempfile, sys
from pathlib import Path
from urllib.request import urlopen
import rasterio
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.dataset_loader import OPTICAL_BANDS, discover_s2_samples
from src.config import get_settings
BASE=os.environ.get('SATQUERY_BASE_URL','http://127.0.0.1:8805'); OUT=ROOT/'artifacts'/'final'/'runtime'/'phase3ac3'; DATA=get_settings().dataset_root; LEVIR=ROOT/'datasets'/'levir_cc'/'development_smoke'/'images'/'val'; CHROME=os.environ.get('SATQUERY_BROWSER_EXECUTABLE',r'C:\Program Files\Google\Chrome\Application\chrome.exe')
S2='S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11'; PAIR='S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22'
def api(path):
    with urlopen(BASE+path,timeout=30) as r:return json.loads(r.read())
def stack(path):
    sample=next(x for x in discover_s2_samples(DATA,allowed_splits=('train',)) if x.patch_id==S2)
    with rasterio.open(sample.optical_paths['B02']) as ref:
        profile=ref.profile.copy();profile.update(count=12)
        with rasterio.open(path,'w',**profile) as out:
            for i,b in enumerate(OPTICAL_BANDS,1):
                with rasterio.open(sample.optical_paths[b]) as src:out.write(src.read(1),i);out.set_band_description(i,b)
def record(page,label,button,output,export,expected):
    before=api('/api/v1/health')['runtime'];
    with page.expect_response(lambda r:'/api/v1/query' in r.url and r.request.method=='POST',timeout=180000) as event: page.locator(button).click()
    response=event.value; body=response.json();page.wait_for_function(f"!document.querySelector('{button}').disabled")
    assert response.status==200 and body['route']==expected and body['status']=='COMPLETED',body
    assert page.locator(output).inner_text(); assert page.locator(export).is_visible();
    with page.expect_download() as d: page.locator(export).click()
    download=d.value; save=OUT/'json_exports'/f'{label}.json';save.parent.mkdir(parents=True,exist_ok=True);download.save_as(str(save)); exported=json.loads(save.read_text())
    assert exported['route']==expected and 'evidence' in exported and 'provenance' in exported
    after=api('/api/v1/health')['runtime']; memory=api('/api/v1/runtime/memory')
    result={'http_status':response.status,'route':body['route'],'answer':body.get('answer'),'warnings':body.get('warnings'),'confidence':body.get('confidence'),'evidence':bool(body.get('evidence')),'provenance':bool(body.get('provenance')),'trace':bool(body.get('execution_trace')),'export':save.name,'loading_cleared':not page.locator(button).is_disabled(),'resident_before':before['resident_specialist'],'resident_after':after['resident_specialist'],'gpu_allocated':memory['allocated'],'gpu_reserved':memory['reserved']}
    (OUT/'browser_sequence'/label).mkdir(parents=True,exist_ok=True);(OUT/'browser_sequence'/label/'receipt.json').write_text(json.dumps(result,indent=2));page.screenshot(path=str(OUT/'browser_sequence'/label/'result.png'),full_page=True);return result
def main():
    OUT.mkdir(parents=True,exist_ok=True);console=[];pages=[];failed=[]
    with tempfile.TemporaryDirectory() as t:
      s2=Path(t)/'approved-s2.tif';stack(s2)
      with sync_playwright() as p:
        b=p.chromium.launch(executable_path=CHROME,headless=True);page=b.new_page(viewport={'width':1440,'height':1200});page.on('console',lambda m:console.append(m.text) if m.type=='error' else None);page.on('pageerror',lambda e:pages.append(str(e)));page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':str(r.failure)}));page.goto(BASE,wait_until='networkidle')
        results={}
        page.get_by_role('button',name='Single image').click();page.locator('#single-image-upload').set_input_files(str(s2));page.locator('#single-image-sensor-declared').check();page.locator('#single-image-band-order').check();page.locator('#single-image-task').select_option('SINGLE_IMAGE_VQA');page.locator('#single-image-question').fill('Is water visible in this image?');results['01_vqa']=record(page,'01_vqa','#run-single-image','#single-image-output','#single-image-export','SINGLE_IMAGE_VQA')
        page.locator('#single-image-task').select_option('SINGLE_IMAGE_SCENE_DESCRIPTION');page.locator('#single-image-upload').set_input_files([]);page.locator('#scene-image-upload').set_input_files(str(LEVIR/'A'/'val_000001.png'));page.locator('#scene-upload-sensor').select_option('generic-rgb');page.locator('#single-image-question').fill('Describe this image.');results['02_scene']=record(page,'02_scene','#run-single-image','#single-image-output','#single-image-export','SINGLE_IMAGE_SCENE_DESCRIPTION')
        page.get_by_role('button',name='Optical + SAR',exact=True).click();page.locator('#v1-patch-id').evaluate("(s,v)=>{if(![...s.options].some(o=>o.value===v))s.add(new Option(v,v));s.value=v}",PAIR);page.locator('#v1-question').fill('Use the SAR and optical information together. Is water visible?');results['03_optical_sar']=record(page,'03_optical_sar','#run-v1','#v1-output','#v1-export','OPTICAL_SAR_ANALYSIS')
        page.get_by_role('button',name='Bi-temporal').click();page.locator('#temporal-t1').set_input_files(str(LEVIR/'A'/'val_000001.png'));page.locator('#temporal-t2').set_input_files(str(LEVIR/'B'/'val_000001.png'));page.locator('#temporal-question').fill('What changed between these two images?');results['04_temporal']=record(page,'04_temporal','#run-temporal-change','#temporal-change-output','#temporal-change-export','TEMPORAL_CHANGE_DESCRIPTION')
        page.get_by_role('button',name='Single image').click();page.locator('#single-image-task').select_option('SINGLE_IMAGE_VQA');page.locator('#single-image-upload').set_input_files(str(s2));page.locator('#single-image-sensor-declared').check();page.locator('#single-image-band-order').check();page.locator('#single-image-question').fill('Is water visible in this image?');results['05_vqa_revisit']=record(page,'05_vqa_revisit','#run-single-image','#single-image-output','#single-image-export','SINGLE_IMAGE_VQA');b.close()
    final={'results':results,'console_errors':console,'page_errors':pages,'failed_requests':failed,'health':api('/api/v1/health'),'ready':api('/api/v1/ready'),'test_access':0};(OUT/'browser_results.json').write_text(json.dumps(final,indent=2));print(json.dumps({k:v['answer'] for k,v in results.items()},indent=2))
if __name__=='__main__':main()
