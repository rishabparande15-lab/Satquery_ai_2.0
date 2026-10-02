"""Real headless-Chrome browser verification for the Phase 3AD SAR panel."""
from __future__ import annotations
import json, os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'artifacts/phase3ad_validation_s1.tif'
OUT=ROOT/'artifacts/phase3ad_browser'
BASE=os.environ.get('SATQUERY_BASE_URL','http://127.0.0.1:8013')
CHROME=os.environ.get('SATQUERY_BROWSER_EXECUTABLE',r'C:\Program Files\Google\Chrome\Application\chrome.exe')

def main():
    OUT.mkdir(exist_ok=True)
    console=[]; page_errors=[]; failed=[]; responses=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=CHROME,headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1200})
        page.on('console',lambda m:console.append(m.text) if m.type=='error' else None)
        page.on('pageerror',lambda e:page_errors.append(str(e)))
        page.on('requestfailed',lambda r:failed.append({'url':r.url,'failure':str(r.failure)}))
        page.on('response',lambda r:responses.append({'status':r.status,'url':r.url,'body':r.json()}) if '/api/v1/query' in r.url else None)
        page.goto(BASE,wait_until='networkidle')
        page.locator('#sar-vqa-upload').set_input_files(str(FIXTURE))
        page.locator('#sar-vqa-question').fill('Do parts of the image correspond to pastures?')
        with page.expect_response(lambda r:'/api/v1/query' in r.url and r.request.method=='POST') as event: page.locator('#run-sar-vqa').click()
        response=event.value; body=response.json()
        # The API response can arrive before the UI continuation has rendered
        # its receipt.  Wait for that real UI state rather than treating the
        # button's initial enabled state as completion.
        page.wait_for_function("document.getElementById('sar-vqa-provenance').textContent.includes('NOT_AVAILABLE')")
        output=page.locator('#sar-vqa-output').inner_text()
        # The receipt is intentionally inside a collapsed <details>; read its
        # DOM text, not only text currently visible to accessibility layout.
        provenance=page.locator('#sar-vqa-provenance').text_content() or ''
        (OUT/'debug.json').write_text(json.dumps({'status':response.status,'body':body,'output':output,'provenance':provenance},indent=2),encoding='utf-8')
        assert response.status==200 and body['route']=='SINGLE_IMAGE_SAR_VQA' and body['answer']
        assert 'SAR_ONLY_SEMANTIC_VALIDATION_LIMITED' in output and 'NOT_AVAILABLE' in provenance
        assert not any(x in str(body) for x in ('E:\\','D:\\','C:\\','models--','snapshots/'))
        with page.expect_download() as download: page.locator('#sar-vqa-export').click()
        saved=OUT/'sar-vqa-export.json'; download.value.save_as(str(saved))
        exported=json.loads(saved.read_text(encoding='utf-8'))
        assert exported['route']=='SINGLE_IMAGE_SAR_VQA' and exported['answer']==body['answer']
        assert not any(x in saved.read_text(encoding='utf-8') for x in ('E:\\','D:\\','C:\\','models--','snapshots/'))
        page.screenshot(path=str(OUT/'sar-vqa-result.png'),full_page=True); browser.close()
    receipt={'http_status':response.status,'route':body['route'],'answer':body['answer'],'output':output,'provenance':json.loads(provenance),'responses':responses,'console_errors':console,'page_errors':page_errors,'failed_requests':failed,'download':saved.name,'test_access':0}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'http':response.status,'route':body['route'],'answer':body['answer'],'console':len(console),'page':len(page_errors),'failed':len(failed)}))
if __name__=='__main__': main()
