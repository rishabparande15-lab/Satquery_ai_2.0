"""Bounded API memory smoke for the four approved non-test SatQuery routes."""
from __future__ import annotations
import json, os, time, uuid
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[1]; BASE=os.environ.get("SATQUERY_BASE_URL","http://127.0.0.1:8798")
S2="S2B_MSIL2A_20170831T095029_N9999_R079_T33UXP_05_11"; PAIR="S2A_MSIL2A_20170717T113321_N9999_R080_T29UPV_35_22"
LEVIR=ROOT/'datasets'/'levir_cc'/'development_smoke'/'images'/'val'

def get(path):
    with urlopen(BASE+path,timeout=30) as r:return json.loads(r.read())
def post(path,payload):
    req=Request(BASE+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    try:
        with urlopen(req,timeout=180) as r:return json.loads(r.read())
    except HTTPError as error:
        return json.loads(error.read())
def upload(t1,t2):
    b='----satquery'+uuid.uuid4().hex; body=b''
    for role,path in [('t1',t1),('t2',t2)]:
        body += f'--{b}\r\nContent-Disposition: form-data; name="{role}"; filename="{path.name}"\r\nContent-Type: image/png\r\n\r\n'.encode()+path.read_bytes()+b'\r\n'
    body += f'--{b}--\r\n'.encode(); req=Request(BASE+'/api/upload',data=body,headers={'Content-Type':f'multipart/form-data; boundary={b}'},method='POST')
    with urlopen(req,timeout=30) as r:return json.loads(r.read())['files']
def run(name, payload_factory):
    rows=[]
    for _ in range(3):
        before=get('/api/v1/runtime/memory'); started=time.perf_counter(); result=payload_factory(); elapsed=time.perf_counter()-started; after=get('/api/v1/runtime/memory')
        rows.append({'status':result.get('status'),'route':result.get('route'),'answer':result.get('answer'),'elapsed_seconds':elapsed,'before':before,'after':after})
    values=[r['after']['allocated'] for r in rows]
    classification='STABLE' if values[1:]==values[:-1] else ('CACHE_GROWTH_THEN_STABLE' if values[2]==values[1] else 'INCONCLUSIVE')
    return {'route':name,'requests':rows,'classification':classification}
def main():
    base=get('/api/v1/runtime/memory')
    payloads={
      'vqa':lambda:post('/api/v1/query',{'requested_task':'SINGLE_IMAGE_VQA','query':'Is water visible in this image?','patch_id':S2,'inputs':[{'role':'SINGLE','modality':'s2','input_id':S2,'source':'approved_local_patch'}]}),
      'scene':lambda:post('/api/v1/query',{'requested_task':'SINGLE_IMAGE_SCENE_DESCRIPTION','query':'Describe this image.','patch_id':S2,'inputs':[{'role':'SINGLE','modality':'rgb','input_id':S2,'source':'approved_local_patch'}]}),
      'optical_sar':lambda:post('/api/v1/query',{'query':'Use the SAR and optical information together. Is water visible?','patch_id':PAIR,'inputs':[{'role':'S1','modality':'s1','input_id':PAIR+':s1','source':'approved_local_patch'},{'role':'S2','modality':'s2','input_id':PAIR+':s2','source':'approved_local_patch'}]}),
      'temporal':lambda: (lambda f:post('/api/v1/query',{'t1_token':f['t1'],'t2_token':f['t2'],'query':'What changed between these two images?','pair_id':'val_000001.png','temporal_order':'PRE_POST','split':'validation','inputs':[{'role':'T1','modality':'optical','input_id':'val_000001.png','temporal_role':'PRE'},{'role':'T2','modality':'optical','input_id':'val_000001.png','temporal_role':'POST'}]}))(upload(LEVIR/'A'/'val_000001.png',LEVIR/'B'/'val_000001.png'))}
    selected=os.environ.get('SATQUERY_AC1_ROUTE')
    active={name:fn for name,fn in payloads.items() if not selected or name==selected}
    result={'baseline':base, **{name:run(name,fn) for name,fn in active.items()}, 'health_after':get('/api/v1/health'), 'ready_after':get('/api/v1/ready')}
    out=ROOT/'artifacts'/'final'/'runtime'/'phase3ac1';out.mkdir(parents=True,exist_ok=True); name=f'phase3ac1_raw_memory_{selected or "all"}.json';(out/name).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:(v.get('classification') if isinstance(v,dict) else None) for k,v in result.items() if k in payloads},indent=2))
if __name__=='__main__':main()
