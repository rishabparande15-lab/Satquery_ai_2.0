"""Preregister and construct the immutable Phase 3O.12 dataset/initial state."""
from __future__ import annotations
import hashlib, json, random, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.s2_asset_gate import sha256_file
OUT=ROOT/'artifacts/training/phase3o/phase3o12_run'; A=ROOT/'artifacts/annotations/image_language'
SEED=3012; REV='66285546d2b821cf421d4f5eb2576359d3770cd3'
def pick(rows,n,key): return sorted(rows,key=lambda x:hashlib.sha256(f'{SEED}|{key}|{x["record_id"]}'.encode()).hexdigest())[:n]
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 config={'experiment_name':'phase3o12_fresh_task_balanced_s2_projector','qwen_revision':REV,'projector_architecture':'S2MultispectralProjector(patch_hidden_size=128)','projector_parameter_count':5549184,'random_seed':SEED,'training_seed':SEED,'dataset_manifest_sources':{'visual_vqa_candidates.jsonl':sha256_file(A/'visual_vqa_candidates.jsonl'),'caption_candidates.jsonl':sha256_file(A/'caption_candidates.jsonl'),'manifest.json':sha256_file(A/'manifest.json')},'allowed_tasks':['binary_qa','multiple_choice_qa','caption'],'excluded_tasks':['test','grounding','sar','temporal','optical_sar','hybrid_830d'],'train_count':600,'validation_count':150,'test_count':0,'task_distribution':{'train':{'binary_qa':200,'multiple_choice_qa':200,'caption':200},'validation':{'binary_qa':50,'multiple_choice_qa':50,'caption':50}},'optimizer':'AdamW','learning_rate':0.001,'weight_decay':0.0001,'epochs':1,'batch_size':1,'gradient_accumulation':8,'precision':'float16','max_sequence_length':32,'generation_interface_version':'phase3o10_multimodal_prefix_v1','hardware':'RTX 5060 8GB','expected_output_paths':['phase3o12_projector_initial.pt','phase3o12_projector_final.pt','training_summary.json','untrained_semantic_audit.json','trained_semantic_audit.json','image_shuffle_map.json','reload_receipt.json']}
 (OUT/'preregistered_config.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
 vqa=[json.loads(x) for x in (A/'visual_vqa_candidates.jsonl').open()]; cap=[json.loads(x) for x in (A/'caption_candidates.jsonl').open()]
 def filt(xs,split,task): return [x for x in xs if x.get('split')==split and x.get('task_type')==task and x.get('eligibility')=='ELIGIBLE_VISUAL_VQA' and x.get('dependency_class')=='VISUAL_ONLY']
 result={}
 for split,total in [('train',200),('validation',50)]:
  binary=[]
  for label,amount in [('yes',total//2),('no',total//2)]: binary += pick([x for x in filt(vqa,split,'binary_qa') if x['answer']==label],amount,f'{split}|binary|{label}')
  mcq=[]
  for label,amount in zip('abcd',[total//4+(i<total%4) for i in range(4)]): mcq += pick([x for x in filt(vqa,split,'multiple_choice_qa') if x['answer']==label],amount,f'{split}|mcq|{label}')
  captions=pick(filt(cap,split,'caption'),total,f'{split}|caption')
  result[split]=sorted(binary+mcq+captions,key=lambda x:x['record_id'])
 trainimgs={x['image_id'] for x in result['train']}; valimgs={x['image_id'] for x in result['validation']}
 if trainimgs&valimgs: raise RuntimeError('PHASE3O12_BLOCKED_IMAGE_OVERLAP')
 manifest={'phase':'3O.12','status':'REGISTERED','seed':SEED,'test_access_count':0,'selection_algorithm':'sha256(seed|stratum|record_id) ascending','selection':result,'counts':{s:{t:sum(x['task_type']==t for x in result[s]) for t in ('binary_qa','multiple_choice_qa','caption')} for s in result},'binary_labels':{s:{a:sum(x['task_type']=='binary_qa' and x['answer']==a for x in result[s]) for a in ('yes','no')} for s in result},'mcq_keys':{s:{a:sum(x['task_type']=='multiple_choice_qa' and x['answer']==a for x in result[s]) for a in 'abcd'} for s in result},'train_validation_image_overlap':0}
 (OUT/'phase3o12_training_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 random.seed(SEED); torch.manual_seed(SEED); p=S2MultispectralProjector(); initial=OUT/'phase3o12_projector_initial.pt'; torch.save({'state_dict':p.state_dict(),'seed':SEED,'architecture':'S2MultispectralProjector'},initial)
 receipt={'sha256':sha256_file(initial),'fingerprint':p.fingerprint(),'parameter_count':p.trainable_parameter_count,'immutable_pretraining_checkpoint':True}
 (OUT/'initial_projector_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n'); print(json.dumps({'manifest':manifest['counts'],'initial':receipt},indent=2))
if __name__=='__main__': main()
