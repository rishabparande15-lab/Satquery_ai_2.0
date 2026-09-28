"""Controlled Phase 3O.2 S2 projector adaptation pilot; no benchmark evaluation."""
from __future__ import annotations
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import psutil, rasterio, torch
from rasterio.enums import Resampling
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.dataset_loader import _read_to_grid
from src.preprocessing import _robust_channel_scale
from src.eo_vlm.s2_asset_gate import S2_BANDS, canonical_bytes, inspect_area, sha256_file
from src.eo_vlm.multispectral_projector import S2MultispectralProjector
from src.eo_vlm.training import build_qwen_token_batch, freeze_qwen
from src.eo_vlm_adapter import Qwen25VLRGBAdapter

MODEL_REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"
MODEL_SNAPSHOT=Path(r"C:\Users\Rishab\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-3B-Instruct\snapshots")/MODEL_REVISION
ROOTS=[Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"),Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2")]
OUT=ROOT/"artifacts/training/phase3o"; ANNOTATIONS=ROOT/"artifacts/annotations/image_language/visual_vqa_candidates.jsonl"

def dump(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf8")
def hash_state(module):
    h=hashlib.sha256()
    for n,p in module.named_parameters():
        x=p.detach().reshape(-1); sample=torch.cat((x[:16],x[-16:])).to("cpu",torch.float32).numpy().tobytes() if x.numel() else b""
        h.update(n.encode());h.update(str(tuple(p.shape)).encode());h.update(sample)
    return h.hexdigest()
def file_index():
    result={}
    for root in ROOTS:
        for p in root.rglob("*.tif"):
            if "_B" in p.stem: result.setdefault(p.name,p)
    return result
def cube(asset):
    base=Path(asset["local_path"]); paths={b:next(base.glob(f"*_{b}.tif")) for b in S2_BANDS}
    with rasterio.open(paths["B02"]) as g: values=np.stack([_read_to_grid(paths[b],g,Resampling.bilinear) for b in S2_BANDS])
    return torch.from_numpy(_robust_channel_scale(values).astype(np.float32)).unsqueeze(0)
def combined_hash(asset):
    base=Path(asset["local_path"]); h=hashlib.sha256()
    for b in S2_BANDS: h.update(b.encode());h.update(bytes.fromhex(sha256_file(next(base.glob(f"*_{b}.tif")))))
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--prefix",default="phase3o2"); args=parser.parse_args(); prefix=args.prefix
    manifest=json.loads((OUT/"phase3o_s2_training_manifest.json").read_text(encoding="utf8")); frozen=dict(manifest); claimed=frozen.pop("manifest_sha256")
    if hashlib.sha256(canonical_bytes(frozen)).hexdigest()!=claimed: raise RuntimeError("TRAINING_BLOCKED_MANIFEST_MISMATCH")
    assets=json.loads((OUT/"verified_s2_training_assets.json").read_text(encoding="utf8"))["records"]
    if len(assets)!=5 or any(a["split"]!="train" or combined_hash(a)!=a["sha256"] for a in assets): raise RuntimeError("TRAINING_BLOCKED_MANIFEST_MISMATCH")
    index=file_index(); annotations={}
    validation=None
    with ANNOTATIONS.open(encoding="utf8") as stream:
        for line in stream:
            r=json.loads(line); annotations[r["record_id"]]=r
            a=r["image_id"]
            if validation is None and r.get("split")=="validation" and r.get("effective_partition")=="validation" and r.get("task_type") in {"binary_qa","multiple_choice_qa"} and r.get("source_category")!="relative pos" and all(f"{a}_{b}.tif" in index for b in S2_BANDS):
                validation=inspect_area(r,{b:index[f"{a}_{b}.tif"] for b in S2_BANDS}).to_dict(); validation_record=r
    if validation is None: raise RuntimeError("ADAPTATION_PILOT_BLOCKED_VALIDATION_UNAVAILABLE")
    train_records=[annotations[a["sample_id"]] for a in assets]
    train_ids={r["image_id"] for r in train_records}; val_ids={validation["image_id"]}
    if train_ids & val_ids: raise RuntimeError("ADAPTATION_PILOT_BLOCKED_SPLIT_CONTAMINATION")
    torch.manual_seed(17);np.random.seed(17); process=psutil.Process(); adapter=Qwen25VLRGBAdapter(); runtime=adapter.load_model(MODEL_SNAPSHOT,dtype="float16",device="cuda");model=runtime["model"];model.eval(); freeze=freeze_qwen(model)
    if freeze["qwen_trainable_parameter_count"]: raise RuntimeError("TRAINING_BLOCKED_QWEN_NOT_FROZEN")
    projector=S2MultispectralProjector().to("cuda"); initial_hash=hash_state(projector);qwen_hash_before=hash_state(model);before={n:p.detach().cpu().clone() for n,p in projector.named_parameters()}; optimizer=torch.optim.AdamW(projector.parameters(),lr=1e-3,weight_decay=1e-4)
    if any(p in {q for g in optimizer.param_groups for q in g["params"]} for p in model.parameters()): raise RuntimeError("TRAINING_BLOCKED_QWEN_NOT_FROZEN")
    def loss_for(record,asset,train=False):
        batch=build_qwen_token_batch(tokenizer=runtime["processor"].tokenizer,model=model,projector=projector,s2=cube(asset),questions=[record["question"]],answers=[record["answer"]]); out=model(**batch); return out.loss
    torch.cuda.reset_peak_memory_stats(); initial_validation=float(loss_for(validation_record,validation).detach().cpu()); initial_train=float(loss_for(train_records[0],assets[0]).detach().cpu()); steps=[]; start=time.perf_counter()
    for step,(record,asset) in enumerate(zip(train_records,assets),1):
        optimizer.zero_grad(set_to_none=True); loss=loss_for(record,asset,True)
        if not torch.isfinite(loss): raise RuntimeError("ADAPTATION_FAILED_NONFINITE_LOSS")
        loss.backward()
        if not all(p.grad is not None for p in projector.parameters()): raise RuntimeError("ADAPTATION_FAILED_MISSING_GRADIENT")
        optimizer.step();steps.append({"step":step,"train_loss":float(loss.detach().cpu()),"finite_loss":True,"learning_rate":0.001,"peak_vram":int(torch.cuda.max_memory_allocated())})
    duration=time.perf_counter()-start
    with torch.no_grad(): final_validation=float(loss_for(validation_record,validation).detach().cpu())
    final_hash=hash_state(projector); deltas=[(p.detach().cpu()-before[n]).abs() for n,p in projector.named_parameters()]; changed=sum(int(d.ne(0).any()) for d in deltas); total=float(sum(d.sum() for d in deltas)); maximum=float(max(d.max() for d in deltas))
    qwen_hash_after=hash_state(model)
    if not changed or not total or qwen_hash_before!=qwen_hash_after: raise RuntimeError("ADAPTATION_FAILED_PARAMETER_INTEGRITY")
    config={"seed":17,"batch_size":1,"gradient_accumulation":1,"learning_rate":0.001,"optimizer":"AdamW","precision":"float16","max_sequence_length":32,"num_steps":5,"qwen_frozen":True,"training_manifest_sha256":claimed}; config_hash=hashlib.sha256(canonical_bytes(config)).hexdigest(); checkpoint=OUT/f"{prefix}_s2_projector.pt"; torch.save({"state_dict":projector.state_dict(),"adapter_id":"phase3g_s2_qwen_token_projector_v1","config":config},checkpoint); checkpoint_hash=sha256_file(checkpoint)
    del optimizer,projector,model,runtime;gc.collect();torch.cuda.empty_cache()
    runtime=adapter.load_model(MODEL_SNAPSHOT,dtype="float16",device="cuda");model=runtime["model"];model.eval();freeze_qwen(model);projector=S2MultispectralProjector().to("cuda");projector.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=False)["state_dict"])
    with torch.no_grad(): reload_loss=float(loss_for(validation_record,validation).detach().cpu())
    match=abs(reload_loss-final_validation)<=1e-5
    receipt={"phase":"3O.2","status":"ADAPTATION_PILOT_COMPLETE" if match else "ADAPTATION_PILOT_FAILED","seed":17,"train_samples":len(train_records),"validation_samples":1,"test_samples_accessed":0,"task_mix":{k:sum(r["task_type"]==k for r in train_records) for k in {r["task_type"] for r in train_records}},"base_model":adapter.model_id,"base_model_revision":MODEL_REVISION,"qwen_frozen":True,"trainable_parameter_count":sum(p.numel() for p in projector.parameters()),"initial_projector_hash":initial_hash,"final_projector_hash":final_hash,"changed_tensor_count":changed,"total_parameter_delta":total,"max_parameter_delta":maximum,"initial_train_loss":initial_train,"final_loss":steps[-1]["train_loss"],"initial_validation_loss":initial_validation,"final_validation_loss":final_validation,"checkpoint_path":str(checkpoint.relative_to(ROOT)),"checkpoint_size":checkpoint.stat().st_size,"checkpoint_sha256":checkpoint_hash,"training_manifest_sha256":claimed,"training_config_sha256":config_hash,"peak_vram":max(x["peak_vram"] for x in steps),"peak_ram":process.memory_info().rss,"training_duration":duration,"reload_validation_loss":reload_loss,"reload_match_status":"PASS" if match else "FAIL","validation_sample_ids":[validation["sample_id"]],"train_sample_ids":[r["record_id"] for r in train_records],"steps":steps,"scientific_baseline_changed":False,"benchmark_inference_executed":False,"representation":"S2_PROJECTED"}
    dump(OUT/f"{prefix}_run_receipt.json",receipt); print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=="__main__": main()
