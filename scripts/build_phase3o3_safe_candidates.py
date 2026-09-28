"""Write a bounded Phase 3O.3-safe visual-VQA view; excludes geometry/spatial relation records."""
from __future__ import annotations
import argparse,json
from pathlib import Path

SOURCE=Path("artifacts/annotations/image_language/visual_vqa_candidates.jsonl")
OUT=Path("artifacts/training/phase3o/phase3o3_safe_candidates.jsonl")
ROOTS=(Path(r"D:\Satquery_ai datasets\comparison\raw-1000\BigEarthNet-S2"),Path(r"D:\Satquery_ai datasets\comparison\raw-5000-additional\BigEarthNet-S2"))
BANDS=("B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12")
parser=argparse.ArgumentParser();parser.add_argument('--train',type=int,default=20);parser.add_argument('--validation',type=int,default=5);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();OUT=args.output
index={}
for root in ROOTS:
    for path in root.rglob("*.tif"):
        if "_B" in path.stem: index.setdefault(path.name,path)
selected={"train":[],"validation":[]}; seen={"train":set(),"validation":set()}
with SOURCE.open(encoding="utf8") as stream:
    for line in stream:
        row=json.loads(line); split=row.get("split"); image=row.get("image_id")
        if split not in selected or len(selected[split]) >= {"train":args.train,"validation":args.validation}[split]: continue
        if row.get("task_type") not in {"binary_qa","multiple_choice_qa"} or row.get("dependency_class")!="VISUAL_ONLY": continue
        if row.get("source_category")=="relative pos" or row.get("box") is not None or row.get("point") is not None: continue
        if image in seen[split] or not all(f"{image}_{band}.tif" in index for band in BANDS): continue
        seen[split].add(image); selected[split].append(row)
if len(selected["train"])!=args.train or len(selected["validation"])!=args.validation: raise RuntimeError("EXPANSION_BLOCKED_INSUFFICIENT_VERIFIED_DATA")
if seen["train"] & seen["validation"]: raise RuntimeError("EXPANSION_BLOCKED_SPLIT_COLLISION")
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text("".join(json.dumps(row,sort_keys=True)+"\n" for split in ("train","validation") for row in selected[split]),encoding="utf8")
print(json.dumps({"train":len(selected["train"]),"validation":len(selected["validation"]),"overlap":len(seen["train"]&seen["validation"])}))
