"""Run the isolated Pass 5B diagnostic and 18-area token-level ridge probe."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset_loader import OPTICAL_BANDS, SAR_BANDS, discover_samples, load_sample
from src.gee_features import extract_gee_features
from src.pixel_features import aggregate_tokens, candidate_maps, edge_strength, local_mean_std, map_statistics
from src.scientific_artifacts import save_artifact_bundle, verify_artifact_bundle


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def inspect_sources(sample):
    import rasterio
    result = {"optical": {}, "sar": {}}
    for group, paths in (("optical", sample.optical_paths), ("sar", sample.sar_paths)):
        for band, path in paths.items():
            with rasterio.open(path) as ds:
                mask = ds.read_masks(1) > 0
                result[group][band] = {"path": str(path), "shape": [ds.height, ds.width], "dtype": ds.dtypes[0],
                    "crs": str(ds.crs), "transform": list(ds.transform), "bounds": list(ds.bounds),
                    "resolution": list(ds.res), "nodata": ds.nodata, "valid_pixels": int(mask.sum()),
                    "invalid_pixels": int(mask.size-mask.sum()), "sha256": sha(path)}
    return result


def plot_diagnostic(out, prepared, maps, valid):
    rgb = np.stack([maps["B04"], prepared.raw_optical[2], prepared.raw_optical[1]], -1)
    lo, hi = np.nanpercentile(rgb, (2, 98)); rgb = np.clip((rgb-lo)/(hi-lo), 0, 1)
    names = ["NDVI", "NDWI", "NDBI", "BSI", "VV_minus_VH"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0,0].imshow(rgb); axes[0,0].set_title("S2 RGB (north up)")
    axes[0,1].imshow(maps["B08"], cmap="gray"); axes[0,1].set_title("B08")
    axes[0,2].imshow(maps["VV"], cmap="gray"); axes[0,2].set_title("S1 VV")
    axes[0,3].imshow(valid, cmap="gray", vmin=0, vmax=1); axes[0,3].set_title("joint valid mask")
    for ax, name in zip(axes[1], names[:4]):
        ax.imshow(maps[name], cmap="RdYlGn" if name=="NDVI" else "coolwarm"); ax.set_title(name)
    for ax in axes.ravel(): ax.axis("off")
    fig.tight_layout(); fig.savefig(out/"spatial_sanity.png", dpi=140); plt.close(fig)


def ridge_fit(x, y, alpha):
    mu, sd = x.mean(0), x.std(0); sd[sd < 1e-8] = 1
    z = (x-mu)/sd; z = np.column_stack((np.ones(len(z)), z))
    reg = np.eye(z.shape[1]); reg[0,0] = 0
    return np.linalg.solve(z.T@z + alpha*reg, z.T@y), mu, sd


def ridge_predict(model, x):
    w, mu, sd = model; return np.column_stack((np.ones(len(x)), (x-mu)/sd))@w


def run_probe(pipeline_root: Path):
    wanted, rows = {"train":10, "validation":4, "test":4}, []
    pm = json.loads((pipeline_root/"prepared/manifest.json").read_text())
    for batch in pm["batches"]:
        pdir = pipeline_root/"prepared"/batch["directory"]
        tdir = pipeline_root/"targets"/batch["directory"]
        meta = json.loads((pdir/"batch.json").read_text())
        raw = torch.load(pdir/"raw_inputs.pt", map_location="cpu", weights_only=True)
        target = torch.load(tdir/"targets.pt", map_location="cpu", weights_only=True)
        for i, item in enumerate(meta["samples"]):
            split=item["split"]
            if sum(r["split"]==split for r in rows) >= wanted[split]: continue
            optical=raw["optical_images"][i].numpy(); sar=raw["SAR_images"][i].numpy()
            physical, _ = extract_gee_features(optical, sar, OPTICAL_BANDS, SAR_BANDS)
            maps=candidate_maps(optical,sar,OPTICAL_BANDS,SAR_BANDS); _, ndvi_std=local_mean_std(maps["NDVI"],5)
            selected={k:maps[k] for k in ("NDVI","NDWI","NDBI","BSI","VV_minus_VH")}; selected["NDVI_local_std_5"]=ndvi_std
            extra=np.column_stack([aggregate_tokens(v)[0][:,:2] for v in selected.values()])
            mask=target["fully_labeled_mask"][i].numpy().astype(bool)
            rows.append({"id":item["patch_id"],"split":split,"physical":np.repeat(physical[None],mask.sum(),0),
                         "extra":extra[mask],"scene_physical":physical,"scene_extra":extra.mean(0),
                         "target":target["class_fractions"][i].numpy()[mask]})
        if all(sum(r["split"]==s for r in rows)>=wanted[s] for s in wanted): break
    parts={}
    for split in wanted:
        rr=[r for r in rows if r["split"]==split]
        parts[split]={k:np.concatenate([r[k] for r in rr]) for k in ("physical","extra","target")}
    results={}
    for name, add in (("physical",False),("physical_plus_selected_token",True)):
        xt=parts["train"]["physical"]; xv=parts["validation"]["physical"]; xe=parts["test"]["physical"]
        if add:
            xt=np.column_stack((xt,parts["train"]["extra"])); xv=np.column_stack((xv,parts["validation"]["extra"])); xe=np.column_stack((xe,parts["test"]["extra"]))
        candidates=[]
        for alpha in (0.1,1,10,100):
            model=ridge_fit(xt,parts["train"]["target"],alpha)
            candidates.append((np.mean(np.abs(ridge_predict(model,xv)-parts["validation"]["target"])),alpha,model))
        val,alpha,model=min(candidates,key=lambda z:z[0]); pred=np.clip(ridge_predict(model,xe),0,1)
        results[name]={"dimension":int(xt.shape[1]),"alpha_selected_on_validation":alpha,
            "validation_mae_pp":float(val*100),"test_mae_pp":float(np.mean(np.abs(pred-parts["test"]["target"]))*100)}
    scene_p=np.stack([r["scene_physical"] for r in rows]); scene_e=np.stack([r["scene_extra"] for r in rows])
    corr=np.corrcoef(scene_e.T,scene_p.T)[:scene_e.shape[1],scene_e.shape[1]:]
    extra_names=[f"{stat}_{name}" for name in ("NDVI","NDWI","NDBI","BSI","VV_minus_VH","NDVI_local_std_5") for stat in ("mean","std")]
    redundancy={name:{"max_abs_correlation_with_physical":float(np.nanmax(np.abs(corr[i]))),
                      "physical_feature_index":int(np.nanargmax(np.abs(corr[i])))} for i,name in enumerate(extra_names)}
    return {"design":"18-area token-level ridge feasibility probe; not final performance", "areas":{s:[r["id"] for r in rows if r["split"]==s] for s in wanted},
            "token_counts":{s:int(len(parts[s]["target"])) for s in wanted}, "selected_features":["mean/std: "+x for x in ("NDVI","NDWI","NDBI","BSI","VV_minus_VH","NDVI_local_std_5")],
            "redundancy":redundancy, "results":results}


def main():
    local_data_root=Path(os.environ.get("SATQUERY_LOCAL_DATA_ROOT",ROOT/"data"/"raw"))
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",type=Path,default=Path(os.environ.get("DATASET_ROOT",local_data_root/"bigearthnet-v2-small-sample"))); ap.add_argument("--pipeline-root",type=Path,default=Path(os.environ.get("PASS3_PIPELINE_ROOT",local_data_root/"pipeline-1000"))); ap.add_argument("--out",type=Path,default=ROOT/"artifacts/pixel_feature_probe/61_39"); args=ap.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=True); sample=next(s for s in discover_samples(args.dataset_root) if s.patch_id=="61_39")
    source=inspect_sources(sample); start=time.perf_counter(); prepared=load_sample(sample); load_s=time.perf_counter()-start
    start=time.perf_counter(); maps=candidate_maps(prepared.raw_optical,prepared.raw_sar,OPTICAL_BANDS,SAR_BANDS); raw_s=time.perf_counter()-start
    local={}; timings={}
    for name in ("NDVI","VV"):
        for w in (3,5,7):
            start=time.perf_counter(); mean,std=local_mean_std(maps[name],w); timings[f"{name}_{w}"]=time.perf_counter()-start; local[f"{name}_local_std_{w}"]=std
    local["NDVI_edge_strength"]=edge_strength(maps["NDVI"])
    arrays={**maps,**local}; token_arrays={}; start=time.perf_counter()
    for name,value in arrays.items(): token_arrays[f"token_{name}"]=aggregate_tokens(value)[0]
    token_s=time.perf_counter()-start
    stats={name:map_statistics(value) for name,value in arrays.items()}; valid=np.logical_and.reduce([np.isfinite(x) for x in maps.values()])
    plot_diagnostic(out,prepared,maps,valid)
    provenance={"dataset":"BigEarthNet v2 three-sample verified extract","sample_id":"61_39","source_bands":{"optical":list(OPTICAL_BANDS),"sar":list(SAR_BANDS)},
        "formulas_version":"pixel_features_v1","preprocessing":"raw bands reprojected to B02 grid; indices computed before normalization","grid":prepared.metadata,
        "source_files":source,"feature_statistics":stats,"timings_seconds":{"load_align":load_s,"raw_features":raw_s,"local_features":timings,"all_token_aggregation":token_s}}
    save_artifact_bundle(out,sample_id="61_39",arrays={"valid_mask":valid.astype(np.uint8),**arrays,**token_arrays},provenance=provenance)
    verified=verify_artifact_bundle(out); probe=run_probe(args.pipeline_root); (out/"experiment.json").write_text(json.dumps(probe,indent=2)+"\n")
    print(json.dumps({"artifact":str(out),"verification":verified["status"],"probe":probe["results"]},indent=2))

if __name__ == "__main__": main()
