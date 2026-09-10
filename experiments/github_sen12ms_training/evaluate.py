import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .train import DATA, EXP, CLASS_NAMES, FusionCNN, H5Dataset, metrics


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint=EXP/"checkpoints"/"best_model.pt"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    model=FusionCNN().to(device); model.load_state_dict(torch.load(checkpoint,map_location=device)["model_state"]); model.eval(); ds=H5Dataset(DATA/"testing.h5"); loader=DataLoader(ds,batch_size=256,shuffle=False,num_workers=0); ys=[]; preds=[]; probs=[]; started=time.perf_counter()
    with torch.no_grad():
      for s1,s2,y,_ in loader:
        logits=model(s1.to(device),s2.to(device)); ys.extend(y.tolist()); preds.extend(logits.argmax(1).cpu().tolist()); probs.append(torch.softmax(logits,1).cpu().numpy())
    result=metrics(ys,preds); result.update({"dataset":"So2Sat-LCZ42","split":"testing","checkpoint":str(checkpoint),"device":str(device),"runtime_seconds":time.perf_counter()-started,"class_names":CLASS_NAMES,"top1_accuracy":result["accuracy"],"top3_accuracy":float(np.mean([y in np.argsort(p)[-3:] for y,p in zip(ys,np.concatenate(probs))]))})
    (EXP/"metrics"/"evaluation_metrics.json").write_text(json.dumps(result,indent=2)); (EXP/"metrics"/"confusion_matrix.csv").write_text("".join(",".join(map(str,row))+"\n" for row in result["confusion_matrix"]))
    np.save(EXP/"predictions"/"test_probabilities.npy",np.concatenate(probs)); (EXP/"reports"/"evaluation_report.json").write_text(json.dumps(result,indent=2)); print(json.dumps({k:result[k] for k in ('accuracy','balanced_accuracy','macro_f1','weighted_f1','top3_accuracy')},indent=2))


if __name__=="__main__": main()