import json, sys, time
from pathlib import Path
import h5py, torch
from .train import DATA, EXP, CLASS_NAMES, FusionCNN


def main():
    index=int(sys.argv[1]) if len(sys.argv)>1 else 0; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); checkpoint=EXP/"checkpoints"/"best_model.pt"; model=FusionCNN().to(device); model.load_state_dict(torch.load(checkpoint,map_location=device)["model_state"]); model.eval()
    with h5py.File(DATA/"testing.h5","r") as f:
        s1=torch.tensor(f["sen1"][index].transpose(2,0,1),dtype=torch.float32).unsqueeze(0); s2=torch.tensor(f["sen2"][index].transpose(2,0,1),dtype=torch.float32).unsqueeze(0); truth=int(f["label"][index].argmax())
    with torch.no_grad(): probabilities=torch.softmax(model(s1.to(device),s2.to(device)),1)[0].cpu().tolist()
    prediction=int(max(range(17),key=lambda i:probabilities[i])); result={"dataset":"So2Sat-LCZ42","split":"testing","sample_id":f"testing_{index}","prediction":CLASS_NAMES[prediction],"predicted_class_index":prediction,"ground_truth":CLASS_NAMES[truth],"ground_truth_class_index":truth,"confidence":probabilities[prediction],"class_probabilities":dict(zip(CLASS_NAMES,probabilities)),"model_checkpoint":str(checkpoint),"device":str(device),"processing_time_seconds":0.0,"status":"success"}
    (EXP/"predictions"/"inference_example.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=="__main__": main()