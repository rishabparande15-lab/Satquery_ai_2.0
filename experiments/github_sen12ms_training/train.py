import csv
import json
import logging
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(r"D:\Satquery_ai datasets\github_datasets\So2Sat-LCZ42")
EXP = ROOT / "experiment"
DATA = ROOT / "data"
CLASS_NAMES = ["Compact high-rise", "Compact mid-rise", "Compact low-rise", "Open high-rise", "Open mid-rise", "Open low-rise", "Lightweight low-rise", "Large low-rise", "Sparsely built", "Heavy industry", "Dense trees", "Scattered trees", "Bush and scrub", "Low plants", "Bare rock or paved", "Bare soil or sand", "Water"]


class H5Dataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.length = 0
        with h5py.File(path, "r") as file:
            self.length = file["label"].shape[0]
        self.file = None

    def _open(self):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        return self.file

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file = self._open()
        sen1 = np.asarray(file["sen1"][index], dtype=np.float32).transpose(2, 0, 1)
        sen2 = np.asarray(file["sen2"][index], dtype=np.float32).transpose(2, 0, 1)
        label = int(np.argmax(file["label"][index]))
        return torch.from_numpy(sen1), torch.from_numpy(sen2), label, index


class FusionCNN(nn.Module):
    def __init__(self, classes=17):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(8, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.s2 = nn.Sequential(nn.Conv2d(10, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, sen1, sen2):
        return self.head(torch.cat([self.s1(sen1).flatten(1), self.s2(sen2).flatten(1)], dim=1))


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)), "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)), "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)), "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), "classification_report": classification_report(y_true, y_pred, labels=list(range(17)), target_names=CLASS_NAMES, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(17))).tolist()}


def run_epoch(model, loader, optimizer, device, training):
    model.train(training); total=0.0; ys=[]; preds=[]; count=0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for sen1, sen2, labels, _ in loader:
            sen1, sen2, labels=sen1.to(device), sen2.to(device), labels.to(device)
            logits=model(sen1,sen2); loss=nn.functional.cross_entropy(logits,labels)
            if training: optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.item())*labels.size(0); count += labels.size(0); ys.extend(labels.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist())
    return total/count, metrics(ys,preds)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    requested_device=os.environ.get("SO2SAT_DEVICE", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this Python environment has CPU-only PyTorch (torch.version.cuda=None). Install a CUDA-enabled PyTorch build before training.")
    device=torch.device(requested_device); batch_size=256; epochs=1
    logging.info("Device: %s", device)
    if device.type == "cuda": logging.info("GPU: %s", torch.cuda.get_device_name(0))
    (EXP/"checkpoints").mkdir(parents=True,exist_ok=True); (EXP/"reports").mkdir(parents=True,exist_ok=True); (EXP/"predictions").mkdir(parents=True,exist_ok=True); (EXP/"metrics").mkdir(parents=True,exist_ok=True); (EXP/"logs").mkdir(parents=True,exist_ok=True)
    train_ds=H5Dataset(DATA/"training.h5"); val_ds=H5Dataset(DATA/"validation.h5")
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    model=FusionCNN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3); history=[]; best=1e9; started=time.perf_counter()
    for epoch in range(1,epochs+1):
        train_loss,train_m=run_epoch(model,train_loader,optimizer,device,True); val_loss,val_m=run_epoch(model,val_loader,optimizer,device,False)
        row={"epoch":epoch,"train_loss":train_loss,"validation_loss":val_loss,"train_accuracy":train_m["accuracy"],"validation_accuracy":val_m["accuracy"],"validation_macro_f1":val_m["macro_f1"]}; history.append(row); (EXP/"checkpoints"/f"epoch_{epoch:03d}.pt").write_bytes(b"") if False else torch.save({"model_state":model.state_dict(),"config":{"seed":seed,"batch_size":batch_size,"epochs":epochs,"architecture":"FusionCNN","class_names":CLASS_NAMES}}, EXP/"checkpoints"/f"epoch_{epoch:03d}.pt")
        if val_loss<best: best=val_loss; torch.save({"model_state":model.state_dict(),"config":{"seed":seed,"batch_size":batch_size,"epochs":epochs,"architecture":"FusionCNN","class_names":CLASS_NAMES}},EXP/"checkpoints"/"best_model.pt")
        logging.info("epoch=%d train_loss=%.4f val_loss=%.4f val_f1=%.4f",epoch,train_loss,val_loss,val_m["macro_f1"])
    (EXP/"reports"/"training_history.json").write_text(json.dumps({"device":str(device),"runtime_seconds":time.perf_counter()-started,"history":history},indent=2))
    (EXP/"reports"/"training_report.json").write_text(json.dumps({"status":"success","dataset":"So2Sat-LCZ42","architecture":"S1+S2 FusionCNN","device":str(device),"checkpoint":str(EXP/"checkpoints"/"best_model.pt"),"history":history},indent=2))
    logging.info("training complete checkpoint=%s",EXP/"checkpoints"/"best_model.pt")


if __name__=="__main__": main()