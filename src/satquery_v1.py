"""SatQuery v1 paired S1+S2 primary inference controller.

This controller is intentionally separate from the historical generic agent
controller.  It exposes only the approved BigEarthNet paired multimodal route;
S2-only is deliberately not used as an implicit fallback.
"""
from __future__ import annotations
from pathlib import Path
import os
from typing import Any, Mapping
import numpy as np
from .croma_adapter import CROMAAdapter
from .eo_vlm.optical_sar_joint import (EXPECTED_JOINT_SHA, TASK_TYPES, load_verified_joint_projector,
    module_fingerprint, run_optical_sar_joint, validate_paired_inputs)
from .eo_vlm.training import freeze_qwen
from .eo_vlm_adapter import MODEL_ID, Qwen25VLRGBAdapter
from .single_image_vqa import QWEN_REVISION
from scripts.phase3o15_contrastive_objective_ablation import SNAPSHOT, hash_state

ROOT=Path(__file__).resolve().parents[1]
JOINT=ROOT/'artifacts/training/phase3q/phase3q1_fusion_adaptation/phase3q1_joint_projector_final.pt'


def _resolve_croma_path(env_names: tuple[str, ...], default: Path | str = '') -> Path:
    for name in env_names:
        value = os.environ.get(name)
        if value and value.strip():
            return Path(value)
    return Path(default)


CS=_resolve_croma_path(('SATQUERY_CROMA_SOURCE','CROMA_SOURCE'))
CK=_resolve_croma_path(('SATQUERY_CROMA_CHECKPOINT','CROMA_CHECKPOINT'))


class SatQueryV1Controller:
    """Lazy, fail-closed controller for `MULTIMODAL_S1_S2` only."""
    route='MULTIMODAL_S1_S2'
    def __init__(self, *, croma_source: Path | None=None, croma_checkpoint: Path | None=None, joint_checkpoint: Path=JOINT,
                 device: str='cuda'):
        source = croma_source if croma_source is not None else _resolve_croma_path(('SATQUERY_CROMA_SOURCE','CROMA_SOURCE'))
        checkpoint = croma_checkpoint if croma_checkpoint is not None else _resolve_croma_path(('SATQUERY_CROMA_CHECKPOINT','CROMA_CHECKPOINT'))
        self.croma_source,self.croma_checkpoint,self.joint_checkpoint,self.device=Path(source),Path(checkpoint),Path(joint_checkpoint),device
        self.croma=self.projector=self.model=self.tokenizer=None;self.qwen_revision=QWEN_REVISION;self.qwen_model_id=MODEL_ID;self._hashes={}
    def load(self)->None:
        if self.model is not None:return
        self.projector,joint_sha,_=load_verified_joint_projector(self.joint_checkpoint,device=self.device)
        self.croma=CROMAAdapter(self.croma_source,self.croma_checkpoint,device=self.device)
        for p in self.croma.model.parameters():p.requires_grad_(False)
        runtime=Qwen25VLRGBAdapter().load_model(SNAPSHOT,dtype='float16',device=self.device);self.model=runtime['model'].eval();self.tokenizer=runtime['processor'].tokenizer;freeze_qwen(self.model)
        self._hashes={'qwen':hash_state(self.model),'croma':module_fingerprint(self.croma.model),'joint':module_fingerprint(self.projector),'joint_sha':joint_sha}
    def run_satquery(self, *, task_type:str, question:str, s1:np.ndarray|None, s2:np.ndarray|None,
                     patch_id:str, s1_patch_id:str|None, s2_patch_id:str|None, spatial_metadata:Mapping[str,Any]|None)->dict:
        if s1 is None:return self._failure(task_type,patch_id,'MISSING_S1_FOR_MULTIMODAL_ROUTE')
        if s2 is None:return self._failure(task_type,patch_id,'MISSING_S2_FOR_MULTIMODAL_ROUTE')
        try:validate_paired_inputs(s1=s1,s2=s2,s1_patch_id=s1_patch_id,s2_patch_id=s2_patch_id,expected_patch_id=patch_id,metadata=spatial_metadata)
        except Exception as error:return self._failure(task_type,patch_id,str(error))
        try:
            self.load()
        except FileNotFoundError as error:
            return self._failure(task_type,patch_id,'MISSING_CROMA_ASSET' if 'CROMA' in str(error) or 'croma' in str(error) else 'MISSING_JOINT_CHECKPOINT')
        except Exception as error:
            return self._failure(task_type,patch_id,'UNAPPROVED_JOINT_CHECKPOINT' if 'CHECKPOINT' in str(error) else str(error))
        raw=run_optical_sar_joint(croma=self.croma,projector=self.projector,model=self.model,tokenizer=self.tokenizer,s1=s1,s2=s2,s1_patch_id=s1_patch_id,s2_patch_id=s2_patch_id,expected_patch_id=patch_id,metadata=spatial_metadata,record_id=patch_id,task_type=task_type,question=question,qwen_revision=self.qwen_revision,joint_checkpoint_sha=EXPECTED_JOINT_SHA)
        provenance = {**raw['provenance'], 'model_id': self.qwen_model_id} if raw['provenance'] else raw['provenance']
        return {'status':'OK' if raw['error_status'] is None else 'FAILED','task_type':task_type,'generated_response':raw['generated_text'],'parsed_answer':raw['parsed_answer'],'route':self.route,'modalities_used':['S1','S2'],'bigearthnet_patch_id':patch_id,'croma_mode':'CROMA_JOINT','joint_projector_sha256':EXPECTED_JOINT_SHA,'model_id':self.qwen_model_id,'qwen_revision':self.qwen_revision,'warnings':['bounded fixed-panel integration; no claim of fusion accuracy gain'],'error_code':raw['error_status'],'provenance':provenance,'diagnostics':raw['diagnostics']}
    def _failure(self,task_type,patch_id,code):
        return {'status':'FAILED','task_type':task_type,'generated_response':None,'parsed_answer':None,'route':self.route,'modalities_used':[],'bigearthnet_patch_id':patch_id,'croma_mode':None,'joint_projector_sha256':EXPECTED_JOINT_SHA,'model_id':self.qwen_model_id,'qwen_revision':self.qwen_revision,'warnings':[],'error_code':code,'provenance':None,'diagnostics':{}}
    def unchanged(self)->dict[str,bool]:
        if self.model is None:return {}
        return {'qwen':hash_state(self.model)==self._hashes['qwen'],'croma':module_fingerprint(self.croma.model)==self._hashes['croma'],'joint_projector':module_fingerprint(self.projector)==self._hashes['joint']}
    def close(self)->None:
        """Release resident model references at a specialist-family boundary."""
        self.croma=self.projector=self.model=self.tokenizer=None
        self._hashes={}
