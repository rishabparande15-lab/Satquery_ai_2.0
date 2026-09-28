import json
from pathlib import Path
import pytest
from src.eo_vlm.phase3o16r_protocol import REQUIRED, canonical_sha, classify, validate_preregistration
def prereg():
 x={'success_gates':{k:{'all_required':sorted(v)} for k,v in REQUIRED.items()}};x['preregistration_sha256']=canonical_sha(x);return x
def test_all_gate_fields_persisted(): validate_preregistration(prereg())
def test_missing_gate_field_fails():
 x=prereg();x['success_gates']['binary']['all_required'].pop();x['preregistration_sha256']=canonical_sha({k:v for k,v in x.items() if k!='preregistration_sha256'})
 try: validate_preregistration(x);assert False
 except ValueError: pass
def test_missing_evidence_fails_closed(): assert classify('binary',{})[0]=='NOT_PROMISING'
def test_deterministic_classification():
 evidence={x:True for x in REQUIRED['mcq']};assert classify('mcq',evidence)==classify('mcq',evidence)==('PROMISING',[])
def test_recovery_artifacts_are_complete_when_present():
 root=Path('artifacts/training/phase3o/phase3o16r_recovery')
 if not root.exists(): pytest.skip('recovery has not been prepared')
 validate_preregistration(json.loads((root/'phase3o16r_preregistration.json').read_text()))
 for task in ('binary','mcq','caption'):
  receipt=json.loads((root/f'phase3o16r_{task}_results.json').read_text())
  for variant in receipt['variants'].values():
   assert (Path(variant['checkpoint_path'])).is_file()
   assert (Path(variant['raw_validation_output_path'])).is_file()
   assert variant['checkpoint_sha256'] and variant['checkpoint_fingerprint']
   assert variant['gate_result'] in {'PROMISING','NOT_PROMISING'}
