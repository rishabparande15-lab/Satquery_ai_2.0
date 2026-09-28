"""Resolve only existing, auditable local scene references."""
from dataclasses import dataclass
from pathlib import Path
import json
@dataclass(frozen=True)
class SceneReference: dataset_id:str; scene_id:str; modality:str; dataset_revision:str|None=None; temporal_pair_id:str|None=None
class SceneResolutionError(ValueError): pass
def resolve(value):
 if not isinstance(value,dict): raise SceneResolutionError('SCENE_REFERENCE_INVALID')
 scene_id,modality=str(value.get('scene_id','')),str(value.get('modality','')).lower()
 if not scene_id or not modality: raise SceneResolutionError('INPUT_VALIDATION_FAILED')
 if value.get('dataset_id')=='google/RSRCC':
  manifest=Path('artifacts/test_fixtures/temporal_change_fixture_manifest.json')
  if not manifest.is_file() or scene_id not in {x['pair_key'] for x in json.loads(manifest.read_text())['records']}: raise SceneResolutionError('SCENE_NOT_FOUND')
  return {'id':scene_id,'modality':'optical_rgb_pair','source_reference':'fixture:RSRCC','temporal_pair_id':scene_id}
 if value.get('dataset_id')=='pipeline3_5000':
  from src.pipeline3_scene_probe import list_pipeline3_area_ids
  if scene_id not in set(list_pipeline3_area_ids()): raise SceneResolutionError('SCENE_NOT_FOUND')
  return {'id':scene_id,'modality':modality,'source_reference':'pipeline3_5000'}
 raise SceneResolutionError('SCENE_NOT_FOUND')
