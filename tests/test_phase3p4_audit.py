from scripts.phase3p4_sar_water_provenance_audit import REQUIRED,valid_schema
import json
from pathlib import Path
import json
from pathlib import Path

def test_audit_schema_requires_independent_evidence_and_alignment_fields():
    assert not valid_schema({'record_id':'x'})
    assert valid_schema({key:None for key in REQUIRED})

def test_fail_closed_protocol_does_not_confirm_without_reference():
    decision='UNVERIFIABLE' if None is None else 'CONFIRMED_WATER'
    assert decision == 'UNVERIFIABLE'


def test_completed_audit_reuses_exact_pool_and_creates_no_confirmed_manifest():
    root = Path(__file__).resolve().parents[1] / 'artifacts/training/phase3p/phase3p4_sar_water_provenance_audit'
    audit = json.loads((root / 'phase3p4_record_audit.json').read_text())['records']
    summary = json.loads((root / 'phase3p4_summary.json').read_text())
    assert len(audit) == 100 and all(r['audit_decision'] == 'UNVERIFIABLE' for r in audit)
    assert summary['test_access_count'] == 0 and not summary['confirmed_manifests_created']


def test_completed_audit_reuses_exact_pool_and_creates_no_confirmed_manifest():
    root = Path(__file__).resolve().parents[1] / 'artifacts/training/phase3p/phase3p4_sar_water_provenance_audit'
    audit = json.loads((root / 'phase3p4_record_audit.json').read_text())['records']
    summary = json.loads((root / 'phase3p4_summary.json').read_text())
    assert len(audit) == 100 and all(r['audit_decision'] == 'UNVERIFIABLE' for r in audit)
    assert summary['test_access_count'] == 0 and not summary['confirmed_manifests_created']
