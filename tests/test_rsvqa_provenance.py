from dataclasses import replace

from src.evaluation.contracts import VerificationStatus
from src.evaluation.rsvqa import AcceptanceState, RSVQADatasetAcceptance
from src.evaluation.rsvqa_provenance import (RSVQASourceEvidence, SourceAuthority,
                                              deterministic_acquisition_manifest)


def evidence(**changes):
    value = RSVQASourceEvidence("RSVQA-LR", SourceAuthority.PRIMARY_AUTHORITATIVE, "10.5281/zenodo.6344334", "ZENODO_VERSION_1.0_DOI", "zenodo-record-6344334", {"all_questions.json": "fed409776edd11790c596ea0848984c7"}, VerificationStatus.VERIFIED, VerificationStatus.VERIFIED, VerificationStatus.VERIFIED, "code-revision")
    return replace(value, **changes)


def test_immutable_primary_source_with_checksums_is_revision_verified():
    assert evidence().revision_verified()


def test_mutable_or_unpinned_source_is_rejected():
    assert not evidence(authority=SourceAuthority.SECONDARY_MIRROR).revision_verified()
    assert "SOURCE_REVISION_UNVERIFIED" in evidence(revision=None).blocking_reasons()


def test_license_requires_annotations_images_and_underlying_imagery():
    assert not evidence(imagery_license=VerificationStatus.UNVERIFIED).acquisition_permitted()
    assert "IMAGE_LICENSE_UNVERIFIED" in evidence(imagery_license=VerificationStatus.UNVERIFIED).blocking_reasons()
    assert "UNDERLYING_IMAGERY_LICENSE_UNVERIFIED" in evidence(underlying_imagery_license=VerificationStatus.UNVERIFIED).blocking_reasons()


def test_split_linkage_provenance_and_leakage_acceptance_states_are_fail_closed():
    complete = {field: AcceptanceState.VERIFIED for field in RSVQADatasetAcceptance.__dataclass_fields__}
    assert not RSVQADatasetAcceptance(**complete).blocking_reasons()
    for name in ("official_test_split", "image_question_linkage", "provenance", "checksums_or_equivalent_artifact_identity", "cross_split_leakage_check"):
        changed = dict(complete); changed[name] = AcceptanceState.MISSING
        assert RSVQADatasetAcceptance(**changed).blocking_reasons()


def test_bounded_acquisition_manifest_is_deterministic_and_records_no_downloads_when_blocked():
    blocked = evidence(imagery_license=VerificationStatus.UNVERIFIED)
    first, second = deterministic_acquisition_manifest(blocked), deterministic_acquisition_manifest(blocked)
    assert first == second
    assert first["license_gate"] == "BLOCKED" and first["downloaded_artifacts"] == ()


def test_evaluator_revision_presence_is_not_an_evaluator_validity_claim():
    assert evidence(code_evaluator_revision="github-commit").code_evaluator_revision == "github-commit"
    assert evidence(code_evaluator_revision=None).code_evaluator_revision is None
