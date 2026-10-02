"""Fail-closed bi-temporal change-description contract.

This module intentionally keeps the LEVIR-CC development data boundary separate
from SatQuery's existing S1+S2 and single-image routes.  A caption is returned
only by an explicitly supplied, validated Chg2Cap-compatible runtime; it never
falls back to a generic VLM or to visual heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping
import importlib
import json
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError


ROUTE = "TEMPORAL_CHANGE_DESCRIPTION"
TEMPORAL_ORDER = "PRE_POST"
SPECIALIST = "Chg2Cap"
ALLOWED_SPLITS = frozenset({"train", "validation", "external_inference"})
ALLOWED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})
ROOT = Path(__file__).resolve().parents[1]
CHG2CAP_SOURCE = ROOT / "artifacts" / "final" / "temporal" / "phase3v2_change_description" / "_source" / "Chg2Cap-main"
CHG2CAP_CHECKPOINT = ROOT / "checkpoints" / "chg2cap" / "LEVIR_CC_batchsize_32_resnet101.pth"
CHG2CAP_VOCAB = ROOT / "artifacts" / "final" / "temporal" / "phase3v2a_inference_unblock" / "rscama_vocab.json"
CHG2CAP_EXPECTED_CHECKPOINT_SHA256 = "d737a92afa3cb76a07f672ee905afb59278211c77ccce517a37ae4637110194c"


class TemporalChangeInputError(ValueError):
    """Structured caller error with a stable API-safe code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TemporalImage:
    path: Path
    identity: str
    width: int
    height: int
    sha256: str


@dataclass(frozen=True)
class TemporalRecord:
    record_id: str
    image_t1_path: str
    image_t2_path: str
    temporal_order: str
    caption_references: tuple[str, ...]
    split: str
    source_dataset: str
    source_provenance: Mapping[str, Any]


def _load_image(path: str | Path, role: str) -> TemporalImage:
    if not path:
        raise TemporalChangeInputError(f"TEMPORAL_{role}_MISSING", f"{role} image is required")
    value = Path(path)
    if not value.is_file():
        raise TemporalChangeInputError(f"TEMPORAL_{role}_MISSING", f"{role} image is unavailable")
    if value.suffix.lower() not in ALLOWED_SUFFIXES:
        raise TemporalChangeInputError("UNSUPPORTED_TEMPORAL_FORMAT", "Temporal inputs must be PNG, JPEG, or TIFF images")
    try:
        with Image.open(value) as image:
            image.verify()
        with Image.open(value) as image:
            width, height = image.size
            if image.mode not in {"RGB", "RGBA"}:
                raise TemporalChangeInputError("INVALID_TEMPORAL_IMAGE", f"{role} must have RGB-compatible image data")
    except TemporalChangeInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise TemporalChangeInputError(f"CORRUPT_TEMPORAL_{role}", f"{role} image cannot be decoded") from exc
    if width < 1 or height < 1:
        raise TemporalChangeInputError("INVALID_TEMPORAL_IMAGE", "Temporal image dimensions must be positive")
    digest = sha256(value.read_bytes()).hexdigest()
    return TemporalImage(value.resolve(), value.name, width, height, digest)


def validate_temporal_change_input(
    *,
    t1_path: str | Path | None,
    t2_path: str | Path | None,
    temporal_order: str,
    pair_id: str | None = None,
    split: str = "external_inference",
) -> tuple[TemporalImage, TemporalImage]:
    """Validate an explicitly ordered, spatially compatible change pair."""
    if temporal_order != TEMPORAL_ORDER:
        raise TemporalChangeInputError("INVALID_TEMPORAL_ORDER", "Temporal change description requires temporal_order=PRE_POST")
    if split not in ALLOWED_SPLITS:
        raise TemporalChangeInputError("TEST_SPLIT_PROHIBITED", "Only TRAIN, VALIDATION, or external inference pairs are allowed")
    if pair_id is not None and (not isinstance(pair_id, str) or not pair_id.strip()):
        raise TemporalChangeInputError("INVALID_TEMPORAL_PAIR", "pair_id must be a non-empty string when supplied")
    t1 = _load_image(t1_path, "T1")
    t2 = _load_image(t2_path, "T2")
    if t1.path == t2.path or t1.sha256 == t2.sha256:
        raise TemporalChangeInputError("IDENTICAL_TEMPORAL_INPUTS", "T1 and T2 must be distinct images")
    if (t1.width, t1.height) != (t2.width, t2.height):
        raise TemporalChangeInputError("INCOMPATIBLE_TEMPORAL_DIMENSIONS", "T1 and T2 must have identical pixel dimensions")
    return t1, t2


def validate_temporal_image_path(path: str | Path, role: str) -> TemporalImage:
    """Validate one staged RGB temporal upload before its counterpart arrives."""
    if role not in {"T1", "T2"}:
        raise ValueError("role must be T1 or T2")
    return _load_image(path, role)


class Chg2CapCaptioner:
    """Lazy, checkpoint-pinned Chg2Cap runner for explicitly ordered RGB pairs."""

    def __init__(self, *, source_root: Path = CHG2CAP_SOURCE, checkpoint: Path = CHG2CAP_CHECKPOINT,
                 vocabulary: Path = CHG2CAP_VOCAB, device: str = "cuda") -> None:
        self.source_root, self.checkpoint, self.vocabulary, self.device = Path(source_root), Path(checkpoint), Path(vocabulary), device
        self.encoder = self.transformer = self.decoder = None
        self.word_vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.last_generated_tokens: list[int] = []

    def _load(self) -> None:
        if self.decoder is not None:
            return
        if self.device != "cuda":
            raise RuntimeError("TEMPORAL_MODEL_REQUIRES_CUDA")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("TEMPORAL_MODEL_UNAVAILABLE")
        if not self.source_root.is_dir() or not self.checkpoint.is_file():
            raise RuntimeError("TEMPORAL_MODEL_UNAVAILABLE")
        if not self.vocabulary.is_file():
            raise RuntimeError("TEMPORAL_VOCAB_UNAVAILABLE")
        if sha256(self.checkpoint.read_bytes()).hexdigest() != CHG2CAP_EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("TEMPORAL_CHECKPOINT_UNAPPROVED")
        vocab = json.loads(self.vocabulary.read_text(encoding="utf-8"))
        if not isinstance(vocab, dict) or len(vocab) != 501 or {key: vocab.get(key) for key in ("<NULL>", "<UNK>", "<START>", "<END>")} != {"<NULL>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}:
            raise RuntimeError("TEMPORAL_VOCAB_UNAVAILABLE")
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        encoder_module = importlib.import_module("model.model_encoder")
        decoder_module = importlib.import_module("model.model_decoder")
        # Chg2Cap uses torchvision's legacy pretrained=True constructor even
        # though the pinned checkpoint replaces every encoder parameter.
        original_resnet101 = encoder_module.models.resnet101
        encoder_module.models.resnet101 = lambda pretrained=True: original_resnet101(weights=None)
        try:
            encoder = encoder_module.Encoder("resnet101")
        finally:
            encoder_module.models.resnet101 = original_resnet101
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        output_dim = checkpoint["decoder_dict"]["wdc.weight"].shape[0]
        if output_dim != len(vocab):
            raise RuntimeError("TEMPORAL_VOCAB_UNAVAILABLE")
        transformer = encoder_module.AttentiveEncoder(n_layers=3, feature_size=[16, 16, 2048], heads=8,
                                                       hidden_dim=512, attention_dim=2048, dropout=0.1)
        decoder = decoder_module.DecoderTransformer(encoder_dim=2048, feature_dim=2048, vocab_size=len(vocab),
                                                     max_lengths=41, word_vocab=vocab, n_head=8, n_layers=1, dropout=0.1)
        encoder.load_state_dict(checkpoint["encoder_dict"], strict=True)
        transformer.load_state_dict(checkpoint["encoder_trans_dict"], strict=True)
        decoder.load_state_dict(checkpoint["decoder_dict"], strict=True)
        self.encoder, self.transformer, self.decoder = encoder.eval().cuda(), transformer.eval().cuda(), decoder.eval().cuda()
        self.word_vocab = {str(key): int(value) for key, value in vocab.items()}
        self.reverse_vocab = {value: key for key, value in self.word_vocab.items()}

    @staticmethod
    def _tensor(path: Path):
        import torch
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
        array = np.moveaxis(array, -1, 0)
        mean = np.asarray([100.6790, 99.5023, 84.9932], dtype=np.float32)[:, None, None]
        std = np.asarray([50.9820, 48.4838, 44.7057], dtype=np.float32)[:, None, None]
        return torch.from_numpy((array - mean) / std).unsqueeze(0).cuda()

    def __call__(self, t1: Path, t2: Path) -> str:
        self._load()
        import torch
        assert self.encoder is not None and self.transformer is not None and self.decoder is not None
        with torch.inference_mode():
            features_t1, features_t2 = self.encoder(self._tensor(t1), self._tensor(t2))
            features_t1, features_t2 = self.transformer(features_t1, features_t2)
            sequence = self.decoder.sample(features_t1, features_t2)
        raw_tokens = sequence.detach().flatten().cpu().tolist() if hasattr(sequence, "detach") else sequence
        tokens = [int(value) for value in raw_tokens]
        self.last_generated_tokens = tokens
        words = [self.reverse_vocab[index] for index in tokens if index in self.reverse_vocab and self.reverse_vocab[index] not in {"<NULL>", "<START>", "<END>"}]
        return " ".join(words).strip()

    @property
    def provenance(self) -> dict[str, Any]:
        return {"model": SPECIALIST, "checkpoint": self.checkpoint.name,
                "checkpoint_sha256": CHG2CAP_EXPECTED_CHECKPOINT_SHA256,
                "vocab_source": str(self.vocabulary.relative_to(ROOT)).replace("\\", "/"),
                "vocab_size": len(self.word_vocab) or 501, "device": self.device,
                "generated_tokens": list(self.last_generated_tokens)}


def validate_temporal_image_path(path: str | Path, role: str) -> TemporalImage:
    """Validate one staged RGB temporal upload before its counterpart arrives."""
    if role not in {"T1", "T2"}:
        raise ValueError("role must be T1 or T2")
    return _load_image(path, role)


class TemporalChangeDescriptionController:
    """Thin controller around an admitted Chg2Cap-compatible caption runner.

    The runner must accept absolute paths in PRE_POST order and return a
    non-empty natural-language caption.  It is deliberately injected so this
    repository cannot silently substitute an unapproved architecture.
    """

    route = ROUTE
    specialist = SPECIALIST

    def __init__(self, captioner: Callable[[Path, Path], str] | None = None, *, model_provenance: Mapping[str, Any] | None = None) -> None:
        self._captioner = captioner or Chg2CapCaptioner()
        self._model_provenance = dict(model_provenance or {})

    def run(
        self,
        *,
        t1_path: str | Path | None,
        t2_path: str | Path | None,
        query: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        t1, t2 = validate_temporal_change_input(
            t1_path=t1_path,
            t2_path=t2_path,
            temporal_order=str(metadata.get("temporal_order", "")),
            pair_id=metadata.get("pair_id"),
            split=str(metadata.get("split", "external_inference")),
        )
        started = perf_counter()
        caption = self._captioner(t1.path, t2.path)
        elapsed = perf_counter() - started
        if not isinstance(caption, str) or not caption.strip():
            raise RuntimeError("CHG2CAP_EMPTY_CAPTION")
        return {
            "status": "COMPLETED",
            "route": ROUTE,
            "task": ROUTE,
            "change_description": caption.strip(),
            "selected_specialist": SPECIALIST,
            "model_tool": SPECIALIST,
            "temporal_order": TEMPORAL_ORDER,
            "pair_id": metadata.get("pair_id"),
            "pair_id": metadata.get("pair_id"),
            "query": str(query or ""),
            "t1_identity": {"filename": t1.identity, "sha256": t1.sha256, "dimensions": [t1.width, t1.height]},
            "t2_identity": {"filename": t2.identity, "sha256": t2.sha256, "dimensions": [t2.width, t2.height]},
            "provenance": {"source_dataset": metadata.get("source_dataset", "external"), "model": {**getattr(self._captioner, "provenance", {}), **self._model_provenance}},
            "warnings": ["Change description is generated language; no change mask, bounding box, or area estimate is produced."],
            "error_code": None,
            "runtime_seconds": elapsed,
        }


def run_temporal_change_description(**kwargs: Any) -> dict[str, Any]:
    """Public wrapper used by the API and direct integrations."""
    controller = kwargs.pop("controller", None) or TemporalChangeDescriptionController()
    return controller.run(**kwargs)
