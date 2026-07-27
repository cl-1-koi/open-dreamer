#!/usr/bin/env python3
"""Evaluate OpenDreamer pixels on immutable paired CoinRun V2 anchors.

Expected anchor-manifest semantics (field aliases listed below are accepted):

* top level: ``schema``, ``manifest_sha256``, ``context`` (32),
  ``max_horizon`` (32), ``evaluation_horizons`` ([1, 8, 32]), ``corpora``,
  and ``anchors``;
* anchor identity: ``id``/``anchor_id``, source identity
  (``corpus_source`` or ``source``), level, trajectory, episode, absolute
  ``source_row_start``, trajectory-relative ``context_start``, and optionally
  the equivalent prediction/action row starts;
* transition contract: 32 ``future_actions`` and a 32-element
  ``terminal_reset_valid_mask``;
* integrity: hashes for the action sequence, context-plus-target RGB, target
  RGB, and the aligned OpenDreamer episode payload/RGB/actions/rewards.

The canonical manifest emitted by
``world-model-coinrun/scripts/build_coinrun_pixel_ontology_anchors.py`` uses
the first name in each description. Alternate names are accepted only when
they resolve to the same required semantic value; conflicting aliases fail.

The evaluator never samples a dataloader. It scans the manifest-declared
ArrayRecord shards in deterministic order and identifies an episode by
source+level identity. It encodes only the 32 true context frames. A single
H=32 dynamics scan uses the model's 32-row ring KV cache, so generated states
replace old context states as the window advances. Predictions feed back in
latent space and no future RGB is encoded or teacher-forced. H=1 and H=8 are
metrics over prefixes of that single H=32 rollout.

Predicted latents are exported as direct, unnormalized tokenizer bottleneck
values converted to float32. This is an exact representation of bfloat16 or
float16 outputs and avoids decode/re-encode. NPY/NPZ files are authoritative;
this script does not create metric-bearing MP4 files.

Example:
    uv run python scripts/eval_coinrun_paired_anchors.py \
      --anchor-manifest /path/to/coinrun_pixel_ontology_anchors.json \
      --array-record-root /path/to/coinrun_paired_v2_open_dreamer \
      --dynamics-ckpt /path/to/dynamics/checkpoints \
      --out-dir logs/eval_coinrun_paired
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import tempfile
import time
from typing import Any

import numpy as np


ANCHOR_SCHEMA = "coinrun-pixel-ontology-anchor-manifest-v1"
RESULT_SCHEMA = "open-dreamer-coinrun-paired-pixel-eval-v1"
CONTEXT = 32
MAX_HORIZON = 32
EVALUATION_HORIZONS = (1, 8, 32)
NUM_ACTIONS = 9
NOOP_ACTION = 4
ACTION_PERMUTATION = tuple((action + 5) % NUM_ACTIONS for action in range(NUM_ACTIONS))
PSNR_CAP_DB = 99.0
SHA256_HEX = frozenset("0123456789abcdef")


class PairedAnchorError(RuntimeError):
    """The immutable benchmark, episode alignment, or rollout is invalid."""


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    source: str
    source_label: str
    level_id: int
    trajectory_index: int
    episode_index: int
    episode_id: str
    trajectory_source_row_start: int
    source_row_start: int
    context_start: int
    prediction_start: int
    action_start: int
    future_actions: np.ndarray
    valid_mask: np.ndarray
    future_actions_sha256: str
    rgb_sequence_sha256: str
    rgb_target_sha256: str
    episode_record_sha256: str
    episode_frame_sha256: str
    episode_action_sha256: str
    episode_reward_sha256: str
    future_action_prefix_sha256: Mapping[str, str]
    rgb_target_prefix_sha256: Mapping[str, str]
    raw: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, int]:
        return self.source, self.level_id


@dataclass(frozen=True)
class Manifest:
    path: Path
    file_sha256: str
    manifest_sha256: str
    benchmark_seed: int
    anchors: tuple[Anchor, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class LocatedEpisode:
    record_index: int
    payload: bytes
    record: Mapping[str, Any]


@dataclass(frozen=True)
class AnchorWindow:
    context_rgb: np.ndarray
    target_rgb: np.ndarray
    context_actions: np.ndarray
    future_actions: np.ndarray
    rewards: np.ndarray
    first_targets: np.ndarray
    done: np.ndarray
    terminal_cause: np.ndarray
    next_state_valid: np.ndarray
    target_timestep: np.ndarray
    episode_provenance: Mapping[str, Any]


StepFunction = Callable[
    [np.ndarray, np.ndarray, int, int, int],
    np.ndarray,
]


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return encoded + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise PairedAnchorError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedAnchorError(f"{label} must be a JSON object")
    return value


def _semantic(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    label: str,
    *,
    required: bool = True,
) -> Any:
    found = [(name, mapping[name]) for name in names if name in mapping]
    if not found:
        if required:
            raise PairedAnchorError(
                f"{label} is missing; accepted field names are {list(names)}"
            )
        return None
    first_name, first_value = found[0]
    first_encoded = _canonical_json_bytes(first_value)
    for name, value in found[1:]:
        if _canonical_json_bytes(value) != first_encoded:
            raise PairedAnchorError(
                f"{label} has conflicting aliases {first_name!r} and {name!r}"
            )
    return first_value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PairedAnchorError(f"{label} must be an integer >= {minimum}")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PairedAnchorError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise PairedAnchorError(f"{label} must be a lowercase SHA256 digest")
    return value


def _normalize_source(value: Any, label: str) -> str:
    source = _require_text(value, label)
    aliases = {
        "random": "random",
        "scripted": "scripted_forward_v0",
        "scripted_forward_v0": "scripted_forward_v0",
    }
    if source not in aliases:
        raise PairedAnchorError(
            f"{label} must identify random or scripted_forward_v0; got {source!r}"
        )
    return aliases[source]


def _hash_semantic(
    anchor: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> str:
    hashes_value = anchor.get("hashes", {})
    hashes = _require_mapping(hashes_value, f"{label}.hashes") if hashes_value else {}
    values: list[tuple[str, Any]] = []
    for name in names:
        if name in anchor:
            values.append((name, anchor[name]))
        if name in hashes:
            values.append((f"hashes.{name}", hashes[name]))
    if not values:
        raise PairedAnchorError(
            f"{label} is missing; accepted hash field names are {list(names)}"
        )
    expected = _require_sha256(values[0][1], f"{label}.{values[0][0]}")
    for name, value in values[1:]:
        actual = _require_sha256(value, f"{label}.{name}")
        if actual != expected:
            raise PairedAnchorError(f"{label} has conflicting hash aliases")
    return expected


def _optional_prefix_hashes(
    anchor: Mapping[str, Any],
    names: Sequence[str],
    label: str,
) -> Mapping[str, str]:
    value = _semantic(anchor, names, label, required=False)
    if value is None:
        return {}
    mapping = _require_mapping(value, label)
    result: dict[str, str] = {}
    for horizon in EVALUATION_HORIZONS:
        key = str(horizon)
        if key not in mapping:
            raise PairedAnchorError(f"{label} is missing horizon {horizon}")
        result[key] = _require_sha256(mapping[key], f"{label}[{key!r}]")
    return result


def parse_anchor(
    value: Any,
    *,
    index: int,
    manifest_context: int = CONTEXT,
    manifest_horizon: int = MAX_HORIZON,
) -> Anchor:
    """Normalize one anchor while rejecting ambiguous aliases and bad semantics."""

    raw = _require_mapping(value, f"anchors[{index}]")
    label = f"anchors[{index}]"
    anchor_id = _require_sha256(
        _semantic(raw, ("id", "anchor_id"), f"{label} identity"),
        f"{label}.id",
    )

    source_label_value = _semantic(
        raw,
        ("source", "source_label"),
        f"{label} source label",
    )
    source_label = _require_text(source_label_value, f"{label}.source")
    corpus_source_value = _semantic(
        raw,
        ("corpus_source", "source_identity", "collector"),
        f"{label} corpus source",
        required=False,
    )
    source = _normalize_source(
        corpus_source_value if corpus_source_value is not None else source_label,
        f"{label} source identity",
    )
    if _normalize_source(source_label, f"{label}.source") != source:
        raise PairedAnchorError(f"{label} source label and corpus source disagree")

    level_id = _require_int(
        _semantic(raw, ("level_id", "level", "level_seed"), f"{label} level"),
        f"{label}.level_id",
    )
    trajectory_index = _require_int(
        _semantic(
            raw,
            ("trajectory_index", "trajectory", "trajectory_id"),
            f"{label} trajectory",
        ),
        f"{label}.trajectory_index",
    )
    episode_index = _require_int(
        _semantic(raw, ("episode_index", "episode"), f"{label} episode"),
        f"{label}.episode_index",
    )
    episode_id = _require_text(
        _semantic(raw, ("episode_id", "open_dreamer_episode_id"), f"{label} episode id"),
        f"{label}.episode_id",
    )
    context_start = _require_int(
        _semantic(
            raw,
            ("context_start", "context_row_start", "trajectory_context_start"),
            f"{label} context start",
        ),
        f"{label}.context_start",
    )
    source_row_start = _require_int(
        _semantic(
            raw,
            ("source_row_start", "sequence_source_row_start", "row_start"),
            f"{label} absolute source row start",
        ),
        f"{label}.source_row_start",
    )
    trajectory_row_value = _semantic(
        raw,
        ("trajectory_source_row_start", "trajectory_row_start"),
        f"{label} trajectory source row start",
        required=False,
    )
    trajectory_source_row_start = (
        source_row_start - context_start
        if trajectory_row_value is None
        else _require_int(
            trajectory_row_value,
            f"{label}.trajectory_source_row_start",
        )
    )
    if trajectory_source_row_start + context_start != source_row_start:
        raise PairedAnchorError(
            f"{label} absolute source row and trajectory-relative context disagree"
        )

    context_length_value = _semantic(
        raw,
        ("context_length", "context"),
        f"{label} context length",
        required=False,
    )
    context_length = (
        manifest_context
        if context_length_value is None
        else _require_int(context_length_value, f"{label}.context_length", minimum=1)
    )
    if context_length != manifest_context or context_length != CONTEXT:
        raise PairedAnchorError(
            f"{label} context length must be exactly {CONTEXT}; got {context_length}"
        )

    prediction_value = _semantic(
        raw,
        ("prediction_start", "target_start"),
        f"{label} prediction start",
        required=False,
    )
    prediction_start = (
        context_start + context_length
        if prediction_value is None
        else _require_int(prediction_value, f"{label}.prediction_start")
    )
    if prediction_start != context_start + context_length:
        raise PairedAnchorError(
            f"{label} prediction_start must equal context_start + context_length"
        )

    action_value = _semantic(
        raw,
        ("action_start", "future_action_start"),
        f"{label} action start",
        required=False,
    )
    action_start = (
        prediction_start - 1
        if action_value is None
        else _require_int(action_value, f"{label}.action_start")
    )
    if action_start != prediction_start - 1:
        raise PairedAnchorError(
            f"{label} action_start must be one row before prediction_start"
        )

    anchor_horizon_value = _semantic(
        raw,
        ("max_horizon", "horizon"),
        f"{label} maximum horizon",
        required=False,
    )
    anchor_horizon = (
        manifest_horizon
        if anchor_horizon_value is None
        else _require_int(anchor_horizon_value, f"{label}.max_horizon", minimum=1)
    )
    if anchor_horizon != manifest_horizon or anchor_horizon != MAX_HORIZON:
        raise PairedAnchorError(
            f"{label} max horizon must be exactly {MAX_HORIZON}; got {anchor_horizon}"
        )

    actions_value = _semantic(
        raw,
        ("future_actions", "action_sequence", "actions"),
        f"{label} future actions",
    )
    if not isinstance(actions_value, list) or len(actions_value) != MAX_HORIZON:
        raise PairedAnchorError(
            f"{label} future actions must be a {MAX_HORIZON}-element JSON list"
        )
    if any(type(action) is not int for action in actions_value):
        raise PairedAnchorError(f"{label} future actions must contain integers")
    future_actions = np.asarray(actions_value, dtype=np.int8)
    if np.any((future_actions < 0) | (future_actions >= NUM_ACTIONS)):
        raise PairedAnchorError(
            f"{label} future actions must be in [0, {NUM_ACTIONS})"
        )

    mask_value = _semantic(
        raw,
        (
            "terminal_reset_valid_mask",
            "validity_mask",
            "valid_mask",
            "transition_valid_mask",
        ),
        f"{label} terminal/reset validity mask",
    )
    if not isinstance(mask_value, list) or len(mask_value) != MAX_HORIZON:
        raise PairedAnchorError(
            f"{label} validity mask must be a {MAX_HORIZON}-element JSON list"
        )
    if any(type(valid) is not bool for valid in mask_value):
        raise PairedAnchorError(f"{label} validity mask must contain booleans")
    valid_mask = np.asarray(mask_value, dtype=np.bool_)
    if not bool(valid_mask.all()):
        raise PairedAnchorError(
            f"{label} crosses a reset/terminal boundary; paired evaluation fails closed"
        )

    return Anchor(
        anchor_id=anchor_id,
        source=source,
        source_label=source_label,
        level_id=level_id,
        trajectory_index=trajectory_index,
        episode_index=episode_index,
        episode_id=episode_id,
        trajectory_source_row_start=trajectory_source_row_start,
        source_row_start=source_row_start,
        context_start=context_start,
        prediction_start=prediction_start,
        action_start=action_start,
        future_actions=future_actions,
        valid_mask=valid_mask,
        future_actions_sha256=_hash_semantic(
            raw,
            ("future_actions_sha256", "action_sequence_sha256"),
            f"{label} future-action hash",
        ),
        rgb_sequence_sha256=_hash_semantic(
            raw,
            ("rgb_sequence_sha256", "sequence_rgb_sha256"),
            f"{label} RGB sequence hash",
        ),
        rgb_target_sha256=_hash_semantic(
            raw,
            ("rgb_target_sha256", "target_rgb_sha256"),
            f"{label} RGB target hash",
        ),
        episode_record_sha256=_hash_semantic(
            raw,
            (
                "open_dreamer_episode_record_sha256",
                "episode_record_sha256",
                "record_sha256",
            ),
            f"{label} OpenDreamer record hash",
        ),
        episode_frame_sha256=_hash_semantic(
            raw,
            (
                "open_dreamer_episode_frame_sha256",
                "episode_frame_sha256",
                "episode_rgb_sha256",
            ),
            f"{label} OpenDreamer episode RGB hash",
        ),
        episode_action_sha256=_hash_semantic(
            raw,
            (
                "open_dreamer_episode_action_sha256",
                "episode_action_sha256",
            ),
            f"{label} OpenDreamer episode action hash",
        ),
        episode_reward_sha256=_hash_semantic(
            raw,
            (
                "open_dreamer_episode_reward_sha256",
                "episode_reward_sha256",
            ),
            f"{label} OpenDreamer episode reward hash",
        ),
        future_action_prefix_sha256=_optional_prefix_hashes(
            raw,
            ("future_action_prefix_sha256", "action_prefix_sha256"),
            f"{label} future-action prefix hashes",
        ),
        rgb_target_prefix_sha256=_optional_prefix_hashes(
            raw,
            ("rgb_target_prefix_sha256", "target_rgb_prefix_sha256"),
            f"{label} RGB target prefix hashes",
        ),
        raw=raw,
    )


def load_anchor_manifest(path: Path | str) -> Manifest:
    """Load and cryptographically validate the immutable anchor manifest."""

    manifest_path = Path(path).resolve()
    try:
        encoded = manifest_path.read_bytes()
        raw_value = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairedAnchorError(
            f"could not read anchor manifest {manifest_path}: {error}"
        ) from error
    raw = _require_mapping(raw_value, "anchor manifest")
    schema = _require_text(
        _semantic(raw, ("schema", "schema_version"), "manifest schema"),
        "manifest schema",
    )
    if schema != ANCHOR_SCHEMA:
        raise PairedAnchorError(
            f"manifest schema must be {ANCHOR_SCHEMA!r}; got {schema!r}"
        )

    recorded_digest = _require_sha256(
        _semantic(
            raw,
            ("manifest_sha256", "canonical_manifest_sha256"),
            "canonical manifest digest",
        ),
        "manifest_sha256",
    )
    digest_payload = dict(raw)
    digest_payload.pop("manifest_sha256", None)
    digest_payload.pop("canonical_manifest_sha256", None)
    actual_digest = _sha256_bytes(_canonical_json_bytes(digest_payload))
    if actual_digest != recorded_digest:
        raise PairedAnchorError(
            "anchor manifest canonical SHA256 mismatch; refusing a mutable or "
            f"corrupted manifest ({actual_digest} != {recorded_digest})"
        )

    context = _require_int(
        _semantic(raw, ("context", "context_length"), "manifest context"),
        "manifest context",
        minimum=1,
    )
    horizon = _require_int(
        _semantic(raw, ("max_horizon", "horizon"), "manifest max horizon"),
        "manifest max_horizon",
        minimum=1,
    )
    horizons = _semantic(
        raw,
        ("evaluation_horizons", "prefix_horizons", "horizons"),
        "manifest evaluation horizons",
    )
    if horizons != list(EVALUATION_HORIZONS):
        raise PairedAnchorError(
            f"manifest evaluation horizons must be {list(EVALUATION_HORIZONS)}"
        )
    if context != CONTEXT or horizon != MAX_HORIZON:
        raise PairedAnchorError(
            f"paired benchmark requires context={CONTEXT}, max_horizon={MAX_HORIZON}; "
            f"got context={context}, max_horizon={horizon}"
        )

    anchor_values = _semantic(
        raw,
        ("anchors", "anchor_records"),
        "manifest anchors",
    )
    if not isinstance(anchor_values, list) or not anchor_values:
        raise PairedAnchorError("manifest anchors must be a non-empty JSON array")
    anchors = tuple(
        parse_anchor(
            value,
            index=index,
            manifest_context=context,
            manifest_horizon=horizon,
        )
        for index, value in enumerate(anchor_values)
    )
    ids = [anchor.anchor_id for anchor in anchors]
    if len(ids) != len(set(ids)):
        raise PairedAnchorError("manifest anchor IDs must be unique")

    selection = _require_mapping(
        _semantic(raw, ("selection",), "manifest selection"),
        "manifest selection",
    )
    seed = _require_int(
        _semantic(selection, ("seed", "benchmark_seed"), "manifest selection seed"),
        "manifest selection seed",
    )
    selected_count = _semantic(
        selection,
        ("selected_anchor_count", "anchor_count"),
        "manifest selected anchor count",
        required=False,
    )
    if selected_count is not None and _require_int(
        selected_count, "manifest selected anchor count", minimum=1
    ) != len(anchors):
        raise PairedAnchorError("manifest selected anchor count does not match anchors")

    return Manifest(
        path=manifest_path,
        file_sha256=_sha256_bytes(encoded),
        manifest_sha256=recorded_digest,
        benchmark_seed=seed,
        anchors=anchors,
        raw=raw,
    )


def _validate_open_dreamer_corpus(
    manifest: Manifest,
    array_record_root: Path,
) -> tuple[tuple[Path, ...], Mapping[str, Any]]:
    root = array_record_root.resolve()
    alignment = _require_mapping(
        _semantic(
            manifest.raw,
            ("open_dreamer_manifest", "array_record_manifest"),
            "OpenDreamer manifest provenance",
        ),
        "OpenDreamer manifest provenance",
    )
    expected_manifest_sha = _require_sha256(
        _semantic(alignment, ("sha256", "manifest_sha256"), "OpenDreamer manifest hash"),
        "OpenDreamer manifest hash",
    )
    filename = _require_text(
        _semantic(alignment, ("filename", "path"), "OpenDreamer manifest filename"),
        "OpenDreamer manifest filename",
    )
    open_dreamer_manifest_path = (root / filename).resolve()
    if not open_dreamer_manifest_path.is_relative_to(root):
        raise PairedAnchorError("OpenDreamer manifest path escapes array-record root")
    if not open_dreamer_manifest_path.is_file():
        raise PairedAnchorError(
            f"OpenDreamer corpus manifest is missing: {open_dreamer_manifest_path}"
        )
    if _sha256_file(open_dreamer_manifest_path) != expected_manifest_sha:
        raise PairedAnchorError("OpenDreamer corpus manifest SHA256 mismatch")
    try:
        open_dreamer_raw = json.loads(open_dreamer_manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairedAnchorError(
            f"could not parse OpenDreamer manifest: {error}"
        ) from error
    open_dreamer = _require_mapping(open_dreamer_raw, "OpenDreamer manifest")
    expected_action_space = {
        "type": "procgen_discrete",
        "categorical_action_dim": NUM_ACTIONS,
        "categorical_noop": NOOP_ACTION,
        "continuous_action_dim": 0,
        "num_binary_actions": 0,
    }
    if open_dreamer.get("action_space") != expected_action_space:
        raise PairedAnchorError(
            "OpenDreamer manifest action dimension/no-op contract is not "
            f"{NUM_ACTIONS} actions with no-op {NOOP_ACTION}"
        )

    source_values = open_dreamer.get("sources")
    if not isinstance(source_values, list):
        raise PairedAnchorError("OpenDreamer manifest sources must be a list")
    source_records: dict[str, Mapping[str, Any]] = {}
    for value in source_values:
        source_record = _require_mapping(value, "OpenDreamer source")
        if source_record.get("split") != "val":
            continue
        source = _normalize_source(source_record.get("source"), "OpenDreamer source")
        if source in source_records:
            raise PairedAnchorError(f"duplicate OpenDreamer val source {source!r}")
        source_records[source] = source_record

    corpora_value = _semantic(
        manifest.raw,
        ("corpora", "source_corpora"),
        "manifest corpora",
    )
    if not isinstance(corpora_value, list):
        raise PairedAnchorError("manifest corpora must be a JSON array")
    required_sources = {anchor.source for anchor in manifest.anchors}
    paths: list[Path] = []
    seen_sources: set[str] = set()
    for index, value in enumerate(corpora_value):
        corpus = _require_mapping(value, f"corpora[{index}]")
        source_label = _semantic(
            corpus,
            ("source", "source_label"),
            f"corpora[{index}] source label",
        )
        source_identity = _semantic(
            corpus,
            ("corpus_source", "source_identity"),
            f"corpora[{index}] source identity",
            required=False,
        )
        source = _normalize_source(
            source_identity if source_identity is not None else source_label,
            f"corpora[{index}] source identity",
        )
        if _normalize_source(source_label, f"corpora[{index}] source label") != source:
            raise PairedAnchorError(
                f"corpora[{index}] source label and source identity disagree"
            )
        if source not in required_sources:
            continue
        if source in seen_sources:
            raise PairedAnchorError(f"manifest has duplicate corpus source {source!r}")
        seen_sources.add(source)
        source_record = source_records.get(source)
        if source_record is None:
            raise PairedAnchorError(f"OpenDreamer manifest lacks val source {source!r}")
        expected_source_record_hash = _require_sha256(
            _semantic(
                corpus,
                (
                    "open_dreamer_source_record_sha256",
                    "source_record_sha256",
                ),
                f"corpora[{index}] OpenDreamer source-record hash",
            ),
            f"corpora[{index}] OpenDreamer source-record hash",
        )
        actual_source_record_hash = _sha256_bytes(
            _canonical_json_bytes(source_record)
        )
        if actual_source_record_hash != expected_source_record_hash:
            raise PairedAnchorError(
                f"OpenDreamer source metadata hash mismatch for {source!r}"
            )

        shard_values = _semantic(
            corpus,
            ("open_dreamer_output_shards", "output_shards", "array_record_shards"),
            f"corpora[{index}] ArrayRecord shards",
        )
        if not isinstance(shard_values, list) or not shard_values:
            raise PairedAnchorError(
                f"corpora[{index}] ArrayRecord shards must be a non-empty list"
            )
        for shard_index, shard_value in enumerate(shard_values):
            shard = _require_mapping(
                shard_value,
                f"corpora[{index}] shard[{shard_index}]",
            )
            relative = _require_text(
                _semantic(
                    shard,
                    ("path", "filename"),
                    f"corpora[{index}] shard[{shard_index}] path",
                ),
                f"corpora[{index}] shard[{shard_index}] path",
            )
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path.suffix != ".array_record":
                raise PairedAnchorError(f"invalid ArrayRecord path {relative!r}")
            expected_size = _require_int(
                _semantic(
                    shard,
                    ("size_bytes", "bytes"),
                    f"corpora[{index}] shard[{shard_index}] size",
                ),
                f"corpora[{index}] shard[{shard_index}] size",
                minimum=1,
            )
            expected_sha = _require_sha256(
                _semantic(
                    shard,
                    ("sha256", "file_sha256"),
                    f"corpora[{index}] shard[{shard_index}] hash",
                ),
                f"corpora[{index}] shard[{shard_index}] hash",
            )
            if not path.is_file():
                raise PairedAnchorError(f"manifest-declared ArrayRecord is missing: {path}")
            if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
                raise PairedAnchorError(
                    f"manifest-declared ArrayRecord size/hash mismatch: {path}"
                )
            paths.append(path)
    if seen_sources != required_sources:
        raise PairedAnchorError(
            f"manifest corpora do not cover anchor sources: "
            f"missing={sorted(required_sources - seen_sources)}"
        )
    if len(paths) != len(set(paths)):
        raise PairedAnchorError("manifest declares an ArrayRecord shard more than once")
    return tuple(sorted(paths)), open_dreamer


def _record_source_level(record: Mapping[str, Any]) -> tuple[str, int]:
    source_value = record.get(
        "source",
        record.get("collector_identity", record.get("collector")),
    )
    source = _normalize_source(source_value, "ArrayRecord source identity")
    level_value = record.get("level_id", record.get("level_seed"))
    level = _require_int(level_value, "ArrayRecord level identity")
    return source, level


def locate_anchor_episodes(
    paths: Sequence[Path],
    anchors: Sequence[Anchor],
) -> Mapping[tuple[str, int], LocatedEpisode]:
    """Scan exact ArrayRecords once and locate source+level episodes."""

    try:
        import grain
    except ImportError as error:
        raise PairedAnchorError("grain is required to read ArrayRecord episodes") from error

    wanted = {anchor.key for anchor in anchors}
    found: dict[tuple[str, int], LocatedEpisode] = {}
    source = grain.sources.ArrayRecordDataSource([str(path) for path in paths])
    for record_index in range(len(source)):
        payload = bytes(source[record_index])
        try:
            record_value = pickle.loads(payload)
        except Exception as error:
            raise PairedAnchorError(
                f"ArrayRecord record {record_index} is not a pickle episode: {error}"
            ) from error
        record = _require_mapping(
            record_value,
            f"ArrayRecord record {record_index}",
        )
        key = _record_source_level(record)
        if key not in wanted:
            continue
        if key in found:
            raise PairedAnchorError(
                f"source+level identity {key!r} matched multiple ArrayRecord episodes"
            )
        found[key] = LocatedEpisode(
            record_index=record_index,
            payload=payload,
            record=record,
        )
    missing = wanted - set(found)
    if missing:
        raise PairedAnchorError(
            f"could not locate exact ArrayRecord episodes for {sorted(missing)}"
        )
    return found


def _record_array(
    record: Mapping[str, Any],
    name: str,
    *,
    dtype: np.dtype[Any],
    length: int,
) -> np.ndarray:
    if name not in record:
        raise PairedAnchorError(f"ArrayRecord episode is missing {name!r}")
    value = np.asarray(record[name])
    if value.dtype != dtype or value.shape != (length,):
        raise PairedAnchorError(
            f"ArrayRecord {name!r} must have dtype={dtype}, shape={(length,)}; "
            f"got dtype={value.dtype}, shape={value.shape}"
        )
    return value


def _assert_hash(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise PairedAnchorError(
            f"{label} SHA256 mismatch ({actual} != {expected})"
        )


def verify_anchor_episode(
    anchor: Anchor,
    record: Mapping[str, Any],
    *,
    payload: bytes | None = None,
) -> AnchorWindow:
    """Verify exact episode/window identity and return immutable benchmark arrays."""

    source, level_id = _record_source_level(record)
    if (source, level_id) != anchor.key:
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} source+level mismatch: "
            f"{(source, level_id)!r} != {anchor.key!r}"
        )
    expected_scalars = {
        "trajectory_index": anchor.trajectory_index,
        "episode_index": anchor.episode_index,
        "episode_id": anchor.episode_id,
        "source_row_start": anchor.trajectory_source_row_start,
        "num_actions": NUM_ACTIONS,
        "categorical_noop": NOOP_ACTION,
    }
    for name, expected in expected_scalars.items():
        if record.get(name) != expected:
            raise PairedAnchorError(
                f"anchor {anchor.anchor_id} record {name!r} is "
                f"{record.get(name)!r}, expected {expected!r}"
            )
    if record.get("split") != "val":
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} matched non-validation record"
        )
    if record.get("action_alignment") != "action_applied_after_frame":
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} action alignment is incompatible"
        )
    if record.get("reward_alignment") != "reward_resulting_from_action":
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} reward alignment is incompatible"
        )

    length = _require_int(record.get("sequence_length"), "record sequence_length", minimum=1)
    frame_shape_value = np.asarray(record.get("frame_shape"))
    if frame_shape_value.shape != (3,) or frame_shape_value.dtype.kind not in "iu":
        raise PairedAnchorError("record frame_shape must be a 3-element integer array")
    frame_shape = tuple(int(value) for value in frame_shape_value)
    if frame_shape[2] != 3 or any(size <= 0 for size in frame_shape):
        raise PairedAnchorError(f"record frame_shape is invalid: {frame_shape}")
    raw_video = record.get("raw_video")
    if not isinstance(raw_video, bytes):
        raise PairedAnchorError("record raw_video must be bytes")
    expected_video_bytes = length * math.prod(frame_shape)
    if len(raw_video) != expected_video_bytes:
        raise PairedAnchorError(
            f"record raw_video has {len(raw_video)} bytes, expected {expected_video_bytes}"
        )
    frames = np.frombuffer(raw_video, dtype=np.uint8).reshape(
        (length, *frame_shape)
    )
    actions = _record_array(
        record,
        "actions",
        dtype=np.dtype(np.int8),
        length=length,
    )
    rewards = _record_array(
        record,
        "rewards",
        dtype=np.dtype(np.float32),
        length=length,
    )
    first = _record_array(
        record,
        "episode_starts",
        dtype=np.dtype(np.bool_),
        length=length,
    )
    done = _record_array(
        record,
        "episode_ends",
        dtype=np.dtype(np.bool_),
        length=length,
    )
    terminal_cause = _record_array(
        record,
        "terminal_cause",
        dtype=np.dtype(np.uint8),
        length=length,
    )
    timestep = _record_array(
        record,
        "timestep",
        dtype=np.dtype(np.int32),
        length=length,
    )
    next_state_valid = _record_array(
        record,
        "next_state_valid",
        dtype=np.dtype(np.bool_),
        length=length,
    )
    if not np.isfinite(rewards).all():
        raise PairedAnchorError("record rewards contain nonfinite values")
    if np.any((actions < 0) | (actions >= NUM_ACTIONS)):
        raise PairedAnchorError("record contains actions outside the 9-action contract")

    if payload is None:
        raise PairedAnchorError(
            "raw ArrayRecord payload is required to verify the episode record hash"
        )
    _assert_hash(
        _sha256_bytes(payload),
        anchor.episode_record_sha256,
        f"anchor {anchor.anchor_id} episode payload",
    )
    episode_hashes = {
        "RGB": (sha256_array(frames), anchor.episode_frame_sha256),
        "actions": (sha256_array(actions), anchor.episode_action_sha256),
        "rewards": (sha256_array(rewards), anchor.episode_reward_sha256),
    }
    for name, (actual, expected) in episode_hashes.items():
        _assert_hash(actual, expected, f"anchor {anchor.anchor_id} episode {name}")
    record_hash_fields = {
        "paired_frame_sha256": episode_hashes["RGB"][0],
        "action_sha256": episode_hashes["actions"][0],
        "reward_sha256": episode_hashes["rewards"][0],
    }
    for name, expected in record_hash_fields.items():
        if record.get(name) != expected:
            raise PairedAnchorError(
                f"anchor {anchor.anchor_id} record's own {name} is invalid"
            )

    context_start = anchor.context_start
    prediction_start = anchor.prediction_start
    sequence_stop = prediction_start + MAX_HORIZON
    action_stop = anchor.action_start + MAX_HORIZON
    if sequence_stop > length or action_stop > length:
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} window exceeds episode length {length}"
        )
    expected_absolute_starts = {
        "source_row_start": anchor.trajectory_source_row_start + context_start,
        "prediction_source_row_start": (
            anchor.trajectory_source_row_start + prediction_start
        ),
        "action_source_row_start": (
            anchor.trajectory_source_row_start + anchor.action_start
        ),
    }
    for name, expected in expected_absolute_starts.items():
        actual = anchor.source_row_start if name == "source_row_start" else anchor.raw.get(name)
        if actual is not None and actual != expected:
            raise PairedAnchorError(
                f"anchor {anchor.anchor_id} {name} is {actual!r}, expected {expected}"
            )

    context_rgb = np.ascontiguousarray(frames[context_start:prediction_start])
    target_rgb = np.ascontiguousarray(frames[prediction_start:sequence_stop])
    sequence_rgb = np.ascontiguousarray(frames[context_start:sequence_stop])
    future_actions = np.ascontiguousarray(
        actions[anchor.action_start:action_stop]
    )
    _assert_hash(
        sha256_array(sequence_rgb),
        anchor.rgb_sequence_sha256,
        f"anchor {anchor.anchor_id} RGB sequence",
    )
    _assert_hash(
        sha256_array(target_rgb),
        anchor.rgb_target_sha256,
        f"anchor {anchor.anchor_id} RGB target",
    )
    _assert_hash(
        sha256_array(future_actions),
        anchor.future_actions_sha256,
        f"anchor {anchor.anchor_id} future actions",
    )
    if not np.array_equal(future_actions, anchor.future_actions):
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} future action list does not match the episode"
        )
    for horizon in EVALUATION_HORIZONS:
        key = str(horizon)
        if anchor.future_action_prefix_sha256:
            _assert_hash(
                sha256_array(future_actions[:horizon]),
                anchor.future_action_prefix_sha256[key],
                f"anchor {anchor.anchor_id} H{horizon} action prefix",
            )
        if anchor.rgb_target_prefix_sha256:
            _assert_hash(
                sha256_array(target_rgb[:horizon]),
                anchor.rgb_target_prefix_sha256[key],
                f"anchor {anchor.anchor_id} H{horizon} RGB prefix",
            )

    action_rows = slice(anchor.action_start, action_stop)
    target_rows = slice(prediction_start, sequence_stop)
    computed_valid = (
        next_state_valid[action_rows]
        & ~first[target_rows]
        & ~done[action_rows]
        & (terminal_cause[action_rows] == 0)
    )
    if not np.array_equal(computed_valid, anchor.valid_mask) or not bool(
        computed_valid.all()
    ):
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} crosses a reset/terminal boundary"
        )

    shifted_actions = np.empty_like(actions)
    shifted_actions[0] = NOOP_ACTION
    shifted_actions[1:] = actions[:-1]
    context_actions = np.ascontiguousarray(
        shifted_actions[context_start:prediction_start]
    )
    if context_rgb.shape[0] != CONTEXT or context_actions.shape != (CONTEXT,):
        raise PairedAnchorError(
            f"anchor {anchor.anchor_id} did not produce exactly {CONTEXT} context rows"
        )

    provenance_names = (
        "adapter_schema_version",
        "source_schema_version",
        "source_contract",
        "source_shard",
        "source_shard_sha256",
        "split",
        "source",
        "collector_identity",
        "episode_id",
        "episode_index",
        "trajectory_index",
        "level_id",
        "source_row_start",
        "source_row_end",
        "action_alignment",
        "reward_alignment",
    )
    return AnchorWindow(
        context_rgb=context_rgb.copy(),
        target_rgb=target_rgb.copy(),
        context_actions=context_actions.copy(),
        future_actions=future_actions.copy(),
        rewards=np.ascontiguousarray(rewards[action_rows]).copy(),
        first_targets=np.ascontiguousarray(first[target_rows]).copy(),
        done=np.ascontiguousarray(done[action_rows]).copy(),
        terminal_cause=np.ascontiguousarray(terminal_cause[action_rows]).copy(),
        next_state_valid=np.ascontiguousarray(next_state_valid[action_rows]).copy(),
        target_timestep=np.ascontiguousarray(timestep[target_rows]).copy(),
        episode_provenance={
            name: record[name] for name in provenance_names if name in record
        },
    )


def derive_step_seed(
    benchmark_seed: int,
    anchor_id: str,
    stream: str,
    step: int,
) -> int:
    """Derive a stable 64-bit per-anchor/per-step random seed."""

    payload = _canonical_json_bytes(
        {
            "schema": RESULT_SCHEMA,
            "benchmark_seed": benchmark_seed,
            "anchor_id": anchor_id,
            "stream": stream,
            "step": step,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sliding_window_rollout(
    initial_values: np.ndarray,
    context_actions: np.ndarray,
    future_actions: np.ndarray,
    *,
    max_context: int,
    benchmark_seed: int,
    anchor_id: str,
    step_fn: StepFunction,
    rng_stream: str = "dynamics",
) -> np.ndarray:
    """Generate one autoregressive rollout using only the latest legal window.

    ``step_fn`` receives read-only copies of the current value window, aligned
    action window, the next action, the step index, and a deterministic seed.
    It never receives future ground truth. Its prediction is appended to the
    history and is therefore visible to every subsequent legal window.
    """

    values = np.asarray(initial_values)
    actions = np.asarray(context_actions)
    future = np.asarray(future_actions)
    if max_context < 1:
        raise ValueError("max_context must be positive")
    if values.ndim < 1 or values.shape[0] != max_context:
        raise ValueError(
            f"initial_values must have exactly max_context={max_context} rows; "
            f"got shape {values.shape}"
        )
    if actions.shape != (max_context,):
        raise ValueError(
            f"context_actions must have shape {(max_context,)}, got {actions.shape}"
        )
    if future.ndim != 1 or future.size < 1:
        raise ValueError("future_actions must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise PairedAnchorError("initial rollout values contain nonfinite outputs")

    value_history = np.array(values, copy=True)
    action_history = np.array(actions, copy=True)
    predictions: list[np.ndarray] = []
    expected_shape = values.shape[1:]
    for step, action_value in enumerate(future):
        value_window = np.array(value_history[-max_context:], copy=True)
        action_window = np.array(action_history[-max_context:], copy=True)
        if value_window.shape[0] > max_context or action_window.shape[0] > max_context:
            raise AssertionError("sliding window exceeded max_context")
        value_window.setflags(write=False)
        action_window.setflags(write=False)
        step_seed = derive_step_seed(
            benchmark_seed,
            anchor_id,
            rng_stream,
            step,
        )
        prediction = np.asarray(
            step_fn(
                value_window,
                action_window,
                int(action_value),
                step,
                step_seed,
            )
        )
        if prediction.shape != expected_shape:
            raise PairedAnchorError(
                f"rollout step {step} returned shape {prediction.shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(prediction).all():
            raise PairedAnchorError(
                f"rollout step {step} returned nonfinite model output"
            )
        prediction = np.array(prediction, copy=True)
        predictions.append(prediction)
        value_history = np.concatenate(
            (value_history, prediction[None]),
            axis=0,
        )
        action_history = np.concatenate(
            (action_history, np.asarray([action_value], dtype=actions.dtype)),
            axis=0,
        )
    return np.stack(predictions, axis=0)


def rollout_prefixes(
    rollout: np.ndarray,
    horizons: Sequence[int] = EVALUATION_HORIZONS,
) -> Mapping[int, np.ndarray]:
    """Return H-prefix copies from one rollout; no horizon is regenerated."""

    value = np.asarray(rollout)
    prefixes: dict[int, np.ndarray] = {}
    for horizon in horizons:
        if type(horizon) is not int or not 1 <= horizon <= value.shape[0]:
            raise ValueError(
                f"prefix horizon {horizon!r} is outside rollout length {value.shape[0]}"
            )
        prefixes[horizon] = np.array(value[:horizon], copy=True)
    return prefixes


def permute_actions(actions: np.ndarray) -> np.ndarray:
    value = np.asarray(actions)
    if value.ndim != 1 or np.any((value < 0) | (value >= NUM_ACTIONS)):
        raise PairedAnchorError("cannot permute actions outside the 9-action contract")
    lut = np.asarray(ACTION_PERMUTATION, dtype=np.int8)
    return np.ascontiguousarray(lut[value])


def psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return PSNR_CAP_DB
    return float(
        min(
            20.0 * math.log10(255.0) - 10.0 * math.log10(mse),
            PSNR_CAP_DB,
        )
    )


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise PairedAnchorError(
            f"pixel metric shape mismatch: {pred.shape} != {truth.shape}"
        )
    error = pred - truth
    result = float(np.mean(error * error))
    if not math.isfinite(result):
        raise PairedAnchorError("pixel metric is nonfinite")
    return result


def _skill(model_mse: float, persistence_mse: float) -> tuple[float | None, str]:
    if persistence_mse == 0.0:
        return None, "undefined_zero_persistence_error"
    value = 1.0 - model_mse / persistence_mse
    if not math.isfinite(value):
        raise PairedAnchorError("normalized skill is nonfinite")
    return float(value), "defined"


def compute_anchor_metrics(
    target_rgb: np.ndarray,
    predicted_rgb: np.ndarray,
    action_permuted_rgb: np.ndarray,
    last_context_rgb: np.ndarray,
) -> Mapping[str, Any]:
    target = np.asarray(target_rgb)
    predicted = np.asarray(predicted_rgb)
    permuted = np.asarray(action_permuted_rgb)
    if target.shape[0] != MAX_HORIZON:
        raise PairedAnchorError("target RGB does not have the required H=32 length")
    persistence = np.broadcast_to(last_context_rgb, target.shape)
    frame_model_mse = [
        _mse(predicted[index], target[index]) for index in range(MAX_HORIZON)
    ]
    frame_persistence_mse = [
        _mse(persistence[index], target[index]) for index in range(MAX_HORIZON)
    ]
    frame_permuted_mse = [
        _mse(permuted[index], target[index]) for index in range(MAX_HORIZON)
    ]
    horizon_metrics: dict[str, Any] = {}
    for horizon in EVALUATION_HORIZONS:
        model_mse = _mse(predicted[:horizon], target[:horizon])
        persistence_mse = _mse(persistence[:horizon], target[:horizon])
        permuted_mse = _mse(permuted[:horizon], target[:horizon])
        skill, skill_status = _skill(model_mse, persistence_mse)
        horizon_metrics[str(horizon)] = {
            "pixel_mse": model_mse,
            "psnr_db": psnr_from_mse(model_mse),
            "persistence_pixel_mse": persistence_mse,
            "persistence_psnr_db": psnr_from_mse(persistence_mse),
            "normalized_skill": skill,
            "normalized_skill_status": skill_status,
            "action_permuted_pixel_mse": permuted_mse,
            "action_permuted_psnr_db": psnr_from_mse(permuted_mse),
            "action_sensitivity_mse_delta": permuted_mse - model_mse,
        }
    return {
        "horizons": horizon_metrics,
        "per_frame": {
            "pixel_mse": frame_model_mse,
            "psnr_db": [psnr_from_mse(value) for value in frame_model_mse],
            "persistence_pixel_mse": frame_persistence_mse,
            "persistence_psnr_db": [
                psnr_from_mse(value) for value in frame_persistence_mse
            ],
            "action_permuted_pixel_mse": frame_permuted_mse,
            "action_permuted_psnr_db": [
                psnr_from_mse(value) for value in frame_permuted_mse
            ],
            "action_sensitivity_mse_delta": [
                permuted - model
                for permuted, model in zip(
                    frame_permuted_mse,
                    frame_model_mse,
                    strict=True,
                )
            ],
        },
    }


def _aggregate_horizons(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for horizon in EVALUATION_HORIZONS:
        key = str(horizon)
        model_mse = float(np.mean([row["horizons"][key]["pixel_mse"] for row in rows]))
        persistence_mse = float(
            np.mean([row["horizons"][key]["persistence_pixel_mse"] for row in rows])
        )
        permuted_mse = float(
            np.mean(
                [row["horizons"][key]["action_permuted_pixel_mse"] for row in rows]
            )
        )
        skill, status = _skill(model_mse, persistence_mse)
        result[key] = {
            "anchor_count": len(rows),
            "pixel_mse": model_mse,
            "psnr_db": psnr_from_mse(model_mse),
            "persistence_pixel_mse": persistence_mse,
            "persistence_psnr_db": psnr_from_mse(persistence_mse),
            "normalized_skill": skill,
            "normalized_skill_status": status,
            "action_permuted_pixel_mse": permuted_mse,
            "action_permuted_psnr_db": psnr_from_mse(permuted_mse),
            "action_sensitivity_mse_delta": permuted_mse - model_mse,
        }
    return result


def aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not rows:
        raise PairedAnchorError("cannot aggregate zero anchor metrics")
    by_source: dict[str, Any] = {}
    for source in ("random", "scripted_forward_v0"):
        selected = [row["metrics"] for row in rows if row["source"] == source]
        if selected:
            by_source[source] = _aggregate_horizons(selected)

    level_rows: list[Mapping[str, Any]] = []
    level_keys = sorted({(row["source"], row["level_id"]) for row in rows})
    for source, level_id in level_keys:
        selected = [
            row["metrics"]
            for row in rows
            if row["source"] == source and row["level_id"] == level_id
        ]
        aggregated = _aggregate_horizons(selected)
        level_rows.append(
            {
                "source": source,
                "level_id": level_id,
                "horizons": aggregated,
            }
        )

    level_macro: dict[str, Any] = {}
    for horizon in EVALUATION_HORIZONS:
        key = str(horizon)
        model_mse = float(
            np.mean([row["horizons"][key]["pixel_mse"] for row in level_rows])
        )
        persistence_mse = float(
            np.mean(
                [row["horizons"][key]["persistence_pixel_mse"] for row in level_rows]
            )
        )
        permuted_mse = float(
            np.mean(
                [
                    row["horizons"][key]["action_permuted_pixel_mse"]
                    for row in level_rows
                ]
            )
        )
        skill, status = _skill(model_mse, persistence_mse)
        level_macro[key] = {
            "level_count": len(level_rows),
            "pixel_mse": model_mse,
            "psnr_db": psnr_from_mse(model_mse),
            "persistence_pixel_mse": persistence_mse,
            "persistence_psnr_db": psnr_from_mse(persistence_mse),
            "normalized_skill": skill,
            "normalized_skill_status": status,
            "action_permuted_pixel_mse": permuted_mse,
            "action_permuted_psnr_db": psnr_from_mse(permuted_mse),
            "action_sensitivity_mse_delta": permuted_mse - model_mse,
        }
    return {
        "anchor_macro": {
            "all": _aggregate_horizons([row["metrics"] for row in rows]),
            "by_source": by_source,
        },
        "level_macro": {
            "all": level_macro,
            "levels": level_rows,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value, newline=True))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parameter_count(model: Any) -> int:
    if not hasattr(model, "num_scaling_params"):
        raise PairedAnchorError(
            f"loaded model {type(model).__name__} cannot report parameter count"
        )
    return int(model.num_scaling_params())


def _memory_metrics(jax: Any) -> Mapping[str, Any]:
    host_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    devices: list[Mapping[str, Any]] = []
    for device in jax.devices():
        stats = device.memory_stats()
        if stats is None:
            devices.append({"device": str(device), "available": False})
            continue
        peak = stats.get("peak_bytes_in_use")
        devices.append(
            {
                "device": str(device),
                "available": peak is not None,
                "peak_bytes_in_use": None if peak is None else int(peak),
            }
        )
    return {
        "host_peak_rss_bytes": host_peak,
        "accelerators": devices,
    }


def _validate_model_contract(
    tokenizer: Any,
    dynamics: Any,
    first_window: AnchorWindow,
    validate_model_configs: Callable[..., Any],
) -> None:
    context_length = getattr(dynamics.cfg, "context_length", None)
    if context_length != CONTEXT:
        raise PairedAnchorError(
            f"dynamics checkpoint context_length must be exactly {CONTEXT}; "
            f"got {context_length!r}"
        )
    action_fields = {
        "categorical_action_dim": NUM_ACTIONS,
        "num_binary_actions": 0,
        "continuous_action_dim": 0,
    }
    for name, expected in action_fields.items():
        actual = int(getattr(dynamics.cfg, name))
        if actual != expected:
            raise PairedAnchorError(
                f"dynamics checkpoint {name} must be {expected}; got {actual}. "
                f"CoinRun no-op is fixed at index {NOOP_ACTION}"
            )
    frame_h = int(tokenizer.cfg.decoder.H)
    frame_w = int(tokenizer.cfg.decoder.W)
    if first_window.context_rgb.shape[1:] != (frame_h, frame_w, 3):
        raise PairedAnchorError(
            "tokenizer frame shape does not match paired native RGB: "
            f"{(frame_h, frame_w, 3)} != {first_window.context_rgb.shape[1:]}"
        )
    dataset_info = {
        "first_record_sequence_length": CONTEXT,
        "first_record_raw_video_nbytes": int(first_window.context_rgb.nbytes),
        "action_max": int(first_window.future_actions.max()),
    }
    validate_model_configs(dynamics.cfg, tokenizer.cfg, dataset_info)


@dataclass(frozen=True)
class Args:
    anchor_manifest: Path
    array_record_root: Path
    dynamics_ckpt: str
    tokenizer_ckpt: str | None
    out_dir: Path
    dynamics_model: str
    denoise_steps: int
    benchmark_seed: int | None
    anchor_shard_index: int
    anchor_shard_count: int
    overwrite: bool


def _parse_args(argv: Sequence[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--array-record-root", type=Path, required=True)
    parser.add_argument("--dynamics-ckpt", required=True)
    parser.add_argument("--tokenizer-ckpt")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/eval_coinrun_paired"))
    parser.add_argument(
        "--dynamics-model",
        choices=("dynamics", "dynamics_ema"),
        default="dynamics_ema",
    )
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--benchmark-seed", type=int)
    parser.add_argument("--anchor-shard-index", type=int, default=0)
    parser.add_argument("--anchor-shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.denoise_steps < 1:
        parser.error("--denoise-steps must be positive")
    if parsed.benchmark_seed is not None and parsed.benchmark_seed < 0:
        parser.error("--benchmark-seed must be non-negative")
    if parsed.anchor_shard_count < 1:
        parser.error("--anchor-shard-count must be positive")
    if not 0 <= parsed.anchor_shard_index < parsed.anchor_shard_count:
        parser.error(
            "--anchor-shard-index must be in "
            "[0, --anchor-shard-count)"
        )
    return Args(
        anchor_manifest=parsed.anchor_manifest,
        array_record_root=parsed.array_record_root,
        dynamics_ckpt=parsed.dynamics_ckpt,
        tokenizer_ckpt=parsed.tokenizer_ckpt,
        out_dir=parsed.out_dir,
        dynamics_model=parsed.dynamics_model,
        denoise_steps=parsed.denoise_steps,
        benchmark_seed=parsed.benchmark_seed,
        anchor_shard_index=parsed.anchor_shard_index,
        anchor_shard_count=parsed.anchor_shard_count,
        overwrite=parsed.overwrite,
    )


def run(args: Args) -> Path:
    """Run all immutable anchors and return the deterministic result path."""

    started = time.perf_counter()
    manifest = load_anchor_manifest(args.anchor_manifest)
    selected_anchors = manifest.anchors[
        args.anchor_shard_index :: args.anchor_shard_count
    ]
    manifest_anchor_indices = {
        anchor.anchor_id: index for index, anchor in enumerate(manifest.anchors)
    }
    if not selected_anchors:
        raise PairedAnchorError(
            "anchor shard is empty: "
            f"index={args.anchor_shard_index}, count={args.anchor_shard_count}, "
            f"anchors={len(manifest.anchors)}"
        )
    paths, open_dreamer_manifest = _validate_open_dreamer_corpus(
        manifest,
        args.array_record_root,
    )
    located = locate_anchor_episodes(paths, selected_anchors)
    windows = {
        anchor.anchor_id: verify_anchor_episode(
            anchor,
            located[anchor.key].record,
            payload=located[anchor.key].payload,
        )
        for anchor in selected_anchors
    }

    # Reuse the existing evaluator's checkpoint restoration and config gate.
    from scripts.eval_coinrun import (
        Args as BaseEvalArgs,
        load_models,
        validate_model_configs,
    )
    import jax
    import jax.numpy as jnp

    from dreamer.actions import Actions
    from dreamer.generation import DenoiseSchedule, latent_rollout
    from dreamer.parallel import build_parallel
    from dreamer.sampler import decode_jit, encode_jit

    mesh, _data_sharding, mesh_rules = build_parallel("data")
    load_started = time.perf_counter()
    with jax.set_mesh(mesh):
        tokenizer, dynamics, artifact_info = load_models(
            BaseEvalArgs(
                dynamics_ckpt=args.dynamics_ckpt,
                tokenizer_ckpt=args.tokenizer_ckpt,
                dynamics_model=args.dynamics_model,
            ),
            mesh_rules,
        )
    load_seconds = time.perf_counter() - load_started
    first_window = windows[selected_anchors[0].anchor_id]
    _validate_model_contract(
        tokenizer,
        dynamics,
        first_window,
        validate_model_configs,
    )
    k_max = int(dynamics.cfg.k_max)
    if args.denoise_steps > k_max or k_max % args.denoise_steps != 0:
        raise PairedAnchorError(
            f"denoise_steps={args.denoise_steps} must divide checkpoint k_max={k_max}"
        )
    schedule = DenoiseSchedule.init(args.denoise_steps, k_max)
    benchmark_seed = (
        manifest.benchmark_seed
        if args.benchmark_seed is None
        else args.benchmark_seed
    )
    parameter_counts = {
        "tokenizer": _parameter_count(tokenizer),
        "dynamics": _parameter_count(dynamics),
    }
    parameter_counts["total"] = (
        parameter_counts["tokenizer"] + parameter_counts["dynamics"]
    )

    run_identity = {
        "schema": RESULT_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_file_sha256": manifest.file_sha256,
        "dynamics_ckpt": artifact_info["dynamics_ckpt"],
        "dynamics_step": artifact_info["dynamics_step"],
        "dynamics_model": artifact_info["dynamics_model"],
        "tokenizer_ckpt": artifact_info["tokenizer_ckpt"],
        "tokenizer_step": artifact_info["tokenizer_step"],
        "denoise_steps": args.denoise_steps,
        "benchmark_seed": benchmark_seed,
        "anchor_shard_index": args.anchor_shard_index,
        "anchor_shard_count": args.anchor_shard_count,
    }
    run_digest = _sha256_bytes(_canonical_json_bytes(run_identity))
    output_root = args.out_dir.resolve()
    result_path = output_root / f"paired-pixel-{run_digest[:16]}"
    output_root.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        if not args.overwrite:
            raise PairedAnchorError(
                f"deterministic result directory already exists: {result_path}; "
                "pass --overwrite to replace this exact run"
            )
        shutil.rmtree(result_path)
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{result_path.name}.tmp-", dir=output_root)
    )

    def model_rollout(
        context_latents: np.ndarray,
        context_actions: np.ndarray,
        future_actions: np.ndarray,
        *,
        seed: int,
    ) -> np.ndarray:
        low = np.uint32(seed & 0xFFFFFFFF)
        high = np.uint32((seed >> 32) & 0xFFFFFFFF)
        rng = jax.random.fold_in(jax.random.PRNGKey(low), high)
        result = latent_rollout(
            dynamics,
            actions_future=Actions(
                categorical=jnp.asarray(future_actions[None], dtype=jnp.int32)
            ),
            schedule=schedule,
            # Dynamics projections run in bfloat16 for this checkpoint.
            # Matching that compute dtype keeps static KV caches portable
            # between Ampere and Hopper GPUs.
            latents_ctx=jnp.asarray(context_latents[None], dtype=jnp.bfloat16),
            actions_ctx=Actions(
                categorical=jnp.asarray(context_actions[None], dtype=jnp.int32)
            ),
            num_steps=MAX_HORIZON,
            rng=rng,
            use_kv_cache=True,
        )
        predicted = np.asarray(
            jax.device_get(result["latents"][0, CONTEXT:])
        )
        if predicted.shape[0] != MAX_HORIZON:
            raise PairedAnchorError(
                "dynamics rollout did not return exactly "
                f"{MAX_HORIZON} generated latent rows"
            )
        return predicted.astype(np.float32)

    metric_rows: list[Mapping[str, Any]] = []
    anchor_summaries: list[Mapping[str, Any]] = []
    try:
        anchors_dir = staging_path / "anchors"
        anchors_dir.mkdir()
        with jax.set_mesh(mesh):
            for shard_anchor_index, anchor in enumerate(selected_anchors):
                anchor_index = manifest_anchor_indices[anchor.anchor_id]
                anchor_started = time.perf_counter()
                window = windows[anchor.anchor_id]
                context_latents_device = encode_jit(
                    tokenizer,
                    jnp.asarray(window.context_rgb[None]),
                )
                context_latents = np.asarray(
                    jax.device_get(context_latents_device[0])
                )
                if context_latents.shape[0] != CONTEXT:
                    raise PairedAnchorError(
                        f"tokenizer returned {context_latents.shape[0]} context rows"
                    )
                if not np.isfinite(context_latents).all():
                    raise PairedAnchorError(
                        f"anchor {anchor.anchor_id} tokenizer emitted nonfinite latents"
                    )

                rollout_seed = derive_step_seed(
                    benchmark_seed,
                    anchor.anchor_id,
                    "dynamics",
                    0,
                )
                predicted_latents = model_rollout(
                    context_latents,
                    window.context_actions,
                    window.future_actions,
                    seed=rollout_seed,
                )
                permuted_actions = permute_actions(window.future_actions)
                permuted_latents = model_rollout(
                    context_latents,
                    window.context_actions,
                    permuted_actions,
                    seed=rollout_seed,
                )
                decoded = np.asarray(
                    jax.device_get(
                        decode_jit(tokenizer, jnp.asarray(predicted_latents[None]))
                    )
                )[0]
                decoded_permuted = np.asarray(
                    jax.device_get(
                        decode_jit(tokenizer, jnp.asarray(permuted_latents[None]))
                    )
                )[0]
                if not np.isfinite(decoded).all() or not np.isfinite(
                    decoded_permuted
                ).all():
                    raise PairedAnchorError(
                        f"anchor {anchor.anchor_id} decoder emitted nonfinite RGB"
                    )
                predicted_rgb = np.clip(decoded, 0, 255).astype(np.uint8)
                action_permuted_rgb = np.clip(
                    decoded_permuted, 0, 255
                ).astype(np.uint8)
                if predicted_rgb.shape != window.target_rgb.shape:
                    raise PairedAnchorError(
                        f"anchor {anchor.anchor_id} decoded RGB shape "
                        f"{predicted_rgb.shape} != target {window.target_rgb.shape}"
                    )

                metrics = compute_anchor_metrics(
                    window.target_rgb,
                    predicted_rgb,
                    action_permuted_rgb,
                    window.context_rgb[-1],
                )
                anchor_seconds = time.perf_counter() - anchor_started
                anchor_dir = anchors_dir / f"{anchor_index:04d}-{anchor.anchor_id}"
                anchor_dir.mkdir()
                np.save(anchor_dir / "context_rgb.npy", window.context_rgb)
                np.save(anchor_dir / "ground_truth_rgb.npy", window.target_rgb)
                np.save(anchor_dir / "predicted_rgb.npy", predicted_rgb)
                np.save(anchor_dir / "predicted_latents.npy", predicted_latents)
                np.save(anchor_dir / "actions.npy", window.future_actions)
                np.save(
                    anchor_dir / "action_permuted_rgb.npy",
                    action_permuted_rgb,
                )
                np.save(
                    anchor_dir / "action_permuted_latents.npy",
                    permuted_latents,
                )
                np.save(anchor_dir / "action_permuted_actions.npy", permuted_actions)
                np.savez(
                    anchor_dir / "transition_metadata.npz",
                    rewards=window.rewards,
                    first_targets=window.first_targets,
                    done=window.done,
                    terminal_cause=window.terminal_cause,
                    next_state_valid=window.next_state_valid,
                    terminal_reset_valid_mask=anchor.valid_mask,
                    target_timestep=window.target_timestep,
                )
                anchor_metrics = {
                    "schema": RESULT_SCHEMA,
                    "anchor_id": anchor.anchor_id,
                    "source": anchor.source,
                    "level_id": anchor.level_id,
                    "wall_time_seconds": anchor_seconds,
                    **metrics,
                }
                _write_json(anchor_dir / "metrics.json", anchor_metrics)
                per_anchor_provenance = {
                    "schema": RESULT_SCHEMA,
                    "anchor_index": anchor_index,
                    "anchor": anchor.raw,
                    "episode": window.episode_provenance,
                    "record_index": located[anchor.key].record_index,
                    "run_identity": run_identity,
                    "rng": {
                        "derivation": (
                            "SHA256(canonical JSON of schema, benchmark seed, "
                            "anchor id, stream='dynamics', and step=0), first "
                            "64 bits; model scan splits the root key per step"
                        ),
                        "benchmark_seed": benchmark_seed,
                        "true_and_action_permuted_use_identical_rollout_rng": True,
                    },
                    "rollout": {
                        "context_length": CONTEXT,
                        "max_horizon": MAX_HORIZON,
                        "dynamics_kv_cache_rows": CONTEXT,
                        "dynamics_cache_policy": (
                            "native ring cache retains the latest 32 rows"
                        ),
                        "teacher_forcing_after_prediction_start": False,
                        "future_truth_encoded": False,
                        "predicted_rgb_reencoded": False,
                        "prefix_horizons": list(EVALUATION_HORIZONS),
                    },
                    "latent_representation": {
                        "description": (
                            "direct unnormalized tokenizer bottleneck output; "
                            "float32 host representation; no RGB re-encoding"
                        ),
                        "dtype": "float32",
                        "shape": list(predicted_latents.shape),
                    },
                    "files": {
                        "context_rgb": "context_rgb.npy",
                        "ground_truth_rgb": "ground_truth_rgb.npy",
                        "predicted_rgb": "predicted_rgb.npy",
                        "predicted_latents": "predicted_latents.npy",
                        "actions": "actions.npy",
                        "action_permuted_rgb": "action_permuted_rgb.npy",
                        "action_permuted_latents": "action_permuted_latents.npy",
                        "action_permuted_actions": "action_permuted_actions.npy",
                        "transition_metadata": "transition_metadata.npz",
                        "metrics": "metrics.json",
                    },
                }
                _write_json(
                    anchor_dir / "provenance.json",
                    _jsonable(per_anchor_provenance),
                )
                row = {
                    "anchor_id": anchor.anchor_id,
                    "source": anchor.source,
                    "level_id": anchor.level_id,
                    "metrics": metrics,
                }
                metric_rows.append(row)
                anchor_summaries.append(
                    {
                        "anchor_id": anchor.anchor_id,
                        "source": anchor.source,
                        "level_id": anchor.level_id,
                        "path": str(anchor_dir.relative_to(staging_path)),
                        "wall_time_seconds": anchor_seconds,
                        "rng_seed_step_0": derive_step_seed(
                            benchmark_seed,
                            anchor.anchor_id,
                            "dynamics",
                            0,
                        ),
                    }
                )
                print(
                    f"shard anchor {shard_anchor_index + 1}/"
                    f"{len(selected_anchors)} (manifest index {anchor_index}) "
                    f"{anchor.source}/level-{anchor.level_id}: {anchor_seconds:.2f}s"
                )

        aggregate = aggregate_metrics(metric_rows)
        total_seconds = time.perf_counter() - started
        summary = {
            "schema": RESULT_SCHEMA,
            "run_id": run_digest,
            "result_directory": str(result_path),
            "benchmark": {
                "context": CONTEXT,
                "max_horizon": MAX_HORIZON,
                "evaluation_horizons": list(EVALUATION_HORIZONS),
                "prefix_semantics": (
                    "H1 and H8 are prefixes of the same H32 autoregressive rollout"
                ),
                "anchor_count": len(selected_anchors),
                "manifest_anchor_count": len(manifest.anchors),
                "anchor_shard_index": args.anchor_shard_index,
                "anchor_shard_count": args.anchor_shard_count,
                "benchmark_seed": benchmark_seed,
            },
            "metrics": aggregate,
            "parameter_counts": parameter_counts,
            "wall_time_seconds": {
                "checkpoint_load": load_seconds,
                "total": total_seconds,
            },
            "peak_memory": _memory_metrics(jax),
            "anchors": anchor_summaries,
            "provenance": {
                "run_identity": run_identity,
                "anchor_manifest": {
                    "path": str(manifest.path),
                    "manifest_sha256": manifest.manifest_sha256,
                    "file_sha256": manifest.file_sha256,
                },
                "array_record_root": str(args.array_record_root.resolve()),
                "array_record_files": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for path in paths
                ],
                "open_dreamer_manifest_schema": open_dreamer_manifest.get(
                    "adapter_schema_version"
                ),
                "checkpoints": artifact_info,
                "lossless_data_files_are_authoritative": True,
                "mp4_is_metric_bearing": False,
            },
        }
        _write_json(staging_path / "results.json", _jsonable(summary))
        os.replace(staging_path, result_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result_path = run(args)
    except PairedAnchorError as error:
        raise SystemExit(f"paired-anchor evaluation failed: {error}") from error
    print(json.dumps({"result_directory": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
