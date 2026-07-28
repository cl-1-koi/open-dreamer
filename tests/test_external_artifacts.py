from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_fetch_module():
    path = REPO_ROOT / "scripts" / "fetch_external_artifacts.py"
    spec = importlib.util.spec_from_file_location("fetch_external_artifacts_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fetch = load_fetch_module()


def artifact_for(payload: bytes, path: str = "weights/model.bin") -> dict:
    return {
        "bytes": len(payload),
        "name": "model",
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "uri": "s3://private-bucket/model.bin",
    }


def test_cached_artifact_is_verified_without_fetch(tmp_path, monkeypatch):
    payload = b"content-addressed"
    artifact = artifact_for(payload)
    output = tmp_path / artifact["path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(payload)

    monkeypatch.setattr(
        fetch,
        "fetch_s3",
        lambda *_args, **_kwargs: pytest.fail("cached artifact was downloaded"),
    )

    assert fetch.fetch_artifact(
        artifact,
        root=tmp_path,
        profile=None,
        endpoint_url=None,
    ) == output


def test_cached_artifact_rejects_wrong_content(tmp_path):
    artifact = artifact_for(b"expected")
    output = tmp_path / artifact["path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="artifact verification failed"):
        fetch.fetch_artifact(
            artifact,
            root=tmp_path,
            profile=None,
            endpoint_url=None,
        )


def test_destination_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes destination root"):
        fetch.destination(tmp_path, "../outside.bin")
