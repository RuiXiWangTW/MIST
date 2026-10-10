import hashlib
from io import BytesIO

import pytest

from mist.download import artifact_catalog, download_artifact


def test_all_pending_artifacts_fail_before_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("placeholder must not make a network request")
    monkeypatch.setattr("mist.download.urlopen", forbidden)
    for artifact in artifact_catalog()["artifacts"]:
        with pytest.raises(ValueError, match="pending"):
            download_artifact(artifact, tmp_path)


def test_download_checksum_and_reuse(tmp_path, monkeypatch):
    payload = b"verified model bytes"
    record = {"filename": "model.ckpt", "url": "https://zenodo.org/records/example/files/model.ckpt",
              "sha256": hashlib.sha256(payload).hexdigest()}
    catalog = {"artifacts": {"model": record}}
    monkeypatch.setattr("mist.download.urlopen", lambda *a, **k: BytesIO(payload))
    path = download_artifact("model", tmp_path, catalog)
    assert path.read_bytes() == payload
    assert download_artifact("model", tmp_path, catalog) == path
    path.unlink()
    monkeypatch.setattr("mist.download.urlopen", lambda *a, **k: BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_artifact("model", tmp_path, catalog)
    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
