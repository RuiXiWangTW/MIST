import hashlib
from io import BytesIO

import pytest

from mist.download import download_artifact


@pytest.mark.parametrize("url,checksum", [
    (None, None),
    ("https://zenodo.org/records/example/files/model.ckpt", None),
    (None, "a" * 64),
])
def test_pending_artifact_fails_before_network(tmp_path, monkeypatch, url, checksum):
    def forbidden(*args, **kwargs):
        raise AssertionError("placeholder must not make a network request")
    monkeypatch.setattr("mist.download.urlopen", forbidden)
    catalog = {"artifacts": {"pending": {
        "filename": "model.ckpt", "url": url, "sha256": checksum,
    }}}
    with pytest.raises(ValueError, match="pending"):
        download_artifact("pending", tmp_path, catalog)


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
