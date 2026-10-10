"""Download a published release artifact and verify its SHA-256 checksum."""

from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen


def artifact_catalog() -> dict:
    return json.loads(files("mist").joinpath("artifacts.json").read_text())


def download_artifact(name: str, output_dir: Path, catalog: dict | None = None) -> Path:
    catalog = artifact_catalog() if catalog is None else catalog
    record = catalog["artifacts"][name]
    url, checksum, filename = (
        record.get("url"),
        record.get("sha256"),
        record["filename"],
    )
    if not url or not checksum:
        raise ValueError(
            f"{name}: Zenodo upload is pending; URL and SHA-256 are not yet available"
        )
    if urlparse(url).scheme != "https":
        raise ValueError("artifact URLs must use HTTPS")
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("expected a lowercase SHA-256 checksum")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("invalid artifact filename")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    if target.exists():
        with target.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() == checksum:
            return target
        raise FileExistsError(f"existing artifact has a different checksum: {target}")
    fd, name_on_disk = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".partial", dir=output_dir
    )
    temporary = Path(name_on_disk)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as handle, urlopen(url, timeout=60) as response:
            for chunk in iter(lambda: response.read(8 << 20), b""):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"SHA-256 mismatch for {name}")
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("downloads"))
    args = parser.parse_args()
    catalog = artifact_catalog()
    if args.list:
        for name, record in catalog["artifacts"].items():
            status = (
                "available"
                if record.get("url") and record.get("sha256")
                else "pending Zenodo upload"
            )
            print(f"{name}: {record['description']} ({status})")
        return
    if args.artifact not in catalog["artifacts"]:
        parser.error("choose an artifact shown by --list")
    try:
        print(download_artifact(args.artifact, args.output_dir, catalog))
    except (ValueError, FileExistsError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
