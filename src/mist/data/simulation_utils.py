"""Shared manifest and molecule utilities for synthetic spectrum caches."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from rdkit import Chem

class SimulationPlanError(ValueError):
    """Invalid synthetic simulation plan."""

HEAVY_ELEMENTS = ("C", "N", "P", "O", "S", "Si", "I", "Cl", "F", "Br", "B", "Se", "Fe", "Co", "As", "Na", "K")

MODEL_ATOMIC_NUMBERS = {
    Chem.GetPeriodicTable().GetAtomicNumber(element)
    for element in ("H", *HEAVY_ELEMENTS)
}

def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()

def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()

def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)

def normalize_glacier_smiles(smiles: str) -> str:
    """Apply the exact structural filtering/canonicalization used for prediction."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SimulationPlanError("rdkit_parse_failed")
    if any(atom.GetAtomicNum() not in MODEL_ATOMIC_NUMBERS for atom in molecule.GetAtoms()):
        raise SimulationPlanError("unsupported_atom")
    molecule = Chem.RemoveHs(molecule)
    if molecule.GetNumAtoms() > 100:
        raise SimulationPlanError("more_than_100_heavy_atoms")
    try:
        return Chem.MolToSmiles(molecule)
    except Exception as error:
        raise SimulationPlanError("rdkit_canonicalization_failed") from error


def load_plan_manifest(output_root: Path) -> dict:
    path = output_root.resolve() / "plan" / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("kind") != "glacier_fixed_pretraining_plan":
        raise SimulationPlanError(f"expected a fixed pretraining simulation plan: {path}")
    if int(manifest.get("schema_version", -1)) != 1:
        raise SimulationPlanError(f"unsupported plan schema: {path}")
    return manifest
