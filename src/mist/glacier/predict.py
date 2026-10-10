"""Run GLACIER inference with the public ms-pred package and release checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mist.data.simulation_utils import normalize_glacier_smiles

MS_PRED_REVISION = "708148c2a8eb0c32120644436fefd2fe90cd2214"


def load_glacier(checkpoint_path: Path, device: str = "cpu"):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not {"state_dict", "hyper_parameters"} <= checkpoint.keys():
        raise ValueError("expected a GLACIER Lightning checkpoint")
    config = checkpoint["hyper_parameters"]
    state = checkpoint["state_dict"]
    if config.get("encoder_type", "graphormer") != "graphormer":
        raise ValueError("only Graphormer GLACIER checkpoints are supported")
    try:
        if "inten_group_embed.weight" in state or config.get(
            "inten_broken_bond_feats", False
        ):
            from mist.glacier.model import CheckpointGLACIER

            model = CheckpointGLACIER(**config)
        else:
            from ms_pred.glacier.joint_model import JointModel

            model = JointModel(**config)
    except ImportError as error:
        raise ImportError(
            "GLACIER requires the optional ms-pred environment; see docs/glacier.md"
        ) from error
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def predict_spectrum(
    model, smiles: str, energy: float, adduct: str, instrument: str, device: str
):
    smiles = normalize_glacier_smiles(smiles)
    if adduct not in {"[M+H]+", "[M+Na]+"}:
        raise ValueError("supported adducts are [M+H]+ and [M+Na]+")
    if instrument not in {"Orbitrap", "QTOF"}:
        raise ValueError("supported instruments are Orbitrap and QTOF")
    if not np.isfinite(energy) or energy < 0:
        raise ValueError("collision energy must be a nonnegative finite value in eV")
    with torch.inference_mode():
        result = model.predict_mol(smiles, energy, adduct, device, instrument)
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument(
        "--energy", type=float, default=20.0, help="Collision energy in eV"
    )
    parser.add_argument("--adduct", default="[M+H]+")
    parser.add_argument("--instrument", default="Orbitrap")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load_glacier(args.checkpoint, args.device)
    result = predict_spectrum(
        model, args.smiles, args.energy, args.adduct, args.instrument, args.device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **result)
    print(json.dumps({"output": str(args.output), "rows": len(result["spec"])}))


if __name__ == "__main__":
    main()
