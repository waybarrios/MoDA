"""Rewrite legacy 'masking.*' parameter keys to 'moda.*' in a saved checkpoint.

Usage: python scripts/convert_legacy_moda_checkpoint.py /path/to/checkpoint_dir
Handles *.safetensors shards, pytorch_model*.bin shards, and the
model.safetensors.index.json / pytorch_model.bin.index.json weight maps.
"""
import argparse
import glob
import json
import os

import torch


def _remap(sd):
    return {k.replace("masking.", "moda."): v for k, v in sd.items()}


def convert(path):
    changed = False
    try:
        from safetensors.torch import load_file, save_file
        for f in glob.glob(os.path.join(path, "*.safetensors")):
            sd = load_file(f)
            if any("masking." in k for k in sd):
                save_file(_remap(sd), f, metadata={"format": "pt"})
                print(f"converted {f}")
                changed = True
    except ImportError:
        print("safetensors not installed; skipping *.safetensors")

    for f in glob.glob(os.path.join(path, "pytorch_model*.bin")):
        sd = torch.load(f, map_location="cpu")
        if any("masking." in k for k in sd):
            torch.save(_remap(sd), f)
            print(f"converted {f}")
            changed = True

    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        f = os.path.join(path, name)
        if os.path.exists(f):
            with open(f) as fh:
                index = json.load(fh)
            wm = index.get("weight_map", {})
            if any("masking." in k for k in wm):
                index["weight_map"] = {k.replace("masking.", "moda."): v for k, v in wm.items()}
                with open(f, "w") as fh:
                    json.dump(index, fh, indent=2)
                print(f"converted {f}")
                changed = True

    if not changed:
        print("no legacy 'masking.*' keys found; nothing to do")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_dir")
    convert(p.parse_args().checkpoint_dir)
