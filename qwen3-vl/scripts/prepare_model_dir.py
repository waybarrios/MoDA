"""
Prepare a local model directory for MoDA-enabled VLMs that LlamaFactory can load
with trust_remote_code=True.

This script:
1. Copies config.json from the original model and adds auto_map + MoDA params
2. Symlinks all weight files from the original model
3. Copies our custom modeling code into the model directory

Supports Qwen3-VL (2B/4B/8B), Qwen2.5-VL (3B), InternVL3.5 (2B), and Granite Vision (2B).

Usage:
    python scripts/prepare_model_dir.py                       # default: 2b
    python scripts/prepare_model_dir.py --model-size 2b
    python scripts/prepare_model_dir.py --model-size 8b
    python scripts/prepare_model_dir.py --model-size qwen25vl-3b
    python scripts/prepare_model_dir.py --model-size internvl3-2b
    python scripts/prepare_model_dir.py --model-size granite-vision-2b
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# ---- Paths per model size ----
# HF_HOME is read from the environment (default: ~/.cache/huggingface)
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

MODEL_CONFIGS = {
    "2b": {
        "hf_model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "output_model_dir": "qwen3_vl_2b_moda",
        "moda_hidden_dim": 512,
        "moda_num_heads": 16,
        "auto_map": {
            "AutoConfig": "modeling_qwen3_vl_moda.Qwen3VLMoDAConfig",
            "AutoModel": "modeling_qwen3_vl_moda.Qwen3VLModelWithMoDA",
            "AutoModelForCausalLM": "modeling_qwen3_vl_moda.Qwen3VLForConditionalGenerationWithMoDA",
            "AutoModelForImageTextToText": "modeling_qwen3_vl_moda.Qwen3VLForConditionalGenerationWithMoDA",
        },
        "code_files": ["moda_adapter.py", "modeling_qwen3_vl_moda.py"],
    },
    "4b": {
        "hf_model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "output_model_dir": "qwen3_vl_4b_moda",
        "moda_hidden_dim": 640,
        "moda_num_heads": 32,
        "auto_map": {
            "AutoConfig": "modeling_qwen3_vl_moda.Qwen3VLMoDAConfig",
            "AutoModel": "modeling_qwen3_vl_moda.Qwen3VLModelWithMoDA",
            "AutoModelForCausalLM": "modeling_qwen3_vl_moda.Qwen3VLForConditionalGenerationWithMoDA",
            "AutoModelForImageTextToText": "modeling_qwen3_vl_moda.Qwen3VLForConditionalGenerationWithMoDA",
        },
        "code_files": ["moda_adapter.py", "modeling_qwen3_vl_moda.py"],
    },
    "8b": {
        "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "output_model_dir": "qwen3_vl_8b_moda",
        "moda_hidden_dim": 1024,
        "moda_num_heads": 32,
        "auto_map": {
            "AutoConfig": "modeling_qwen3_vl_moda.Qwen3VLMoDAConfig",
            "AutoModel": "modeling_qwen3_vl_moda.Qwen3VLModelWithMoDA",
            "AutoModelForCausalLM": "modeling_qwen3_vl_moda.Qwen3VLForConditionalGenerationWithMoDA",
            "AutoModelForImageTextToText": "modeling_qwen3_vl_moda.Qwen3VLForConditionalGenerationWithMoDA",
        },
        "code_files": ["moda_adapter.py", "modeling_qwen3_vl_moda.py"],
    },
    "internvl3-2b": {
        "hf_model_id": "OpenGVLab/InternVL3_5-2B-HF",
        "output_model_dir": "internvl3_2b_moda",
        "moda_hidden_dim": 512,
        "moda_num_heads": 16,
        "auto_map": {
            "AutoConfig": "modeling_internvl_moda.InternVLMoDAConfig",
            "AutoModel": "modeling_internvl_moda.InternVLModelWithMoDA",
            "AutoModelForCausalLM": "modeling_internvl_moda.InternVLForConditionalGenerationWithMoDA",
            "AutoModelForImageTextToText": "modeling_internvl_moda.InternVLForConditionalGenerationWithMoDA",
        },
        "code_files": ["moda_adapter.py", "modeling_internvl_moda.py"],
    },
    "granite-vision-2b": {
        "hf_model_id": "ibm-granite/granite-vision-3.2-2b",
        "output_model_dir": "granite_vision_moda",
        "moda_hidden_dim": 512,
        "moda_num_heads": 16,
        "auto_map": {
            "AutoConfig": "modeling_granite_vision_moda.GraniteVisionMoDAConfig",
            "AutoModel": "modeling_granite_vision_moda.LlavaNextModelWithMoDA",
            "AutoModelForCausalLM": "modeling_granite_vision_moda.LlavaNextForConditionalGenerationWithMoDA",
            "AutoModelForImageTextToText": "modeling_granite_vision_moda.LlavaNextForConditionalGenerationWithMoDA",
        },
        "code_files": ["moda_adapter.py", "modeling_granite_vision_moda.py"],
    },
    "qwen25vl-3b": {
        "hf_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "output_model_dir": "qwen25_vl_3b_moda",
        "moda_hidden_dim": 512,
        "moda_num_heads": 16,
        "auto_map": {
            "AutoConfig": "modeling_qwen25_vl_moda.Qwen25VLMoDAConfig",
            "AutoModel": "modeling_qwen25_vl_moda.Qwen25VLModelWithMoDA",
            "AutoModelForCausalLM": "modeling_qwen25_vl_moda.Qwen25VLForConditionalGenerationWithMoDA",
            "AutoModelForImageTextToText": "modeling_qwen25_vl_moda.Qwen25VLForConditionalGenerationWithMoDA",
        },
        "code_files": ["moda_adapter.py", "modeling_qwen25_vl_moda.py"],
    },
    "tinyllava-2b": {
        "hf_model_id": "tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B",
        "output_model_dir": "tinyllava_2b_moda",
        "moda_hidden_dim": 512,
        "moda_num_heads": 16,
        "auto_map": {
            "AutoConfig": "configuration_tinyllava.TinyLlavaMoDAConfig",
            "AutoModelForCausalLM": "modeling_tinyllava_moda.TinyLlavaForConditionalGenerationWithMoDA",
            "AutoProcessor": "processing_tinyllava.TinyLlavaProcessor",
        },
        "code_files": ["moda_adapter.py", "configuration_tinyllava.py", "modeling_tinyllava_moda.py", "processing_tinyllava.py"],
        "model_type_override": "tinyllava_moda",
        "add_image_token": True,
    },
}

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODA_CODE_DIR = PROJECT_DIR / "model"


def resolve_hf_snapshot(hf_model_id: str) -> Path:
    """Find the snapshot directory for a HF model in the local cache.

    Searches $HF_HOME/hub/models--<org>--<name>/snapshots/ for a directory
    containing config.json.
    """
    # Convert "Qwen/Qwen3-VL-2B-Instruct" -> "models--Qwen--Qwen3-VL-2B-Instruct"
    cache_dir_name = "models--" + hf_model_id.replace("/", "--")

    # Check both HF cache layouts: $HF_HOME/hub/models--... and $HF_HOME/models--...
    candidates_dirs = [
        HF_HOME / "hub" / cache_dir_name / "snapshots",
        HF_HOME / cache_dir_name / "snapshots",
    ]
    snapshots_dir = None
    for d in candidates_dirs:
        if d.exists():
            snapshots_dir = d
            break

    if snapshots_dir is None:
        raise FileNotFoundError(
            f"Model not found in HF cache. Searched:\n"
            f"  {candidates_dirs[0]}\n"
            f"  {candidates_dirs[1]}\n"
            f"Download it first:\n"
            f"  python -c \"from transformers import AutoModelForImageTextToText; "
            f"AutoModelForImageTextToText.from_pretrained('{hf_model_id}')\""
        )

    # Find snapshot with config.json (prefer most recent if multiple)
    candidates = [
        d for d in snapshots_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ]
    if not candidates:
        # Try all subdirectories
        candidates = [d for d in snapshots_dir.iterdir() if d.is_dir()]

    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    return max(candidates, key=lambda d: d.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Prepare MoDA-enabled VLM model directory")
    parser.add_argument(
        "--model-size", choices=["2b", "4b", "8b", "internvl3-2b", "granite-vision-2b", "qwen25vl-3b", "tinyllava-2b"], default="2b",
        help="Model variant: 2b/4b/8b (Qwen3-VL), qwen25vl-3b (Qwen2.5-VL), internvl3-2b (InternVL3.5), granite-vision-2b, or tinyllava-2b (default: 2b)"
    )
    parser.add_argument("--num-layers", type=int, default=2, help="Number of MoDA cross-attention layers (default: 2)")
    parser.add_argument("--sparsity-lambda", type=float, default=1e-3, help="Sparsity regularization weight (default: 1e-3)")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix to append to output directory name")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    original_model_dir = resolve_hf_snapshot(cfg["hf_model_id"])
    output_dir_name = cfg["output_model_dir"] + args.output_suffix
    output_model_dir = PROJECT_DIR / output_dir_name

    print(f"Model size: {args.model_size}")
    print(f"Source: {original_model_dir}")
    print(f"Output: {output_model_dir}")

    output_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modify config.json to add auto_map and MoDA parameters
    print("Modifying config.json...")
    with open(original_model_dir / "config.json") as f:
        config = json.load(f)

    # Add auto_map so transformers loads our custom classes
    config["auto_map"] = cfg["auto_map"]

    # Add MoDA-specific parameters (scaled per model size)
    config["moda_hidden_dim"] = cfg["moda_hidden_dim"]
    config["moda_num_heads"] = cfg["moda_num_heads"]
    config["moda_num_layers"] = args.num_layers
    config["moda_dropout"] = 0.1
    config["moda_sparsity_lambda"] = args.sparsity_lambda
    config["moda_enabled"] = True

    # Override model_type if needed (e.g., tinyllava -> tinyllava_moda)
    if "model_type_override" in cfg:
        config["model_type"] = cfg["model_type_override"]

    with open(output_model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 2. Symlink all other files from original model
    print("Creating symlinks to original model files...")
    for src_file in original_model_dir.iterdir():
        if src_file.name == "config.json":
            continue  # Already handled
        dst_file = output_model_dir / src_file.name
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        dst_file.symlink_to(src_file.resolve())
        print(f"  Linked: {src_file.name}")

    # 3. Copy only the relevant custom modeling code (not __init__.py or unrelated models)
    print("Copying MoDA modeling code...")
    for filename in cfg["code_files"]:
        src = MODA_CODE_DIR / filename
        dst = output_model_dir / filename
        shutil.copy2(src, dst)
        print(f"  Copied: {filename}")

    # 4. For TinyLLaVA: add <image> special token and save processor
    if cfg.get("add_image_token"):
        from transformers import AutoTokenizer, SiglipImageProcessor

        print("Adding <image> special token to tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(output_model_dir), trust_remote_code=True)
        if "<image>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens":
                tokenizer.special_tokens_map.get("additional_special_tokens", []) + ["<image>"]
            })
            tokenizer.save_pretrained(str(output_model_dir))

        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        print(f"  <image> token ID: {image_token_id}")

        # Update config with the correct image token index
        # NOTE: we do NOT change vocab_size — the model never embeds <image>
        # (it's replaced with 0 in _apply_moda, removed in prepare_inputs_labels)
        with open(output_model_dir / "config.json") as f:
            config = json.load(f)
        config["image_token_index"] = image_token_id
        with open(output_model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save SigLIP image processor for LlamaFactory
        print("Saving SigLIP image processor...")
        image_proc = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        image_proc.save_pretrained(str(output_model_dir))

    print(f"\nModel directory ready at: {output_model_dir}")
    print("You can now use this with LlamaFactory:")
    print(f"  model_name_or_path: {output_model_dir}")
    print("  trust_remote_code: true")


if __name__ == "__main__":
    main()
