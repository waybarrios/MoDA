#!/bin/bash
# =============================================================================
# MoDA + Qwen3-VL-2B Full SFT Training Script
# =============================================================================
# Usage: bash scripts/train.sh [NUM_GPUS]

set -e

# ---- Configuration (edit these) ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-}"
NUM_GPUS=${1:-1}

# ---- Activate environment ----
if [ -n "${CONDA_ENV}" ]; then
    export PATH="${CONDA_ENV}/bin:$PATH"
fi
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

cd "${PROJECT_DIR}"

# ---- Step 1: Prepare model directory (if not already done) ----
if [ ! -f "qwen3_vl_2b_moda/config.json" ]; then
    echo ">>> Preparing model directory..."
    python scripts/prepare_model_dir.py --model-size 2b
fi

# ---- Step 2: Convert dataset (if not already done) ----
if [ ! -f "data/llava_v1_5_mix665k_llamafactory.json" ]; then
    echo ">>> Converting dataset to LlamaFactory format..."
    python data/convert_llava_to_llamafactory.py \
        --input "${LLAVA_JSON:-data/llava_v1_5_mix665k.json}" \
        --output data/llava_v1_5_mix665k_llamafactory.json \
        --image_base_dir "${IMAGE_BASE_DIR:-data/images}"
fi

# ---- Step 3: Train ----
echo ">>> Starting training with ${NUM_GPUS} GPU(s)..."
if [ "${NUM_GPUS}" -gt 1 ]; then
    FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NUM_GPUS} \
    llamafactory-cli train configs/qwen3vl_moda_full_sft.yaml
else
    llamafactory-cli train configs/qwen3vl_moda_full_sft.yaml
fi

echo ">>> Training complete!"
