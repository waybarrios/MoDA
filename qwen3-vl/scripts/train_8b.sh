#!/bin/bash
# =============================================================================
# MoDA + Qwen3-VL-8B Full SFT Training Script
# =============================================================================
# Usage: bash scripts/train_8b.sh [NUM_GPUS] [ZERO_STAGE]
#
# Arguments:
#   NUM_GPUS    - Number of GPUs (default: 4)
#   ZERO_STAGE  - DeepSpeed ZeRO stage: 2 or 3 (default: 2)
#
# Examples:
#   bash scripts/train_8b.sh 4 2    # 4 GPUs, ZeRO-2
#   bash scripts/train_8b.sh 4 3    # 4 GPUs, ZeRO-3
#   bash scripts/train_8b.sh 8 3    # 8 GPUs, ZeRO-3

set -e

# ---- Configuration (edit these) ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-}"
NUM_GPUS=${1:-4}
ZERO_STAGE=${2:-2}

# Select config based on ZeRO stage
if [ "${ZERO_STAGE}" = "3" ]; then
    CONFIG="configs/qwen3vl_8b_moda_full_sft_z3.yaml"
    echo ">>> Using ZeRO-3 config (parameter sharding across GPUs)"
elif [ "${ZERO_STAGE}" = "2" ]; then
    CONFIG="configs/qwen3vl_8b_moda_full_sft_z2.yaml"
    echo ">>> Using ZeRO-2 config (optimizer + gradient sharding)"
else
    echo "ERROR: ZERO_STAGE must be 2 or 3, got: ${ZERO_STAGE}"
    exit 1
fi

# ---- Activate environment ----
if [ -n "${CONDA_ENV}" ]; then
    export PATH="${CONDA_ENV}/bin:$PATH"
fi
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "${PROJECT_DIR}"

# ---- Step 1: Prepare model directory (if not already done) ----
if [ ! -f "qwen3_vl_8b_moda/config.json" ]; then
    echo ">>> Preparing 8B model directory..."
    python scripts/prepare_model_dir.py --model-size 8b --num-layers 2
fi

# ---- Step 2: Convert dataset (if not already done) ----
if [ ! -f "data/llava_v1_5_mix665k_llamafactory.json" ]; then
    echo ">>> Converting dataset to LlamaFactory format..."
    python data/convert_llava_to_llamafactory.py \
        --input "${LLAVA_JSON:-data/llava_v1_5_mix665k.json}" \
        --output data/llava_v1_5_mix665k_llamafactory.json \
        --image_base_dir "${IMAGE_BASE_DIR:-data/images}"
fi

# ---- Step 3: Train with DeepSpeed ----
echo ">>> Starting 8B training with ${NUM_GPUS} GPU(s), ZeRO-${ZERO_STAGE}..."
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NUM_GPUS} \
llamafactory-cli train "${CONFIG}"

echo ">>> Training complete!"
