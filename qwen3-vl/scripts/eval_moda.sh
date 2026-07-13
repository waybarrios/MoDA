#!/bin/bash
# Evaluate Qwen3-VL-2B + MoDA on all benchmarks
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES=${1:-5}

MODEL="${PROJECT_DIR}/checkpoints/qwen3vl_moda_full_sft"
OUTPUT_DIR="${PROJECT_DIR}/eval_results/moda"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=== MoDA model evaluation on GPU $CUDA_VISIBLE_DEVICES ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

# Non-GPT benchmarks
for TASK in gqa scienceqa realworldqa chartqa mmstar; do
    echo ""
    echo ">>> Running $TASK ..."
    python -m lmms_eval \
        --model qwen3_vl_moda \
        --model_args pretrained=$MODEL \
        --tasks $TASK \
        --batch_size 1 \
        --output_path "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/${TASK}.log"
    echo ">>> $TASK done."
done

# MMVet (requires OpenAI API)
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo ""
    echo ">>> Running mmvet (GPT-eval) ..."
    python -m lmms_eval \
        --model qwen3_vl_moda \
        --model_args pretrained=$MODEL \
        --tasks mmvet \
        --batch_size 1 \
        --output_path "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/mmvet.log"
    echo ">>> mmvet done."
else
    echo ""
    echo ">>> Skipping mmvet (OPENAI_API_KEY not set)"
fi

echo ""
echo "=== MoDA model evaluation complete ==="
