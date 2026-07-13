#!/bin/bash
# Launch base 8B and MoDA 8B evaluations in parallel on separate GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Optional: load OpenAI API key for MMVet
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY detected — will include mmvet benchmark"
fi

echo "Starting parallel 8B evaluation..."
echo "  Base 8B model -> GPU 4"
echo "  MoDA 8B model -> GPU 5"
echo ""

# Run both in parallel
bash "$SCRIPT_DIR/eval_base_8b.sh" 4 &
PID_BASE=$!

bash "$SCRIPT_DIR/eval_moda_8b.sh" 5 &
PID_MODA=$!

echo "Base PID: $PID_BASE"
echo "MoDA PID: $PID_MODA"
echo ""
echo "Waiting for both to complete..."

wait $PID_BASE
echo "Base 8B model evaluation finished."

wait $PID_MODA
echo "MoDA 8B model evaluation finished."

echo ""
echo "=== All 8B evaluations complete ==="
echo "Run: python scripts/compare_results.py --base eval_results/base_8b --moda eval_results/moda_8b"
