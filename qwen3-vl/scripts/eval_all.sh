#!/bin/bash
# Launch base and MoDA evaluations in parallel on separate GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Optional: load OpenAI API key for MMVet
if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY detected — will include mmvet benchmark"
fi

echo "Starting parallel evaluation..."
echo "  Base model -> GPU 4"
echo "  MoDA model -> GPU 5"
echo ""

# Run both in parallel
bash "$SCRIPT_DIR/eval_base.sh" 4 &
PID_BASE=$!

bash "$SCRIPT_DIR/eval_moda.sh" 5 &
PID_MODA=$!

echo "Base PID: $PID_BASE"
echo "MoDA PID: $PID_MODA"
echo ""
echo "Waiting for both to complete..."

wait $PID_BASE
echo "Base model evaluation finished."

wait $PID_MODA
echo "MoDA model evaluation finished."

echo ""
echo "=== All evaluations complete ==="
echo "Run: python scripts/compare_results.py"
