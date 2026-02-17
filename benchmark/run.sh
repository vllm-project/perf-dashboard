#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Defaults (match InferenceMAX CI config) ─────────────────────────────────
MODEL="openai/gpt-oss-120b"
IMAGE="vllm/vllm-openai:v0.13.0"
TP=8
ISL=1024
OSL=1024
CONC=64
PORT=8888
OUTPUT_DIR="${SCRIPT_DIR}/results"
GPU_MEM_UTIL=0.9
RANDOM_RANGE_RATIO=1.0
HF_CACHE="${HF_HOME:-/dev/shm/.cache/huggingface}"

# ─── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run the GPT-OSS-120B FP4 vLLM benchmark on an H200 machine.

Options:
  --model MODEL           HuggingFace model name          (default: $MODEL)
  --image IMAGE           vLLM Docker image                (default: $IMAGE)
  --tp TP                 Tensor parallel size             (default: $TP)
  --isl ISL               Input sequence length            (default: $ISL)
  --osl OSL               Output sequence length           (default: $OSL)
  --conc CONC             Max concurrency                  (default: $CONC)
  --port PORT             Server port                      (default: $PORT)
  --output-dir DIR        Directory for results            (default: $OUTPUT_DIR)
  --gpu-mem-util FLOAT    GPU memory utilization           (default: $GPU_MEM_UTIL)
  --host-mode             Use host vLLM instead of Docker
  --random-range-ratio R  Random range ratio               (default: $RANDOM_RANGE_RATIO)
  --hf-cache DIR          HuggingFace cache directory      (default: $HF_CACHE)
  -h, --help              Show this help message

Examples:
  # Run with defaults (Docker mode, TP=8, ISL=1024, OSL=1024, CONC=64)
  ./run.sh

  # Run in host mode with custom parameters
  ./run.sh --host-mode --tp 4 --isl 8192 --osl 1024 --conc 32

  # Run a concurrency sweep
  for c in 4 8 16 32 64; do
    ./run.sh --conc \$c --output-dir results/sweep
  done
EOF
    exit 0
}

# ─── Parse CLI args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2";           shift 2 ;;
        --image)           IMAGE="$2";           shift 2 ;;
        --tp)              TP="$2";              shift 2 ;;
        --isl)             ISL="$2";             shift 2 ;;
        --osl)             OSL="$2";             shift 2 ;;
        --conc)            CONC="$2";            shift 2 ;;
        --port)            PORT="$2";            shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2";    shift 2 ;;
        --random-range-ratio) RANDOM_RANGE_RATIO="$2"; shift 2 ;;
        --hf-cache)        HF_CACHE="$2";        shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

# ─── Derived values ──────────────────────────────────────────────────────────
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    MAX_MODEL_LEN=$((ISL + OSL + 20))
elif [ "$ISL" -ge 8192 ] || [ "$OSL" -ge 8192 ]; then
    MAX_MODEL_LEN=$((ISL + OSL + 200))
else
    MAX_MODEL_LEN=$((ISL + OSL + 200))
fi

NUM_PROMPTS=$((CONC * 10))
NUM_WARMUPS=$((CONC * 2))
RESULT_FILENAME="gptoss_fp4_vllm_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo " GPT-OSS FP4 vLLM Benchmark"
echo "============================================="
echo " Model:          $MODEL"
echo " TP:             $TP"
echo " ISL:            $ISL"
echo " OSL:            $OSL"
echo " Concurrency:    $CONC"
echo " Max Model Len:  $MAX_MODEL_LEN"
echo " Num Prompts:    $NUM_PROMPTS"
echo " Num Warmups:    $NUM_WARMUPS"
echo " Port:           $PORT"
echo " Output Dir:     $OUTPUT_DIR"
echo "============================================="

# ─── Generate vLLM config.yaml ───────────────────────────────────────────────
CONFIG_FILE="${OUTPUT_DIR}/config.yaml"
cat > "$CONFIG_FILE" <<EOF
async-scheduling: true
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: $MAX_MODEL_LEN
EOF
echo "Generated config: $CONFIG_FILE"

# ─── Server log ──────────────────────────────────────────────────────────────
SERVER_LOG="${OUTPUT_DIR}/server.log"

# ─── Cleanup trap ────────────────────────────────────────────────────────────
SERVER_PID=""
CONTAINER_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "$CONTAINER_ID" ]]; then
        echo "Stopping Docker container $CONTAINER_ID..."
        docker stop "$CONTAINER_ID" 2>/dev/null || true
        docker rm "$CONTAINER_ID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ─── Launch vLLM server ─────────────────────────────────────────────────────
echo ""
echo "Launching vLLM server (host mode)..."
export VLLM_MXFP4_USE_MARLIN=1
# export TORCH_CUDA_ARCH_LIST="9.0"

PYTHONNOUSERSITE=1 vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --config "$CONFIG_FILE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$TP" \
    --max-num-seqs "$CONC" \
    --disable-log-requests > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "vLLM server started (PID $SERVER_PID)"

# ─── Wait for server health ─────────────────────────────────────────────────
echo ""
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600  # 10 minutes
INTERVAL=5
ELAPSED=0

# Tail the log in the background so user can see progress
tail -f -n +1 "$SERVER_LOG" 2>/dev/null &
TAIL_PID=$!

while true; do
    if curl --output /dev/null --silent --fail "http://0.0.0.0:${PORT}/health" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo ""
        echo "Server is ready!"
        break
    fi

    # Check if server process is still alive
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo "ERROR: vLLM server died before becoming healthy."
        echo "Check logs: $SERVER_LOG"
        exit 1
    fi

    ELAPSED=$((ELAPSED + INTERVAL))
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo "ERROR: Server did not become healthy within ${MAX_WAIT}s."
        exit 1
    fi

    sleep "$INTERVAL"
done

# ─── Run benchmark ──────────────────────────────────────────────────────────
echo ""
echo "Running benchmark..."
echo "  Num prompts: $NUM_PROMPTS"
echo "  Num warmups: $NUM_WARMUPS"
echo "  Max concurrency: $CONC"

python3 "${SCRIPT_DIR}/lib/benchmark_serving.py" \
    --model "$MODEL" \
    --backend vllm \
    --base-url "http://0.0.0.0:${PORT}" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --num-warmups "$NUM_WARMUPS" \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --result-dir "$OUTPUT_DIR" \
    --result-filename "${RESULT_FILENAME}.json"

BENCHMARK_EXIT=$?
if [[ $BENCHMARK_EXIT -ne 0 ]]; then
    echo "ERROR: Benchmark failed with exit code $BENCHMARK_EXIT"
    exit $BENCHMARK_EXIT
fi

# ─── Post-process results ───────────────────────────────────────────────────
echo ""
echo "Post-processing results..."
DATE="$(date +'%Y-%m-%d %H:%M:%S')"

python3 "${SCRIPT_DIR}/process_result.py" \
    --raw-result "${OUTPUT_DIR}/${RESULT_FILENAME}.json" \
    --output-dir "$OUTPUT_DIR" \
    --device h200 \
    --date "$DATE" \
    --tp "$TP" \
    --conc "$CONC" \
    --framework vllm \
    --precision fp4 \
    --model "$MODEL" \
    --image "$IMAGE" \
    --isl "$ISL" \
    --osl "$OSL"

echo ""
echo "============================================="
echo " Benchmark Complete"
echo "============================================="
echo " Raw result:  ${OUTPUT_DIR}/${RESULT_FILENAME}.json"
echo " Aggregated:  ${OUTPUT_DIR}/agg_${RESULT_FILENAME}.json"
echo "============================================="

# --- Upload results ---
# aws s3 cp "${OUTPUT_DIR}/agg_${RESULT_FILENAME}.json" "s3://vllm-perf/gptoss-fp4/${DATE}/agg_${RESULT_FILENAME}.json"
# aws s3 cp "${OUTPUT_DIR}/${RESULT_FILENAME}.json" "s3://vllm-perf/gptoss-fp4/${DATE}/${RESULT_FILENAME}.json"
buildkite-agent artifact upload "${OUTPUT_DIR}/agg_${RESULT_FILENAME}.json"
buildkite-agent artifact upload "${OUTPUT_DIR}/${RESULT_FILENAME}.json"
