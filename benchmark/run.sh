#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Defaults ─────────────────────────────────────────────────────────────────
PORT=8888
OUTPUT_DIR="${SCRIPT_DIR}/results"

# ─── Required (no defaults) ──────────────────────────────────────────────────
MODEL=""
IMAGE=$(buildkite-agent meta-data get "image")
TP=""
ISL=""
OSL=""
CONC=""
GPU_MEM_UTIL=""
RANDOM_RANGE_RATIO=""
HF_CACHE=""
CONFIG_FILE=""

# ─── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --model MODEL           HuggingFace model name
  --image IMAGE           vLLM Docker image
  --device DEVICE         Device type (h200, b200)
  --tp TP                 Tensor parallel size
  --precision PRECISION   Precision (fp4, fp8, fp16)
  --isl ISL               Input sequence length
  --osl OSL               Output sequence length
  --conc CONC             Max concurrency
  --gpu-mem-util FLOAT    GPU memory utilization
  --random-range-ratio R  Random range ratio
  --hf-cache DIR          HuggingFace cache directory
  --config CONFIG_FILE    vLLM config file

Optional:
  --date DATE             Timestamp for results            (default: current date/time)
  --port PORT             Server port                      (default: $PORT)
  --output-dir DIR        Directory for results            (default: $OUTPUT_DIR)
  -h, --help              Show this help message

Example:
  ./run.sh --model openai/gpt-oss-120b --image vllm/vllm-openai:v0.13.0 \\
           --tp 8 --precision fp4 --isl 1024 --osl 1024 --conc 64 \\
           --gpu-mem-util 0.9 --random-range-ratio 1.0 \\
           --hf-cache /dev/shm/.cache/huggingface --config configs/h200_gpt_oss.yaml
EOF
    exit 0
}

# ─── Parse CLI args ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)           MODEL="$2";           shift 2 ;;
        --image)           IMAGE="$2";           shift 2 ;;
        --device)          DEVICE="$2";          shift 2 ;;
        --tp)              TP="$2";              shift 2 ;;
        --precision)       PRECISION="$2";       shift 2 ;;
        --isl)             ISL="$2";             shift 2 ;;
        --osl)             OSL="$2";             shift 2 ;;
        --conc)            CONC="$2";            shift 2 ;;
        --date)            DATE="$2";            shift 2 ;;
        --port)            PORT="$2";            shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";      shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2";    shift 2 ;;
        --random-range-ratio) RANDOM_RANGE_RATIO="$2"; shift 2 ;;
        --hf-cache)        HF_CACHE="$2";        shift 2 ;;
        --config)          CONFIG_FILE="$2";      shift 2 ;;
        -h|--help)         usage ;;
        *)                 echo "Unknown option: $1"; usage ;;
    esac
done

# ─── Validate required args ──────────────────────────────────────────────────
MISSING=()
[[ -z "$MODEL" ]]              && MISSING+=("--model")
[[ -z "$IMAGE" ]]              && MISSING+=("--image")
[[ -z "$DEVICE" ]]             && MISSING+=("--device")
[[ -z "$TP" ]]                 && MISSING+=("--tp")
[[ -z "$PRECISION" ]]          && MISSING+=("--precision")
[[ -z "$ISL" ]]                && MISSING+=("--isl")
[[ -z "$OSL" ]]                && MISSING+=("--osl")
[[ -z "$CONC" ]]               && MISSING+=("--conc")
[[ -z "$GPU_MEM_UTIL" ]]       && MISSING+=("--gpu-mem-util")
[[ -z "$RANDOM_RANGE_RATIO" ]] && MISSING+=("--random-range-ratio")
[[ -z "$HF_CACHE" ]]           && MISSING+=("--hf-cache")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required options: ${MISSING[*]}"
    echo ""
    usage
fi

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
RESULT_FILENAME="${MODEL}_${PRECISION}_tp${TP}_isl${ISL}_osl${OSL}_conc${CONC}"

mkdir -p "$OUTPUT_DIR"

echo "============================================="
echo "vLLM Benchmark"
echo "============================================="
echo " Model:          $MODEL"
echo " Device:         $DEVICE"
echo " TP:             $TP"
echo " Precision:      $PRECISION"
echo " ISL:            $ISL"
echo " OSL:            $OSL"
echo " Concurrency:    $CONC"
echo " Max Model Len:  $MAX_MODEL_LEN"
echo " Num Prompts:    $NUM_PROMPTS"
echo " Num Warmups:    $NUM_WARMUPS"
echo " Port:           $PORT"
echo " Output Dir:     $OUTPUT_DIR"
echo "============================================="

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
if [ "$PRECISION" = "fp4" ]; then
    export VLLM_MXFP4_USE_MARLIN=1
fi
# export TORCH_CUDA_ARCH_LIST="9.0"

PYTHONNOUSERSITE=1 vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    ${CONFIG_FILE:+--config "$CONFIG_FILE"} \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
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
nvidia-smi || true

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
DATE="${DATE:-$(date +'%Y-%m-%d %H:%M:%S')}"

python3 "${SCRIPT_DIR}/process_result.py" \
    --raw-result "${OUTPUT_DIR}/${RESULT_FILENAME}.json" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --date "$DATE" \
    --tp "$TP" \
    --conc "$CONC" \
    --framework vllm \
    --precision ${PRECISION} \
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
buildkite-agent artifact upload "${OUTPUT_DIR}/agg_${RESULT_FILENAME}.json"
buildkite-agent artifact upload "${OUTPUT_DIR}/${RESULT_FILENAME}.json"
