#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-openai/gpt-oss-20b}"
IMAGE="${IMAGE:-}"
TP="${TP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/ncompass}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
WORKLOAD_MODE="${WORKLOAD_MODE:-single}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-}"
DEFAULT_DECODE_INPUT_LEN="${DEFAULT_DECODE_INPUT_LEN:-2}"
DEFAULT_DECODE_OUTPUT_LEN="${DEFAULT_DECODE_OUTPUT_LEN:-64}"
DEFAULT_PREFILL_INPUT_LEN="${DEFAULT_PREFILL_INPUT_LEN:-8192}"
DEFAULT_PREFILL_OUTPUT_LEN="${DEFAULT_PREFILL_OUTPUT_LEN:-1}"
HIGH_SEQ_LEN_THRESHOLD="${HIGH_SEQ_LEN_THRESHOLD:-1024}"
MAX_MODEL_LEN_PADDING_LARGE_SEQ="${MAX_MODEL_LEN_PADDING_LARGE_SEQ:-128}"
MAX_MODEL_LEN_PADDING_DEFAULT="${MAX_MODEL_LEN_PADDING_DEFAULT:-200}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
NCOMPASS_CACHE_DIR="${NCOMPASS_CACHE_DIR:-${SCRIPT_DIR}/ncompass-config}"
NCOMPASS_PROFILER_TYPE="${NCOMPASS_PROFILER_TYPE:-NSYS}"
NCOMPASS_NSYS_ARGS="${NCOMPASS_NSYS_ARGS:-}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-5}"
DECODE_CONCURRENCY_LIST="${DECODE_CONCURRENCY_LIST:-1 32 256}"
PREFILL_CONCURRENCY_LIST="${PREFILL_CONCURRENCY_LIST:-1}"

LOCAL_MODE=0
REPORT_FILE=""
PROFILE_LAUNCHER="ncompass"
PROFILE_NSYS_ARGS="<ncompass defaults>"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Run nCompass+nsys profiled vLLM benchmark cases (decode/prefill) and publish traces.

Options:
  --model MODEL                  Model name (default: ${MODEL})
  --image IMAGE                  Image label metadata only (default: empty)
  --tp TP                        Tensor parallel size (default: ${TP})
  --output-dir DIR               Output directory (default: ${OUTPUT_DIR})
  --gpu-mem-util FLOAT           GPU memory utilization (default: ${GPU_MEM_UTIL})
  --workload MODE                decode|prefill|suite|single (default: ${WORKLOAD_MODE})
  --concurrency-list LIST        Override conc list, e.g. "1 32 256" or "1,32,256"
  --ncompass-cache-dir DIR       nCompass config cache dir
  --local                        Load env vars from .env file (for local debugging)
  -h, --help                     Show help

Environment selectors:
  PROFILE_CASES                  Comma-separated MODE:INPUT:OUTPUT:CONC list
                                 Example: decode:${DEFAULT_DECODE_INPUT_LEN}:${DEFAULT_DECODE_OUTPUT_LEN}:1,decode:${DEFAULT_DECODE_INPUT_LEN}:${DEFAULT_DECODE_OUTPUT_LEN}:32,prefill:${DEFAULT_PREFILL_INPUT_LEN}:${DEFAULT_PREFILL_OUTPUT_LEN}:1
  INPUT_LEN                      Single-case input length (requires OUTPUT_LEN and CONCURRENCY/BATCH_SIZE)
  OUTPUT_LEN                     Single-case output length (requires INPUT_LEN and CONCURRENCY/BATCH_SIZE)
  CONCURRENCY                    Single-case concurrency
  BATCH_SIZE                     Alias of CONCURRENCY for single-case mode
  WORKLOAD / WORKLOAD_MODE       decode|prefill|suite|single selectors

  NCOMPASS_NSYS_ARGS             Extra args to pass to 'ncompass profile --nsys'
                                 Example: NCOMPASS_NSYS_ARGS='--capture-range=none'

Optional trace sharing:
  NCOMPASS_SHARE_COMMAND         Shell template for sharing trace, use {trace} placeholder.
                                 Example: ncompass upload --share {trace}
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --workload) WORKLOAD_MODE="$2"; shift 2 ;;
        --concurrency-list) CONCURRENCY_LIST="$2"; shift 2 ;;
        --ncompass-cache-dir) NCOMPASS_CACHE_DIR="$2"; shift 2 ;;
        --local) LOCAL_MODE=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ "$LOCAL_MODE" -eq 1 ]]; then
    ENV_FILE="${SCRIPT_DIR}/.env"
    if [[ -f "$ENV_FILE" ]]; then
        echo "Loading env vars from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo "WARNING: --local specified but no .env file found at $ENV_FILE"
    fi
fi

if [[ -n "${WORKLOAD:-}" ]]; then
    WORKLOAD_MODE="${WORKLOAD}"
fi

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $cmd"
        exit 1
    fi
}

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}


share_trace() {
    local trace_path="$1"
    local out=""
    local share_link=""

    if [[ -z "${NCOMPASS_SHARE_COMMAND:-}" ]]; then
        echo ""
        return 0
    fi

    local share_cmd="${NCOMPASS_SHARE_COMMAND//\{trace\}/${trace_path}}"
    set +e
    out="$(bash -lc "$share_cmd" 2>&1)"
    local rc=$?
    set -e

    printf '%s\n' "$out" > "${trace_path}.share.log"

    if [[ $rc -ne 0 ]]; then
        echo ""
        return 0
    fi

    share_link="$(printf '%s\n' "$out" | grep -Eo 'https://share\.ncompass\.tech[^[:space:]]+' | head -n 1 || true)"
    echo "$share_link"
}

append_case() {
    local mode="$1"
    local input_len="$2"
    local output_len="$3"
    local conc="$4"

    if ! is_positive_integer "$input_len" || ! is_positive_integer "$output_len" || ! is_positive_integer "$conc"; then
        echo "ERROR: invalid case tuple ${mode}:${input_len}:${output_len}:${conc}"
        exit 1
    fi
    CASE_MODES+=("$mode")
    CASE_INPUTS+=("$input_len")
    CASE_OUTPUTS+=("$output_len")
    CASE_CONCURRENCIES+=("$conc")
}

append_cases_from_list() {
    local mode="$1"
    local input_len="$2"
    local output_len="$3"
    local list="$4"
    local conc
    for conc in ${list//,/ }; do
        append_case "$mode" "$input_len" "$output_len" "$conc"
    done
}

# Install nsys if not already present in the image
if ! command -v nsys >/dev/null 2>&1; then
    echo "INFO: nsys not found, installing Nsight Systems CLI..."
    apt-get update -y \
        && apt-get install -y --no-install-recommends wget ca-certificates \
        && wget -q https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
        && dpkg -i NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb \
        && rm NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb
    export PATH="/usr/local/nsight-systems/bin:${PATH}"
fi

require_command curl
require_command grep
require_command find
require_command ncompass
require_command nsys
require_command vllm

mkdir -p "$OUTPUT_DIR"
mkdir -p "$NCOMPASS_CACHE_DIR"

REPORT_FILE="${OUTPUT_DIR}/nightly_ncompass_profile_report.md"

if [[ ! -d "$NCOMPASS_CACHE_DIR" ]]; then
    echo "ERROR: nCompass config dir does not exist: $NCOMPASS_CACHE_DIR"
    exit 1
fi

if [[ -z "$(find "$NCOMPASS_CACHE_DIR" -mindepth 1 -type f | head -n 1 || true)" ]]; then
    echo "WARNING: NCOMPASS_CACHE_DIR appears empty: $NCOMPASS_CACHE_DIR"
    echo "         Provide injector configs to ensure inference-only tracing."
fi

declare -a NCOMPASS_NSYS_ARGV=()
if [[ -n "$NCOMPASS_NSYS_ARGS" ]]; then
    read -r -a NCOMPASS_NSYS_ARGV <<< "$NCOMPASS_NSYS_ARGS"
    PROFILE_NSYS_ARGS="$NCOMPASS_NSYS_ARGS"
else
    PROFILE_NSYS_ARGS="<direct defaults>"
fi

if ! is_positive_integer "$TP"; then
    echo "ERROR: TP must be a positive integer"
    exit 1
fi
if ! is_positive_integer "$NUM_PROMPTS_MULTIPLIER"; then
    echo "ERROR: NUM_PROMPTS_MULTIPLIER must be a positive integer"
    exit 1
fi

declare -a CASE_MODES=()
declare -a CASE_INPUTS=()
declare -a CASE_OUTPUTS=()
declare -a CASE_CONCURRENCIES=()

if [[ -n "${PROFILE_CASES:-}" ]]; then
    IFS=',' read -r -a explicit_cases <<< "${PROFILE_CASES}"
    for tuple in "${explicit_cases[@]}"; do
        IFS=':' read -r mode in_len out_len conc <<< "$tuple"
        if [[ -z "${mode:-}" || -z "${in_len:-}" || -z "${out_len:-}" || -z "${conc:-}" ]]; then
            echo "ERROR: invalid PROFILE_CASES tuple: $tuple"
            exit 1
        fi
        append_case "$mode" "$in_len" "$out_len" "$conc"
    done
else
    selected_concurrency="${CONCURRENCY:-${BATCH_SIZE:-}}"
    if [[ -n "${INPUT_LEN:-}" || -n "${OUTPUT_LEN:-}" || -n "${selected_concurrency:-}" ]]; then
        if [[ -z "${INPUT_LEN:-}" || -z "${OUTPUT_LEN:-}" || -z "${selected_concurrency:-}" ]]; then
            echo "ERROR: set INPUT_LEN, OUTPUT_LEN and CONCURRENCY (or BATCH_SIZE) together"
            exit 1
        fi
        append_case "custom" "$INPUT_LEN" "$OUTPUT_LEN" "$selected_concurrency"
    else
        case "$WORKLOAD_MODE" in
            single|default)
                append_case "decode" "$DEFAULT_DECODE_INPUT_LEN" "$DEFAULT_DECODE_OUTPUT_LEN" 1
                ;;
            decode)
                append_cases_from_list "decode" "$DEFAULT_DECODE_INPUT_LEN" "$DEFAULT_DECODE_OUTPUT_LEN" "${CONCURRENCY_LIST:-$DECODE_CONCURRENCY_LIST}"
                ;;
            prefill)
                append_cases_from_list "prefill" "$DEFAULT_PREFILL_INPUT_LEN" "$DEFAULT_PREFILL_OUTPUT_LEN" "${CONCURRENCY_LIST:-$PREFILL_CONCURRENCY_LIST}"
                ;;
            suite|both)
                append_cases_from_list "decode" "$DEFAULT_DECODE_INPUT_LEN" "$DEFAULT_DECODE_OUTPUT_LEN" "$DECODE_CONCURRENCY_LIST"
                append_cases_from_list "prefill" "$DEFAULT_PREFILL_INPUT_LEN" "$DEFAULT_PREFILL_OUTPUT_LEN" "$PREFILL_CONCURRENCY_LIST"
                ;;
            *)
                echo "ERROR: unsupported workload mode: $WORKLOAD_MODE"
                exit 1
                ;;
        esac
    fi
fi

if [[ ${#CASE_MODES[@]} -eq 0 ]]; then
    echo "ERROR: no profile cases to run"
    exit 1
fi

cat > "$REPORT_FILE" <<REPORT
# Nightly nCompass NSYS Profile Report

- model: ${MODEL}
- image: ${IMAGE:-n/a}
- tp: ${TP}
- profiler_launcher: ${PROFILE_LAUNCHER}
- ncompass_cache_dir: ${NCOMPASS_CACHE_DIR}
- nsys_args: ${PROFILE_NSYS_ARGS}
- generated_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

## Cases
REPORT

echo "==========================================================="
echo "nCompass NSYS profiled benchmark suite"
echo "Model: ${MODEL}"
echo "TP: ${TP}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Cases: ${#CASE_MODES[@]}"
echo "==========================================================="

case_count=${#CASE_MODES[@]}
for ((i = 0; i < case_count; ++i)); do
    mode="${CASE_MODES[$i]}"
    input_len="${CASE_INPUTS[$i]}"
    output_len="${CASE_OUTPUTS[$i]}"
    conc="${CASE_CONCURRENCIES[$i]}"
    case_name="${mode}_isl${input_len}_osl${output_len}_conc${conc}"
    case_dir="${OUTPUT_DIR}/${case_name}"
    mkdir -p "$case_dir"
    result_filename="${case_dir}/bench_${case_name}.json"
    CURRENT_SERVER_LOG="${case_dir}/server.log"
    num_prompts=$((conc * NUM_PROMPTS_MULTIPLIER))

    echo ""
    echo "--- Running case: ${case_name}"
    echo "INPUT_LEN=${input_len} OUTPUT_LEN=${output_len} CONCURRENCY=${conc} NUM_PROMPTS=${num_prompts}"

    mapfile -t traces_before < <(find "$case_dir" -type f -name '*.nsys-rep' | sort)
    declare -A before_mtime=()
    for t in "${traces_before[@]}"; do
        before_mtime["$t"]="$(stat -c %Y "$t" 2>/dev/null || echo 0)"
    done

    if [[ "$input_len" -ge "$HIGH_SEQ_LEN_THRESHOLD" && "$output_len" -ge "$HIGH_SEQ_LEN_THRESHOLD" ]]; then
        max_model_len=$((input_len + output_len + MAX_MODEL_LEN_PADDING_LARGE_SEQ))
    else
        max_model_len=$((input_len + output_len + MAX_MODEL_LEN_PADDING_DEFAULT))
    fi

    (
        cd "$case_dir"
        export NCOMPASS_CACHE_DIR="$NCOMPASS_CACHE_DIR"
        export NCOMPASS_PROFILER_TYPE="$NCOMPASS_PROFILER_TYPE"
        export VLLM_MXFP4_USE_MARLIN="${VLLM_MXFP4_USE_MARLIN:-1}"
        export PYTHONNOUSERSITE=1

        echo "INFO: Launching vLLM offline engine directly in python..."
        exec ncompass profile --nsys \
            --trace=cuda,nvtx \
            --sample=process-tree \
            --cuda-graph-trace=node \
            --force-overwrite=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            ${NCOMPASS_NSYS_ARGV[@]+"${NCOMPASS_NSYS_ARGV[@]}"} \
            -- python3 "${SCRIPT_DIR}/profile_offline.py" \
            --model "$MODEL" \
            --tp "$TP" \
            --input-len "$input_len" \
            --output-len "$output_len" \
            --concurrency "$conc" \
            --num-prompts "$num_prompts" \
            --gpu-mem-util "$GPU_MEM_UTIL" \
            --max-model-len "$max_model_len" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --result-file "$result_filename"
    ) > "$CURRENT_SERVER_LOG" 2>&1
    
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "ERROR: Offline profiling failed for ${case_name}"
        exit 1
    fi

    # NSYS can take a bit to finalize/write .nsys-rep after process shutdown.
    # Wait up to 60s, polling every 0.5s.
    mapfile -t traces_after < <(find "$case_dir" -type f -name '*.nsys-rep' | sort)
    new_traces=()
    for ((wait_iter = 0; wait_iter <= 120; ++wait_iter)); do
        mapfile -t traces_after < <(find "$case_dir" -type f -name '*.nsys-rep' | sort)
        new_traces=()
        for t in "${traces_after[@]}"; do
            if [[ -z "${before_mtime[$t]:-}" ]]; then
                new_traces+=("$t")
                continue
            fi
            after_mtime="$(stat -c %Y "$t" 2>/dev/null || echo 0)"
            if [[ "$after_mtime" -gt "${before_mtime[$t]}" ]]; then
                new_traces+=("$t")
            fi
        done

        if [[ ${#new_traces[@]} -gt 0 ]]; then
            break
        fi

        if [[ "$wait_iter" -lt 120 ]]; then
            sleep 0.5
        fi
    done

    if [[ ${#new_traces[@]} -eq 0 ]]; then
        echo "WARNING: no new .nsys-rep found for ${case_name} after waiting 60s."
    else
        # Flatten ncompass's timestamped subdirs: move .nsys-rep files directly
        # into .nsys_traces/ and remove the now-empty subdirs.
        nsys_traces_dir="${case_dir}/.nsys_traces"
        flattened_traces=()
        for trace in "${new_traces[@]}"; do
            trace_parent="$(dirname "$trace")"
            if [[ "$trace_parent" != "$nsys_traces_dir" && -d "$nsys_traces_dir" ]]; then
                dest="${nsys_traces_dir}/$(basename "$trace")"
                mv "$trace" "$dest"
                find "$trace_parent" -empty -type d -delete 2>/dev/null || true
                flattened_traces+=("$dest")
            else
                flattened_traces+=("$trace")
            fi
        done
        new_traces=("${flattened_traces[@]}")
    fi

    {
        echo "### ${case_name}"
        echo ""
        echo "- mode: ${mode}"
        echo "- input_len: ${input_len}"
        echo "- output_len: ${output_len}"
        echo "- concurrency: ${conc}"
        echo "- num_prompts: ${num_prompts}"
        echo "- server_log: ${CURRENT_SERVER_LOG}"
        echo "- results_file: ${result_filename}"
        if [[ ${#new_traces[@]} -eq 0 ]]; then
            echo "- traces: none found"
        else
            for trace in "${new_traces[@]}"; do
                share_link="$(share_trace "$trace")"
                if [[ -n "$share_link" ]]; then
                    echo "- trace: ${trace}"
                    echo "  - share_link: ${share_link}"
                else
                    echo "- trace: ${trace}"
                    echo "  - share_link: not generated (set NCOMPASS_SHARE_COMMAND)"
                fi
            done
        fi
        echo ""
    } >> "$REPORT_FILE"

    # --- Upload this case's traces to Supabase Storage ---
    if [[ -n "${SUPABASE_URL:-}" && -n "${SUPABASE_ANON_KEY:-}" ]]; then
        echo "Uploading traces for ${case_name} to Supabase Storage..."
        python3 "${SCRIPT_DIR}/upload_to_supabase.py" \
            --output-dir "$case_dir" \
            --model "$MODEL" \
            --tp "$TP" \
            --isl "$input_len" \
            --osl "$output_len" \
            --conc "$conc" \
            --image "${IMAGE:-}" \
            --links-file "${case_dir}/supabase_trace_links.txt" \
            || echo "WARNING: Supabase upload failed for ${case_name} (non-fatal)"
    fi
done

echo ""
echo "Profile suite complete. Report: $REPORT_FILE"

if command -v buildkite-agent >/dev/null 2>&1; then
    shopt -s nullglob globstar
    artifacts=("$REPORT_FILE")
    for path in "$OUTPUT_DIR"/**/*.nsys-rep "$OUTPUT_DIR"/**/*.json "$OUTPUT_DIR"/**/*.log "$OUTPUT_DIR"/**/*.txt; do
        artifacts+=("$path")
    done

    if [[ ${#artifacts[@]} -gt 0 ]]; then
        buildkite-agent artifact upload "${artifacts[@]}"
    fi
fi
