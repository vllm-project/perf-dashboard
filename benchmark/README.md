# Standalone GPT-OSS vLLM Benchmark (H200)

Self-contained benchmark for running GPT-OSS-120B FP4 on H200 with vLLM. No dependency on GitHub Actions, SLURM, or the `standalone_benchmark/` directory.

## Requirements

- NVIDIA H200 GPU(s)
- vLLM installed locally (`pip install vllm`)
- Python 3.10+ with `numpy`, `aiohttp`, `tqdm`, `transformers`
- HuggingFace access token for gated models (set `HF_TOKEN`)

Install Python dependencies from repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
./benchmark/run.sh \
  --model openai/gpt-oss-120b \
  --image vllm/vllm-openai:v0.13.0 \
  --tp 8 --isl 1024 --osl 1024 --conc 64 \
  --gpu-mem-util 0.9 --random-range-ratio 1.0 \
  --hf-cache /dev/shm/.cache/huggingface
```

## nCompass NSYS Profile Suite

Use `run_ncompass_profile.sh` to run inference-only traces for pure decode/prefill
workloads in CI or locally.

```bash
# Default suite:
# - decode: INPUT_LEN=2, OUTPUT_LEN=1024, CONCURRENCY=1/32/256
# - prefill: INPUT_LEN=8192, OUTPUT_LEN=1, CONCURRENCY=1
./benchmark/run_ncompass_profile.sh

# Decode-only sweep
WORKLOAD_MODE=decode ./benchmark/run_ncompass_profile.sh

# Prefill-only
WORKLOAD_MODE=prefill ./benchmark/run_ncompass_profile.sh

# Single custom case (BATCH_SIZE aliases CONCURRENCY)
INPUT_LEN=4096 OUTPUT_LEN=16 CONCURRENCY=8 ./benchmark/run_ncompass_profile.sh
```

The profiled server is launched with nCompass + NSYS:

```bash
NCOMPASS_CACHE_DIR=<path-to-ncompass-configs> NCOMPASS_PROFILER_TYPE=NSYS \
ncompass profile --nsys -- vllm serve ...
```

The runner polls server terminal logs to detect engine readiness, validates
`/health`, runs `vllm bench serve`, collects `.nsys-rep` traces, and writes a
nightly markdown report:

- `benchmark/results/ncompass/nightly_ncompass_profile_report.md`

If your nCompass NVTX/CudaProfiler configs are placeholders (for example `{}`),
the runner automatically falls back to a direct `nsys profile` launch (capture
starts immediately) so `.nsys-rep` files are still emitted. You can still
override nCompass NSYS args explicitly with:

```bash
NCOMPASS_NSYS_ARGS='--capture-range=none' ./benchmark/run_ncompass_profile.sh
```

Optional sharing:

- set `NCOMPASS_SHARE_COMMAND` with `{trace}` placeholder to generate share links
- example: `NCOMPASS_SHARE_COMMAND='ncompass upload --share {trace}'`

## CLI Reference

### Required

| Option | Description |
|---|---|
| `--model` | HuggingFace model name |
| `--image` | vLLM Docker image |
| `--tp` | Tensor parallel size |
| `--isl` | Input sequence length |
| `--osl` | Output sequence length |
| `--conc` | Max concurrency |
| `--gpu-mem-util` | GPU memory utilization |
| `--random-range-ratio` | Random range ratio for input/output lengths |
| `--hf-cache` | HuggingFace cache directory |

### Optional

| Option | Default | Description |
|---|---|---|
| `--port` | `8888` | Server port |
| `--output-dir` | `benchmark/results` | Directory for results |

## Example Sweep

```bash
# Concurrency sweep at TP=8, ISL/OSL=1024
for c in 4 8 16 32 64; do
  ./benchmark/run.sh \
    --model openai/gpt-oss-120b --image vllm/vllm-openai:v0.13.0 \
    --tp 8 --isl 1024 --osl 1024 --conc $c \
    --gpu-mem-util 0.9 --random-range-ratio 1.0 \
    --hf-cache /dev/shm/.cache/huggingface \
    --output-dir results/conc_sweep
done

# TP sweep at CONC=64, ISL/OSL=1024
for tp in 2 4 8; do
  ./benchmark/run.sh \
    --model openai/gpt-oss-120b --image vllm/vllm-openai:v0.13.0 \
    --tp $tp --isl 1024 --osl 1024 --conc 64 \
    --gpu-mem-util 0.9 --random-range-ratio 1.0 \
    --hf-cache /dev/shm/.cache/huggingface \
    --output-dir results/tp_sweep
done
```

## Output Format

Each run produces two files in the output directory:

### Raw result (`gptoss_fp4_vllm_tp8_isl1024_osl1024_conc64.json`)

Standard benchmark_serving.py output including throughput, latency percentiles, and request-level metrics.

### Aggregated result (`agg_gptoss_fp4_vllm_tp8_isl1024_osl1024_conc64.json`)

```json
{
  "device": "h200",
  "conc": 64,
  "model": "openai/gpt-oss-120b",
  "framework": "vllm",
  "precision": "fp4",
  "tp": 8,
  "isl": 1024,
  "osl": 1024,
  "tput_per_gpu": 1234.56,
  "output_tput_per_gpu": 567.89,
  "mean_ttft": 0.123,
  "median_tpot": 0.0045,
  "median_intvty": 222.22
}
```

Key computed fields:
- `tput_per_gpu` = total_token_throughput / tp
- `output_tput_per_gpu` = output_throughput / tp
- `*_intvty` = 1000 / median_tpot_ms (interactivity: tokens/second)
- Latency fields converted from ms to seconds

## Directory Structure

```
benchmark/
├── run.sh                      # Main entrypoint
├── run_ncompass_profile.sh     # NSYS+nCompass profile suite entrypoint
├── process_result.py           # Post-processor (CLI-based)
├── lib/
│   ├── benchmark_serving.py    # Benchmark client
│   ├── backend_request_func.py # Backend HTTP functions
│   └── benchmark_utils.py      # Utilities
└── README.md
```
