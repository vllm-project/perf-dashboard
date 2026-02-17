# Standalone GPT-OSS vLLM Benchmark (H200)

Self-contained benchmark for running GPT-OSS-120B FP4 on H200 with vLLM. No dependency on GitHub Actions, SLURM, or the `standalone_benchmark/` directory.

## Requirements

- NVIDIA H200 GPU(s)
- **Docker mode** (default): Docker with NVIDIA Container Toolkit
- **Host mode** (`--host-mode`): vLLM installed locally (`pip install vllm`)
- Python 3.10+ with `numpy`, `aiohttp`, `tqdm`, `transformers`
- HuggingFace access token for gated models (set `HF_TOKEN`)

## Quickstart

```bash
# Docker mode (default) — pulls vLLM image and runs everything
./benchmark/run.sh

# Host mode — uses locally installed vLLM
./benchmark/run.sh --host-mode

# Custom parameters
./benchmark/run.sh --tp 4 --isl 8192 --osl 1024 --conc 32
```

## CLI Reference

| Option | Default | Description |
|---|---|---|
| `--model` | `openai/gpt-oss-120b` | HuggingFace model name |
| `--image` | `vllm/vllm-openai:v0.13.0` | vLLM Docker image |
| `--tp` | `8` | Tensor parallel size |
| `--isl` | `1024` | Input sequence length |
| `--osl` | `1024` | Output sequence length |
| `--conc` | `64` | Max concurrency |
| `--port` | `8888` | Server port |
| `--output-dir` | `benchmark/results` | Directory for results |
| `--gpu-mem-util` | `0.9` | GPU memory utilization |
| `--host-mode` | off | Use host vLLM instead of Docker |
| `--random-range-ratio` | `1.0` | Random range ratio for input/output lengths |
| `--hf-cache` | `~/.cache/huggingface` | HuggingFace cache directory |

## Example Sweep

```bash
# Concurrency sweep at TP=8, ISL/OSL=1024
for c in 4 8 16 32 64; do
  ./benchmark/run.sh --conc $c --output-dir results/conc_sweep
done

# TP sweep at CONC=64, ISL/OSL=1024
for tp in 2 4 8; do
  ./benchmark/run.sh --tp $tp --output-dir results/tp_sweep
done

# Sequence length configurations (from CI)
./benchmark/run.sh --isl 1024 --osl 1024
./benchmark/run.sh --isl 1024 --osl 8192
./benchmark/run.sh --isl 8192 --osl 1024
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
├── process_result.py           # Post-processor (CLI-based)
├── lib/
│   ├── benchmark_serving.py    # Benchmark client
│   ├── backend_request_func.py # Backend HTTP functions
│   └── benchmark_utils.py      # Utilities
└── README.md
```
