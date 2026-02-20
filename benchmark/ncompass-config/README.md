# nCompass Config Cache

Place nCompass injector/profile configs in this directory when running
`benchmark/run_ncompass_profile.sh` locally.

This repo includes minimal bootstrap configs at:

- `.cache/ncompass/profiles/.default/NVTX/current/config.json`
- `.cache/ncompass/profiles/.default/CudaProfiler/current/config.json`

These satisfy nCompass initialization checks. Replace/extend them with your
real injector config to capture the exact inference region you want.

Note: placeholder configs like `{}` do not inject any CUDA profiler start/stop
markers. With nCompass default NSYS args (`--capture-range=cudaProfilerApi`),
that yields `No reports were generated`. The profile runner now auto-falls back
to a direct `nsys profile` launch when no rewrite targets are present.

In CI, you can override this path with:

```bash
NCOMPASS_CACHE_DIR=/path/to/ncompass/configs
```

The profile runner will export `NCOMPASS_CACHE_DIR` and
`NCOMPASS_PROFILER_TYPE=NSYS` before launching:

```bash
ncompass profile --nsys -- vllm serve ...
```
