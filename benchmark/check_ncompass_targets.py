import json
import pathlib
import sys

def main():
    if len(sys.argv) < 3:
        sys.exit(1)
        
    cache_dir = pathlib.Path(sys.argv[1])
    profiler_type_str = sys.argv[2].strip()

    profiler_aliases = {
        "NSYS": ["NVTX", "CudaProfiler"],
    }

    if profiler_type_str in profiler_aliases:
        profiler_types = profiler_aliases[profiler_type_str]
    else:
        profiler_types = [p.strip() for p in profiler_type_str.split(",") if p.strip()]

    rewrite_profiler_types = [p for p in profiler_types if p != "NCU"]

    for profiler_type in rewrite_profiler_types:
        config_path = (
            cache_dir
            / ".cache"
            / "ncompass"
            / "profiles"
            / ".default"
            / profiler_type
            / "current"
            / "config.json"
        )
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception:
            continue

        targets = cfg.get("targets")
        if isinstance(targets, dict) and len(targets) > 0:
            sys.exit(0)

    sys.exit(1)

if __name__ == "__main__":
    main()