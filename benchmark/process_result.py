#!/usr/bin/env python3
"""Simplified post-processor for standalone benchmark results.

Takes CLI args instead of environment variables. Reads the raw benchmark JSON,
adds metadata, computes per-GPU metrics, and writes agg_*.json.
"""
import argparse
import json
import os
import sys
import requests


def main():
    parser = argparse.ArgumentParser(
        description="Post-process raw benchmark results into aggregated format."
    )
    parser.add_argument("--raw-result", required=True,
                        help="Path to the raw benchmark JSON file")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write aggregated result")
    parser.add_argument("--device", required=True,
                        help="Device type (e.g., h200)")
    parser.add_argument("--date", required=True,
                        help="Date of the benchmark run")
    parser.add_argument("--tp", type=int, required=True,
                        help="Tensor parallel size")
    parser.add_argument("--conc", type=int, default=None,
                        help="Concurrency (overrides value from raw result)")
    parser.add_argument("--framework", required=True,
                        help="Framework name (e.g., vllm)")
    parser.add_argument("--precision", required=True,
                        help="Precision (e.g., fp4, fp8, fp16)")
    parser.add_argument("--model", required=True,
                        help="Model name")
    parser.add_argument("--image", default="",
                        help="Container image used")
    parser.add_argument("--isl", type=int, required=True,
                        help="Input sequence length")
    parser.add_argument("--osl", type=int, required=True,
                        help="Output sequence length")
    parser.add_argument("--model-prefix", default="gptoss",
                        help="Model prefix for InferenceMAX")
    parser.add_argument("--spec-decoding", default="false",
                        help="Whether speculative decoding is used")
    args = parser.parse_args()

    # Read raw benchmark result
    if not os.path.isfile(args.raw_result):
        print(f"ERROR: Raw result file not found: {args.raw_result}",
              file=sys.stderr)
        sys.exit(1)

    with open(args.raw_result) as f:
        bmk_result = json.load(f)

    # Resolve concurrency
    conc = args.conc if args.conc is not None else int(bmk_result.get("max_concurrency", 1))

    # Build aggregated data
    data = {
        "date": args.date,
        "device": args.device,
        "conc": conc,
        "image": args.image,
        "model": bmk_result.get("model_id", args.model),
        "infmax_model_prefix": args.model_prefix,
        "framework": args.framework,
        "precision": args.precision,
        "spec_decoding": args.spec_decoding,
        "disagg": "false",
        "isl": args.isl,
        "osl": args.osl,
        "is_multinode": "false",
        "tp": args.tp,
        "ep": 1,
        "dp_attention": "false",
    }

    # Per-GPU throughput metrics
    total_token_throughput = float(bmk_result.get("total_token_throughput", 0))
    output_throughput = float(bmk_result.get("output_throughput", 0))
    input_throughput = total_token_throughput - output_throughput

    data["tput_per_gpu"] = total_token_throughput / args.tp
    data["output_tput_per_gpu"] = output_throughput / args.tp
    data["input_tput_per_gpu"] = input_throughput / args.tp

    # Convert latency ms -> seconds and compute interactivity (intvty)
    for key, value in bmk_result.items():
        if key.endswith("_ms"):
            data[key.replace("_ms", "")] = float(value) / 1000.0
        if "tpot" in key:
            data[key.replace("_ms", "").replace("tpot", "intvty")] = (
                1000.0 / float(value)
            )

    # Send data to DB
    DATABRICKS_WORKSPACE_URL = os.getenv("DATABRICKS_WORKSPACE_URL")
    ZEROBUS_INGEST_URL = os.getenv("ZEROBUS_INGEST_URL")
    CATALOG = "vllm_data_warehouse"
    SCHEMA = "default"
    TABLE = os.getenv("TABLE")
    access_token = os.getenv("ACCESS_TOKEN")
    JSON_OBJECT = [
        {
            "message": data
        }
    ]

    serialized_objects = [{k: json.dumps(v) for k,v in i.items()} for i in JSON_OBJECT]

    import requests

    try:
        endpoint = "https://vllm-perf-data-ingest-224810116257.us-central1.run.app/"
        headers = {
            "Content-Type": "application/json",
            "X-Source": "Manual Test"
        }
        payload = data
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Successfully sent result to ingestion endpoint (status={response.status_code})")
    except Exception as e:
        print(f"Warning: Failed to POST to ingest endpoint: {e}")

    # Write output
    raw_basename = os.path.splitext(os.path.basename(args.raw_result))[0]
    output_filename = f"agg_{raw_basename}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    print(json.dumps(data, indent=2))

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nAggregated result written to: {output_path}")


if __name__ == "__main__":
    main()
