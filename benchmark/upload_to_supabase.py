#!/usr/bin/env python3
"""Upload .nsys-rep trace files to Supabase Storage and produce a links manifest."""

import argparse
import os
import pathlib
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(
        description="Upload .nsys-rep files to Supabase Storage and write a links manifest."
    )
    parser.add_argument("--output-dir", required=True, help="Directory to scan for .nsys-rep files")
    parser.add_argument("--model", required=True, help="Model name (used in object key path)")
    parser.add_argument("--tp", required=True, help="Tensor parallel size (used in object key path)")
    parser.add_argument("--isl", default="", help="Input sequence length")
    parser.add_argument("--osl", default="", help="Output sequence length")
    parser.add_argument("--conc", default="", help="Concurrency")
    parser.add_argument("--image", default="", help="Image label (metadata only)")
    parser.add_argument("--links-file", required=True, help="Path to write the links manifest")
    parser.add_argument("--bucket", default=None, help="Supabase Storage bucket name")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_ANON_KEY", "")
    bucket_name = args.bucket or os.environ.get("SUPABASE_BUCKET", "vllm-traces-public")

    if not url or not key:
        print("WARNING: SUPABASE_URL or SUPABASE_ANON_KEY not set. Skipping upload.")
        sys.exit(0)

    from supabase import create_client

    supabase = create_client(url, key)
    storage = supabase.storage.from_(bucket_name)

    model_slug = args.model.replace("/", "-")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    build_id = os.environ.get(
        "BUILDKITE_BUILD_NUMBER",
        os.environ.get("BUILDKITE_BUILD_ID", datetime.now(timezone.utc).strftime("%H%M%S")),
    )

    # Build a descriptive base name: model_tp_isl_osl_conc
    name_parts = [model_slug, f"tp{args.tp}"]
    if args.isl:
        name_parts.append(f"isl{args.isl}")
    if args.osl:
        name_parts.append(f"osl{args.osl}")
    if args.conc:
        name_parts.append(f"conc{args.conc}")
    base_name = "_".join(name_parts)

    output_path = pathlib.Path(args.output_dir)
    nsys_files = sorted(output_path.rglob("*.nsys-rep"))

    if not nsys_files:
        print("No .nsys-rep files found. Nothing to upload.")
        sys.exit(0)

    links = []
    for i, nsys_file in enumerate(nsys_files):
        # e.g. openai-gpt-oss-120b_tp8_isl1024_osl1024_conc32.nsys-rep
        suffix = f"_{i}" if len(nsys_files) > 1 else ""
        object_key = f"{date_str}/build-{build_id}/{base_name}{suffix}.nsys-rep"

        print(f"Uploading {nsys_file} -> {bucket_name}/{object_key}")
        try:
            with open(nsys_file, "rb") as f:
                storage.upload(
                    path=object_key,
                    file=f,
                    file_options={"content-type": "application/octet-stream"},
                )
            public_url = storage.get_public_url(object_key)
            links.append((object_key, public_url))
            print(f"  OK: {public_url}")
        except Exception as e:
            print(f"  ERROR uploading {nsys_file}: {e}")

    links_path = pathlib.Path(args.links_file)
    links_path.parent.mkdir(parents=True, exist_ok=True)
    with open(links_path, "w") as f:
        f.write("# Supabase NSYS Trace Links\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# TP: {args.tp}\n")
        f.write(f"# ISL: {args.isl or 'n/a'}\n")
        f.write(f"# OSL: {args.osl or 'n/a'}\n")
        f.write(f"# Concurrency: {args.conc or 'n/a'}\n")
        f.write(f"# Image: {args.image or 'n/a'}\n")
        f.write(f"# Date: {date_str}\n")
        f.write(f"# Build: {build_id}\n\n")
        for rel_path, pub_url in links:
            f.write(f"{rel_path}\n  {pub_url}\n\n")

    print(f"\nUploaded {len(links)}/{len(nsys_files)} traces. Links file: {args.links_file}")


if __name__ == "__main__":
    main()
