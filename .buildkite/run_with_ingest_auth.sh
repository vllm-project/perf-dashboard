#!/usr/bin/env bash
set -euo pipefail

INGEST_BEARER_TOKEN="$(buildkite-agent secret get INGEST_BEARER_TOKEN)"
export INGEST_BEARER_TOKEN

exec "$@"
