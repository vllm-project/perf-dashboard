#!/usr/bin/env bash
set -euo pipefail

echo "Running benchmark..."
IMAGE=$(buildkite-agent meta-data get image)
CONFIG=$(buildkite-agent meta-data get config)
DATE=$(buildkite-agent meta-data get date --default "")
echo "Image: $IMAGE"
echo "Config: $CONFIG"
echo "Date: ${DATE:-<auto>}"
for config in $CONFIG; do
    config=$(echo "$config" | tr -d '\r')
    IMAGE=$IMAGE DATE=$DATE buildkite-agent pipeline upload ./.buildkite/configs/${config}.yaml
done
exit 0
