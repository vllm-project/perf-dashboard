#!/usr/bin/env bash
set -euo pipefail

echo "Running nCompass profiling bootstrap..."
IMAGE=$(buildkite-agent meta-data get image)
CONFIG=$(buildkite-agent meta-data get config)
echo "Image: $IMAGE"
echo "Config: $CONFIG"

for config in $CONFIG; do
    config=$(echo "$config" | tr -d '\r')
    echo "Uploading nCompass config for $config..."
    IMAGE=$IMAGE buildkite-agent pipeline upload ./.buildkite/configs/ncompass/${config}.yaml
done

exit 0
