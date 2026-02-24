#!/usr/bin/env bash
set -euo pipefail

echo "Running benchmark..."
IMAGE=$(buildkite-agent meta-data get image)
CONFIG=$(buildkite-agent meta-data get config)
echo "Image: $IMAGE"
echo "Config: $CONFIG"
for config in $CONFIG; do
    IMAGE=$IMAGE buildkite-agent pipeline upload ./.buildkite/configs/${config}.yaml
done
exit 0
