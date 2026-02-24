#!/usr/bin/env bash
set -euo pipefail

echo "Running benchmark..."
IMAGE=$(buildkite-agent meta-data get image)
CONFIG=$(buildkite-agent meta-data get config)
echo "Image: $IMAGE"
echo "Config: $CONFIG"
for config in $CONFIG; do
    if [[ $config == "h200_gpt_oss_120b" ]]; then IMAGE=$IMAGE buildkite-agent pipeline upload ./.buildkite/h200_gpt-oss-120b.yaml; fi
    if [[ $config == "b200_gpt_oss_120b" ]]; then IMAGE=$IMAGE buildkite-agent pipeline upload ./.buildkite/b200_gpt-oss-120b.yaml; fi
    if [[ $config == "h200_qwen35_397b_a17b" ]]; then IMAGE=$IMAGE buildkite-agent pipeline upload ./.buildkite/h200_qwen35-397b-a17b.yaml; fi
    if [[ $config == "h200_qwen35_397b_a17b_fp8" ]]; then IMAGE=$IMAGE buildkite-agent pipeline upload ./.buildkite/h200_qwen35-397b-a17b-fp8.yaml; fi
done
exit 0