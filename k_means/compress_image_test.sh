#!/usr/bin/env bash

# K-Means Image Compression Test Script

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

IMAGE_FILE=$1
STRATEGIES=("f" "m" "x" "b" "k" "g")
K_VALUES=(16)

for strategy in "${STRATEGIES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        cargo run --package k_means --release -- "$IMAGE_FILE" $k $strategy
    done
done

echo "All compressions complete!"
