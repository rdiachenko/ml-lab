#!/usr/bin/env bash

# K-Means Image Compression Test Script

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_file>"
    exit 1
fi

IMAGE_FILE=$1
STRATEGIES=("f" "m" "x" "b" "k" "g")
K_VALUES=(64 16 8 2)

get_strategy_name() {
    case $1 in
        f) echo "Forgy" ;;
        m) echo "MacQueen" ;;
        x) echo "Maximin" ;;
        b) echo "BradleyFayyad" ;;
        k) echo "KmeansPP" ;;
        g) echo "GreedyKmeansPP" ;;
        *) echo "Unknown" ;;
    esac
}

for strategy in "${STRATEGIES[@]}"; do
    for k in "${K_VALUES[@]}"; do
        strategy_name=$(get_strategy_name $strategy)
        echo "Compressing image using strategy: $strategy_name, k: $k"
        cargo run --package k_means --release -- "$IMAGE_FILE" $k $strategy
        echo "Compression complete for $strategy_name, k: $k"
        echo "----------------------------------------"
    done
done

echo "All compressions complete!"
