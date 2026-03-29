#!/bin/bash
set -e
cd /Users/ommidazimi/Desktop/ReX

IMAGES="swan wolf elephant lobster leopard snake surfer bus"
MASKING_CONFIGS="resnet50_blur.toml resnet50_median.toml resnet50_circle.toml"

for img in $IMAGES; do
  for cfg in $MASKING_CONFIGS; do
    tag="${img}_${cfg%.toml}"
    echo "=== START: $tag ==="
    poetry run ReX test_images/${img}.jpg \
      --model resnet50.onnx \
      -c $cfg \
      --output outputs/batch/${tag}_expl.png \
      --heatmap outputs/batch/${tag}_heat.png \
      -v 2>&1 | grep -E "classified as|sufficient explanation|Time taken|Error|error|WARNING"
    echo "--- done: $tag ---"
  done
done

ELLIPSE_IMAGES="leopard snake surfer bus"
SPOTLIGHT_CONFIGS="resnet50_circle.toml resnet50_ellipse_h.toml resnet50_ellipse_v.toml"

for img in $ELLIPSE_IMAGES; do
  for cfg in $SPOTLIGHT_CONFIGS; do
    tag="${img}_multi_${cfg%.toml}"
    echo "=== START: $tag ==="
    poetry run ReX test_images/${img}.jpg \
      --model resnet50.onnx \
      -c $cfg \
      --multi 5 \
      --output outputs/batch/${tag}_expl.png \
      --heatmap outputs/batch/${tag}_heat.png \
      -v 2>&1 | grep -E "classified as|sufficient explanation|Time taken|Error|error|WARNING|found a total"
    echo "--- done: $tag ---"
  done
done
