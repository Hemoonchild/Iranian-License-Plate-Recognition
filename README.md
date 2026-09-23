# Iranian Automatic License Plate Recognition (ALPR) System

An end-to-end, two-stage deep learning pipeline for detecting and recognizing Iranian vehicular license plates (7 digits + 1 Persian letter) in real time.

## Architecture Overview
- **Stage 1 (Detection):** Ultralytics YOLOv8n (96.89% mAP@0.50, 99.67% recall).
- **Stage 2 (Recognition):** Custom 4-stage Residual CNN (ResNet) built from scratch with Skip Connections, BatchNorm, LayerNorm, Dropout, and 8 parallel classification heads.
- **Inference Pipeline:** Cascaded YOLO detection + 3% margin crop padding + multi-head recognition running at **32.4 FPS (30.86 ms/frame)**.

## Key Empirical Results

| Metric | Stage 1 (YOLOv8) | Stage 2 (Custom ResNet) | Pretrained (MobileNetV3) | End-to-End Pipeline |
| :--- | :---: | :---: | :---: | :---: |
| **Detection Recall** | 99.67% | — | — | 99.67% |
| **mAP @ 0.50** | 96.89% | — | — | — |
| **Plate Exact-Match Acc** | — | **92.31%** | 74.90% | **83.33%** |
| **Character Accuracy** | — | **98.26%** | 94.77% | **84.98%** |
| **Character Macro F1** | — | — | — | **89.34%** |
| **Inference Latency** | 2.48 ms | 4.90 ms | 7.79 ms | **30.86 ms (~32 FPS)** |

## Key Findings & Ablations
1. **Residual Connections vs. Plain CNN:** Removing skip connections dropped exact plate accuracy by 8.16% (92.31% -> 84.15%) and caused severe gradient norm degradation.
2. **Scratch Training vs. Transfer Learning:** Custom ResNet (92.31%) outperformed ImageNet-pretrained MobileNetV3 (74.90%) due to domain-specific typographical stroke features and avoiding feature map collapse on rectangular inputs.
3. **Crop Padding Margin:** Adding a 3% margin around YOLO bounding boxes eliminated border-digit clipping and surged end-to-end accuracy from 70.00% to 83.33%.
