# **AtomicVAD Evaluation Guide**



## 1. Overview

This module provides tools to evaluate AtomicVAD using:

* **SPI** — fast segment-level inference
* **SWI** — sliding-window inference (production-grade)

---

## 2. Directory Structure

```
evaluation/
├── evaluate_spi.py
├── evaluate_swi.py
├── batch_evaluate.py
├── eval_utils.py
```

---

## 3. Evaluation Protocols

### **SPI — Segment-Level Performance Inference**

* 1 prediction per audio segment
* Best for quick experimentation
* Very fast

```bash
python evaluation/evaluate_spi.py \
    --model_path path/to/model.keras \
    --manifest_path data/manifest/ava_test_manifest.json \
    --output_csv results/spi.csv
```

---

### **SWI — Sliding Window Inference**

* Overlapping window predictions
* Best for fine temporal accuracy
* Recommended for publication / deployment

```bash
python evaluation/evaluate_swi.py \
    --model_path models/... \
    --manifest_path data/manifest/ava_test_manifest.json \
    --overlap 0.875 \
    --batch_size 256 \
    --output_csv results/swi.csv
```

---

## 4. Metrics

* AUROC
* F1
* F2
* TPR@FPR=0.315
* Median vs Mean aggregation

---

## 5. Batch Evaluation

```bash
python evaluation/batch_evaluate.py \
    --model_dir experiments/.../checkpoints \
    --manifest_path data/manifest/ava_test_manifest.json \
    --protocol swi \
    --output_csv results/batch_results.csv
```

---

## 6. Troubleshooting

* OOM → reduce `batch_size`
* Slow evaluation → use SPI or reduce overlap
* Constant predictions → check preprocessing pipeline

---