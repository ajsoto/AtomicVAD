# **AtomicVAD — A Tiny Voice Activity Detection Model for Efficient Inference in Intelligent IoT Systems**

AtomicVAD is a lightweight, high-accuracy **Voice Activity Detection (VAD)** system designed for real-time speech detection under noisy and multi-domain conditions. It introduces the **GGCU (Generalized Growing Cosine Unit)** activation and supports both **segment-level** and **streaming-compatible sliding-window inference**. With only 0.3k trainable parameters, it achieves state-of-the-art accuracy while maintaining a minimal computational footprint, making it ideal for TinyML and edge AI applications.

This repository provides the code, pre-trained models, and scripts used in the paper:  
📘 *"AtomicVAD: A Tiny Voice Activity Detection Model for Efficient Inference in Intelligent IoT Systems."*

---

## 🚀 Features

### **Model Architecture**

* GGCU activation for improved gradient flow
* Lightweight convolutional blocks with depthwise separable convolutions
* Skip connections between core blocks
* Integrated SpecAugment and SpecCutout

### **Training Pipeline**

* 10-fold with different seeds
* Mixed-precision training
* Cosine Decay with Restarts scheduling
* Binary focal loss with class balancing
* Full reproducibility utilities

### **Evaluation**

* Two protocols:

  * **SPI** — Simple-Pass inference
  * **SWI** — Sliding-Window inference
* AUROC, F1, F2, TPR@FPR metrics
* Batch evaluation support

### **Datasets Supported**

* AVA-Speech
* Google Speech Commands
* FreeSound
* LibriSpeech
* MUSAN
* VoxCeleb
* MLS (Multilingual LibriSpeech)
* Speech Activity Detection Dataset (Kaggle)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/AtomicVAD.git
cd AtomicVAD
pip install -r requirements.txt
```

Full installation steps → **docs/GETTING_STARTED.md**

---

## 📚 Documentation

| Topic             | File                        |
| ----------------- | --------------------------- |
| Getting Started   | `docs/GETTING_STARTED.md`   |
| Data Preparation  | `docs/DATA_PREPARATION.md`  |
| Training          | `docs/TRAINING.md`          |
| Evaluation        | `docs/EVALUATION.md`        |
| Project Structure | `docs/PROJECT_STRUCTURE.md` |

---

## 🧠 Quick Example: Evaluate a Model

```bash
python evaluation/evaluate_swi.py \
    --model_path models/atomicvad_best.keras \
    --manifest_path manifest/ava_test_manifest.json \
    --output_csv results/swi_ava.csv \
    --overlap 0.875
```

---

## 📈 Expected Performance

| Dataset     | Protocol | AUROC | F1    | F2    |
| ----------- | -------- | ----- | ----- | ----- |
| AVA-Speech  | SWI      | 0.XX+ | 0.XX+ | 0.XX+ |
| LibriSpeech | SWI      | 0.XX+ | 0.XX+ | 0.XX+ |
| MUSAN       | SWI      | 0.XX+ | 0.XX+ | 0.XX+ |

---

## 📝 Citation

```bibtex
@article{SotoVergel2026,
  title = {AtomicVAD: A tiny voice activity detection model for efficient inference in intelligent IoT systems},
  volume = {35},
  ISSN = {2542-6605},
  url = {http://dx.doi.org/10.1016/j.iot.2025.101822},
  DOI = {10.1016/j.iot.2025.101822},
  journal = {Internet of Things},
  publisher = {Elsevier BV},
  author = {Soto-Vergel,  Angelo J. and Sankaran,  Prashant and Velez,  Juan C. and Amaya-Mier,  Rene and Ramirez-Rios,  Diana},
  year = {2026},
  month = jan,
  pages = {101822}
}
```
---
