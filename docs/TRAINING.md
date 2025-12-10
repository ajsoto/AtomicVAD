# **AtomicVAD Training Guide**



## 1. Overview

This directory contains all components required to train the AtomicVAD model, including configuration, data loading, augmentation, and reproducibility utilities.

---

## 2. Directory Overview

```
training/
├── train.py              # Main training script (10-fold with different seeds)
├── config.py             # Training configuration
├── utils/
│   ├── gpu.py            # GPU management
│   └── seed.py           # Deterministic seeding
```

---

## 3. Running Training

### **Default Training (10-fold with different seeds)**

```bash
python training/train.py
```

Results stored under:

```
experiments/atomicvad-ggcu/
    ├── checkpoints/fold_*/
    └── backups/fold_*/
```

---

## 4. Configuration

Edit `training/config.py`:

```python
config = TrainingConfig(
    batch_size=256,
    epochs=150,
    initial_lr=0.001,
)
```

---

## 5. Model Architecture Summary

AtomicVAD consists of:

1. Spectrogram extraction
2. SpecAugment + SpecCutout
3. Two convolutional depthwise blocks
4. Skip-connection between blocks
5. Dense classification head

A detailed breakdown is provided in the project's main README.

---

## 6. Training Details

### Optimizer

AdamW + Cosine Decay with Restarts

### Loss

Binary focal cross-entropy (`gamma=2.0`)

### Metrics

* AUROC
* Accuracy
* F1, F2

### Callbacks

* ModelCheckpoint
* EarlyStopping (based on AUC)
* BackupAndRestore
* CSVLogger

---

## 7. Reproducibility

```python
from training.utils.seed import set_global_seed
set_global_seed(42)
```

Enable deterministic ops (optional):

```python
tf.config.experimental.enable_op_determinism()
```

---

## 8. Tips

* Reduce `batch_size` if GPU runs out of memory
* Enable/disable XLA depending on GPU behavior
* Monitor training with TensorBoard or CSV logs

## References

- AVA-Speech: https://research.google.com/ava/
- LibriSpeech: http://www.openslr.org/12/
- MUSAN: https://www.openslr.org/17/
- VoxCeleb: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- SAD-Kaggle: https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets

---