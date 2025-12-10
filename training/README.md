# AtomicVAD Training

This directory contains the training scripts and utilities for the AtomicVAD model.

## Directory Structure

```
training/
├── train.py              # Main training script
├── config.py             # Training configuration
├── data/
│   ├── data_loader.py    # Data loading utilities
│   └── augmentation.py   # Audio augmentation functions
├── model/
│   ├── atomicvad.py      # Model architecture
│   ├── layers.py         # Custom layers (AGCU, Spectrogram, etc.)
│   └── spec_augment.py   # SpecAugment implementation
├── utils/
│   ├── seed.py           # Random seed utilities
│   └── gpu.py            # GPU configuration
└── README.md             # This file
```

## Quick Start

### Basic Training

```bash
python training/train.py
```

This will train the AtomicVAD model using the default configuration with 10-fold seeds.

### Configuration

Edit `config.py` to modify training parameters:

```python
config = TrainingConfig(
    batch_size=256,
    epochs=150,
    initial_lr=0.001,
    # ... other parameters
)
```

## Model Architecture

The AtomicVAD model consists of:

1. **Spectrogram Extraction**: Converts raw audio to MFCC/Mel spectrograms
2. **Data Augmentation**: SpecAugment and SpecCutout for regularization
3. **Core Blocks**: Two core processing blocks with depthwise convolutions
4. **Skip Connection**: Residual connection from Block 1 to Block 2
5. **Classification Head**: Dense layer with softmax activation

### Core Block Architecture

```
Input
  ↓
DepthwiseConv2D
  ↓
MaxPooling2D
  ↓
Normalization (Layer/Batch)
  ↓
Conv2D (1x1)
  ↓
GGCU Activation
  ↓
Dropout
  ↓
Conv2D (1x1)
  ↓
Output
```

## Key Features

### GGCU Activation Function

The Generalized Growing Cosine Unit (GGCU) is a custom activation function:

```
GGCU(x) = (w1*x + b1) * cos(w2*x + b2)
```

where `w1, w2` and `b1, b2` are learnable parameters with regularization.

### Data Augmentation

**Audio-level augmentation:**
- Time shift perturbation (±5ms)
- White noise addition with varying SNR
- Background noise mixing

**Spectrogram-level augmentation:**
- SpecAugment: Time and frequency masking
- SpecCutout: Random rectangular cutouts

### Mixed Precision Training

The training uses mixed precision (float16/float32) for improved performance on modern GPUs:

```python
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

## Training Details

### Optimizer

AdamW optimizer with Cosine Decay with Restarts learning rate schedule:
- Initial learning rate: 0.001
- First decay steps: 10% of total steps
- T_mul: 2.0 (cycle length multiplier)
- M_mul: 1.0 (learning rate multiplier)

### Loss Function

Binary Focal Cross-Entropy with:
- Class balancing enabled
- Gamma: 2.0
- Applied to 2-class output (speech vs. background)

### Metrics

- AUC (Area Under Curve)
- Binary Accuracy
- F1 Score (threshold: 0.5)
- F2 Score (beta: 2.0, threshold: 0.5)

### Callbacks

- **ModelCheckpoint**: Saves best model based on validation AUC
- **BackupAndRestore**: Enables training resumption after interruption
- **EarlyStopping**: Stops training if validation AUC doesn't improve for 20 epochs
- **CSVLogger**: Logs training metrics to CSV file

## Cross-Validation

The training script performs 10-fold with different random seeds:

```python
seeds = [43185, 86648, 92813, 89816, 69375, 47845, 10357, 37555, 79511, 78417]
```

Each fold produces:
- Checkpoint files: `experiments/atomicvad-ggcu/checkpoints/fold_{i}/`
- Training logs: `experiments/atomicvad-ggcu/checkpoints/fold_{i}/training_log.csv`
- Backup files: `experiments/atomicvad-ggcu/backups/fold_{i}/`

## Output

After training, the script outputs:
- Best model weights for each fold
- Training history (loss, metrics per epoch)
- Test set evaluation results per fold
- Aggregate statistics (mean ± std) across all folds

Example output:

```
Aggregate Test Results Across All Folds:
============================================================

loss: 0.1234 ± 0.0056
auc: 0.9876 ± 0.0023
binary_accuracy: 0.9456 ± 0.0034
f1_score: 0.9423 ± 0.0041
f2_score: 0.9401 ± 0.0038
```

## Data Format

The training expects JSON manifest files with the following structure:

```json
{
  "audio_filepath": "path/to/audio.wav",
  "duration": 0.63,
  "label": "speech",
  "offset": 0.0
}
```

### Required Manifest Files

- `balanced_speech_training_manifest.json`
- `balanced_background_training_manifest.json`
- `balanced_speech_validation_manifest.json`
- `balanced_background_validation_manifest.json`
- `balanced_speech_testing_manifest.json`
- `balanced_background_testing_manifest.json`

## Requirements

See `requirements.txt` in the project root.

Key dependencies:
- TensorFlow >= 2.13
- NumPy
- Python >= 3.9

## GPU Configuration

The training automatically configures GPUs with memory growth enabled to avoid OOM errors:

```python
from utils.gpu import configure_gpu
configure_gpu()
```

## Reproducibility

Random seeds are set for:
- NumPy
- TensorFlow
- Python's random module
- Dataset shuffling
- Layer initialization

For full determinism (at the cost of performance), you can enable:

```python
tf.config.experimental.enable_op_determinism()
```

## Tips

1. **Memory Issues**: Reduce `batch_size` in `config.py`
2. **Faster Training**: Enable XLA compilation with `jit_compile=True` (already enabled)
3. **Monitoring**: Use TensorBoard with the CSV logs
4. **Resume Training**: Keep backup directories intact to resume interrupted training

## Citation

If you use this training code, please cite:

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