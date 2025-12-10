# **AtomicVAD Data Preparation Guide**




## 1. Overview

AtomicVAD training relies on:

* Google Speech Commands (positive samples)
* FreeSound (background/negative samples)
* Evaluation datasets including AVA-Speech, LibriSpeech, MUSAN, VoxCeleb, MLS, etc.

---

## 2. Training Data Preparation

### Command

```bash
python data/process_vad_data.py \
    --speech_data_root=data/raw/google_speech_commands \
    --background_data_root=data/raw/freesound \
    --out_dir=data/manifest \
    --rebalance_method='fixed' \
    --log
```

Generates:

* balanced_speech_training_manifest.json
* balanced_background_training_manifest.json
* validation + test splits

---

## 3. AVA-Speech Preparation

```bash
python data/prepare_ava_speech.py \
    --csv_file data/ava-speech/ava_speech_labels_v1.csv \
    --audio_dir data/ava-speech/audio \
    --output_manifest data/manifest/ava_test_manifest.json \
    --min_duration 0.63 \
    --analyze
```

Supports skipping validation, corruption filtering, etc.

---

## 4. Other Datasets (Manifests)

```bash
python data/generate_manifests.py \
    --dataset librispeech \
    --data_root /path/to/librispeech \
    --splits test-clean dev-clean
```

Repeat for MUSAN, VoxCeleb, MLS, etc.

---

## 5. Manifest Format

```json
{
  "audio_filepath": "path/to/audio.wav",
  "duration": 0.63,
  "label": "speech",
  "offset": 0.0,
  "text": "_"
}
```

---

## 6. Troubleshooting

* Missing files → check absolute paths
* Class imbalance → use `--rebalance_method=fixed`
* Long processing time → reduce stride or enable `--demo`

---