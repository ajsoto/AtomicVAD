# **Getting Started with AtomicVAD**



## 1. Installation

```bash
git clone https://github.com/yourusername/AtomicVAD.git
cd AtomicVAD
pip install -r requirements.txt
```

Optional: Conda environment included.

---

## 2. Prepare Training Data

```bash
python data/process_vad_data.py \
    --speech_data_root=data/raw/google_speech_commands \
    --background_data_root=data/raw/freesound \
    --out_dir=data/manifest \
    --rebalance_method='fixed'
```

---

## 3. Train Model

```bash
python training/train.py
```

---

## 4. Evaluate Model

SPI:

```bash
python evaluation/evaluate_spi.py ...
```

SWI:

```bash
python evaluation/evaluate_swi.py ...
```

---