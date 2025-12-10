
# **AtomicVAD Project Structure**



```
AtomicVAD/
├── data/
│   ├── process_vad_data.py
│   ├── prepare_ava_speech.py
│   ├── generate_manifests.py
│   ├── augmentation.py
│   ├── data_loader.py
│   └── manifest/
├── training/
|   └── experiments/
│   ├── train.py
│   ├── config.py
│   └── utils/
├── evaluation/
│   ├── evaluate_spi.py
│   ├── evaluate_swi.py
│   ├── batch_evaluate.py
│   └── eval_utils.py
├── src/
│   └── models/
│       ├── atomicvad.py
│       ├── layers.py
│       └── spec_augment.py
├── models/                # Pretrained Models
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

---