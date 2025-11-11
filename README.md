# 🧠 AtomicVAD: A Tiny Voice Activity Detection Model for Efficient Inference in Intelligent IoT Systems  

**AtomicVAD** is an ultra-lightweight, end-to-end Voice Activity Detection (VAD) model designed for **resource-constrained microcontrollers** and **IoT systems**.  
With only **0.3k trainable parameters**, it achieves **state-of-the-art accuracy** while maintaining a minimal computational footprint, making it ideal for **TinyML** and **edge AI** applications.  

This repository provides the code, pre-trained models, and scripts used in the paper:  
📘 *"AtomicVAD: A Tiny Voice Activity Detection Model for Efficient Inference in Intelligent IoT Systems."*

---

## 🚀 Key Features

- ⚡ **Ultra-Efficient Architecture** — ~300 parameters, <75 kB Flash, <65 kB SRAM.  
- 🎧 **High Accuracy** — AUROC = 0.903, F2 = 0.891 on AVA-Speech dataset.  
- 🌍 **Real-World Ready** — Validated on **LoRaWAN IoT** nodes and **Arduino Nano 33 BLE Sense**.  
- 🔁 **Oscillatory Activation Function (GGCU)** — Enhances robustness to noise and improves gradient flow.  
- 🧪 **Fully Reproducible Pipelines** — Includes training, inference, and deployment scripts.  

---

## 📂 Repository Structure

```
AtomicVAD/
│
├── models/                # Trained model files
├── src/                   # Core model definition (GGCU, OscilloCore, AtomicVAD)
├── training/              # Training and hyperparameter optimization scripts (BOHB)
├── evaluation/            # SWI & SPI evaluation protocols for AVA-Speech
├── deployment/            # TinyML deployment scripts for microcontrollers (TFLite-Micro)
├── data/                  # Data preprocessing and MFCC extraction scripts
├── notebooks/             # Jupyter notebooks for experimentation and visualization
├── requirements.txt       # Python dependencies
├── LICENSE
└── README.md
```

---

## 🧩 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/AtomicVAD.git
cd AtomicVAD
pip install -r requirements.txt
```

---

## 🎓 Usage

### 1️⃣ Training

Train AtomicVAD using the Speech Commands + FreeSound (SCF) dataset:

```bash
python training/train_atomicvad.py --dataset ./data/SCF --epochs 100
```

### 2️⃣ Evaluation

Evaluate performance using the AVA-Speech dataset with the Sliding-Window Inference (SWI) method:

```bash
python evaluation/evaluate_atomicvad.py --dataset ./data/AVA_Speech --method swi
```

### 3️⃣ Quantization for Microcontrollers

Convert and quantize the trained model to INT8 for deployment:

```bash
python deployment/quantize_model.py --model ./models/atomicvad_fp32.tflite
```

### 4️⃣ Deployment to Arduino Nano 33 BLE Sense

Flash the quantized model and test inference latency:

```bash
python deployment/deploy_arduino.py --port /dev/ttyUSB0
```

---

## 🧠 Model Overview

AtomicVAD is built around the **OscilloCore** module, combining depthwise separable convolutions with the **Generalized Growing Cosine Unit (GGCU)** activation.  
This oscillatory activation captures the periodic nature of speech, allowing shallower networks to maintain high accuracy while minimizing parameters.

| Metric | Value |
|--------|-------|
| Trainable Parameters | ~0.3 k |
| AUROC (AVA-Speech, SWI) | 0.903 |
| F2-Score | 0.891 |
| Flash Memory | 74 kB |
| RAM Usage | 65 kB |
| Inference Latency | 26 ms (Cortex-M7, INT8) |

---

## 🌐 Real-World Validation

AtomicVAD was successfully deployed in a **LoRaWAN IoT network**, demonstrating that on-device VAD reduces transmission latency from **minutes to milliseconds**, enabling efficient edge-based audio intelligence for:

- 🔊 Smart Home Control  
- 🌋 Disaster-Response Sensor Networks  
- 🏥 Remote Health Monitoring  
- 🏭 Industrial IoT Applications  

---

## 📊 Citation

If you use this repository, please cite:

```bibtex
@article{soto2025atomicvad,
  title={AtomicVAD: A Tiny Voice Activity Detection Model for Efficient Inference in Intelligent IoT Systems},
  author={Soto-Vergel, Angelo J. and others},
  journal={Internet of Things},
  year={2025}
}
```

---

## 🤝 Contributing

Contributions, issues, and pull requests are welcome!  
If you improve the model, add new datasets, or optimize inference for other microcontrollers, please open a PR.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 💬 Contact

For research collaboration or technical questions:  
📧 **Angelo Joseph Soto-Vergel**  
🌐 [LinkedIn](https://linkedin.com/in/angelo-joseph-soto-vergel-b851b5a3)  
🔗 [Google Scholar](https://scholar.google.com/citations?user=bSuhGuUAAAAJ)  
