# DUWBN_SE-T-RB-GRU_UAV  
*A Transformer-SE-Residual-BiGRU Model for Gesture Recognition via Distributed UWB Network to Control UAVs*

![Model Architecture](./SE-T-Res-Bi-GRU%20Diagram1.png)  
*Figure 1: Model Architecture*

---

## 📘 Overview

**DUWBN_SE-T-RB-GRU_UAV** is a deep learning framework designed to perform **gesture recognition** using signals captured by a **Distributed Ultra-Wideband (DUWB) Network**, enabling **real-time UAV control**. The system leverages a hybrid architecture combining  **Squeeze-and-Excitation (SE)** blocks, **Transformer**, **Residual connections**, and a **Bidirectional GRU (BiGRU)** to enhance temporal and spatial signal understanding.

It supports:
- Model **training and testing**
- Real-time **inference on PC**
- Optimized **inference on Jetson Xavier NX**
- Full UAV flight control via **Jetson-based inference**

---

## 🧠 Acronym Breakdown

- **DUWBN**: Distributed Ultra-Wideband Network  
- **SE-T**: Transformer with Squeeze-and-Excitation Block
- **R**: Residual  
- **B**: Bidirectional
- **GRU**: Gated Recurrent Unit  
- **UAV**: Unmanned Aerial Vehicle

---

## 📂 Repository Contents

| Folder | Description |
|--------|-------------|
| `train_and_val.py` | Scripts for model training and evaluation |
| `inference2-PC.py` | Inference code runnable on standard PC |
| `inference-jetson.py` | Inference code optimized for Jetson NX |
| `fly_NX3.py` | Jetson NX code integrated with UAV flight commands |
| `dataset.zip` | Training datasets |
| `test1.zip` | Testing datasets |

---

## 🔧 Installation & Setup

**Requirements:**
- Python ≥ 3.8
- tensorflow = 2.10.1
- NumPy, OpenCV, etc.
- Jetson Xavier NX with JetPack SDK (for deployment)
- Decawave DWM1001




## 🧪 Results

![Flight Result](./flight-path.png)  
*Figure 2: UAV flight path based on gesture input*

---

## 🖼️ System Architecture

![System Setup](./System_diagram.png)  
*Figure 3: Distributed UWB Network and UAV control setup*

---

## 📊 Dataset

Both training and testing datasets are included :
- Collected from multiple UWB anchors and a wearable tag  
- Includes labeled gesture sequences  
- Preprocessed for training
- `inference2.5.py` is used for data collection

---

## 🔬 Model Details

- **Transformer**: Extracts global dependencies from sequential UWB features  
- **SE blocks**: Recalibrate feature maps based on inter-channel relations  
- **Residual blocks**: Increase the gradient signal to obtain a more effective learning
- **BiGRU**: Captures bidirectional temporal dynamics

---

## 📚 References & Related Work

- TBA

---

## 👤 Author

**Your Name**  
- GitHub: [@Felixgun](https://github.com/Felixgun)  
- Email: felix.iniemail@yahoo.com  
- LinkedIn: [https://linkedin.com/in/felixg26/](https://linkedin.com/in/felixg26/)


