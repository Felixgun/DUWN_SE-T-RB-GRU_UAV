# DUWN_T-SE-RB-GRU_UAV  
*A Transformer-SE-Residual-BiGRU Model for Gesture Recognition via Distributed UWB Network to Control UAVs*

![Model Architecture](./Tr-SE-Res-Bi-GRU%20Diagram1.png)  
*Figure 1: Model Architecture*

---

## 📘 Overview

**DUWN_T-SE-RB-GRU_UAV** is a deep learning framework designed to perform **gesture recognition** using signals captured by a **Distributed Ultra-Wideband (UWB) Network**, enabling **real-time UAV control**. The system leverages a hybrid architecture combining **Transformer**, **Squeeze-and-Excitation (SE)** modules, **Residual connections**, and a **Bidirectional GRU (BiGRU)** to enhance temporal and spatial signal understanding.

It supports:
- Model **training and testing**
- Real-time **inference on PC**
- Optimized **inference on Jetson Xavier NX**
- Full UAV flight control via **Jetson-based inference**

---

## 🧠 Acronym Breakdown

- **DUWN**: Distributed Ultra-Wideband Network  
- **T-SE**: Transformer with Squeeze-and-Excitation  
- **RB**: Residual Block  
- **GRU**: Gated Recurrent Unit  
- **UAV**: Unmanned Aerial Vehicle

---

## 📂 Repository Contents

| Folder | Description |
|--------|-------------|
| `train_test/` | Scripts for model training and evaluation |
| `inference_pc/` | Inference code runnable on standard PC |
| `inference_nx/` | Inference code optimized for Jetson NX |
| `inference_nx_flight/` | Jetson NX code integrated with UAV flight commands |
| `data/` | Training and testing datasets |
| `assets/` | Images: model diagram, hardware setup, flight result |

---

## 🔧 Installation & Setup

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 1.11
- NumPy, OpenCV, etc.
- Jetson Xavier NX with JetPack SDK (for deployment)

**To install:**

```bash
git clone https://github.com/your-username/DUWN_T-SE-RB-GRU_UAV.git
cd DUWN_T-SE-RB-GRU_UAV
pip install -r requirements.txt
