# 🌱 Crop Survival Rate Prediction System

This project predicts the **survival suitability of crops** based on environmental and soil parameters using **unsupervised machine learning**, specifically **One-Class Support Vector Machines (OCSVM)**.

Since the dataset contains **only valid (survived) crop conditions**, a traditional supervised model cannot be used. Instead, this system learns the *normal growing conditions* of each crop and estimates survival likelihood for new inputs.


## 🚀 Project Overview

- **Problem Type**: Unsupervised Learning
- **Model Used**: One-Class SVM (per crop)
- **Approach**:
  - Train one OCSVM model per crop using only healthy/survived data
  - Detect how close a new input is to ideal crop conditions
  - Convert anomaly score into a **Survival Rate (%)**


<<<<<<< HEAD
## 📂 Project Structure
```
Cultivated
│
├── dataset.csv # Original crop dataset
├── model.ipynb # Training & experimentation notebook
├── gui_app.py # GUI application for prediction
│
├── trained_models/ # Saved models and scalers
│ ├── apple_ocsvm.pkl
│ ├── apple_scaler.pkl
│ ├── banana_ocsvm.pkl
│ ├── banana_scaler.pkl
│ └── ... (other crops)
│
└── README.md
└── requirements.txt
```

## 📊 Dataset Description

The dataset contains **only survived crop records** with the following features:

- Temperature
- Humidity
- Rainfall
- Soil nutrients (N, P, K)
- Crop name



## 🧠 Model Explanation (One-Class SVM)

- The model learns **normal/healthy growth conditions**
- New input is checked against learned boundaries
- Output:
  - `+1` → Suitable / survived
  - `-1` → Anomalous / poor survival condition

### Survival Rate Formula (Conceptual)

Survival Rate (%) = (decision_score - score_min)/ (score_max - score_min) × 100

Higher distance → better survival suitability.



## 🖥️ GUI Application

The `gui_app.py` allows users to:
- Select a crop
- Enter environmental conditions
- Get predicted **survival rate (%)**

Each crop uses:
- Its own trained OCSVM model
- Its own MinMax scaler



## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd Cultivated
2️⃣ Install Dependencies
bash

pip install -r requirements.txt
3️⃣ Run the GUI App
bash

python gui_app.py
