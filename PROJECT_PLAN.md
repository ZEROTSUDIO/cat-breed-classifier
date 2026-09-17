# 🐱 Cat Breed Classifier — Project Plan

> **Train your own image classifier on 67 cat breeds, then deploy it as a web app.**  
> Stack: Google Colab (free) · TensorFlow/Keras · Streamlit · Google Drive

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Goal** | Classify a cat photo into one of 67 breeds |
| **Dataset** | [Cat Breeds — Kaggle](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds) |
| **Images** | 11,000+ high-quality photos |
| **Classes** | 67 cat breeds |
| **Model** | MobileNetV2 (Transfer Learning) |
| **Training** | Google Colab Free Tier (T4 GPU) |
| **Frontend** | Streamlit Web App |
| **Hosting** | Streamlit Cloud (free) |

---

## 🗂️ Project Structure

```
cat-breed-classifier/
│
├── 📓 cat_breed_classifier.ipynb   ← Colab training notebook
├── 🐍 app.py                       ← Streamlit web app
├── 📋 requirements.txt             ← Python dependencies
├── 📄 README.md                    ← Project readme
│
└── model/                          ← Downloaded from Google Drive
    ├── cat_breed_model.h5          ← Trained model weights
    ├── class_names.json            ← 67 breed label mapping
    └── breed_info.json             ← Breed info cards
```

---

## 🔄 Full Pipeline

```
Kaggle Dataset (42MB)
    ↓
[COLAB] Load + EDA
    ↓
[COLAB] Preprocess + Augment
    ↓
[COLAB] MobileNetV2 Transfer Learning
    ├── Phase 1: Feature Extraction  (~10 min)
    └── Phase 2: Fine-tuning         (~20 min)
    ↓
[COLAB] Evaluate + Save to Google Drive
    ↓
[LOCAL] Streamlit Web App
    └── Upload photo → Get breed + info card
```

---

## 📓 Part 1 — Colab Training Notebook

**File**: `cat_breed_classifier.ipynb`  
**Runtime**: T4 GPU · ~30–35 minutes total

### Section Breakdown

| # | Section | Description | Est. Time |
|---|---|---|---|
| 1 | 🔧 Setup | Install kaggle, mount Drive, download dataset | 3 min |
| 2 | 📂 Load Data | Scan folders, encode labels, train/val/test split | 1 min |
| 3 | 🔍 EDA | Sample images grid, class distribution chart | 2 min |
| 4 | 🧹 Preprocess | Resize 224×224, normalize, augmentation pipeline | 1 min |
| 5 | 🧠 Build Model | MobileNetV2 + custom classification head | 1 min |
| 6 | 🏋️ Train | Phase 1 (frozen base) + Phase 2 (fine-tune) | 20–25 min |
| 7 | 📊 Evaluate | Accuracy curves, confusion matrix, classification report | 2 min |
| 8 | 💾 Save | Export model + class names + breed info to Drive | 1 min |

### Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Base — pretrained on ImageNet (FROZEN in Phase 1)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Dense 256 → ReLU
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense 67 → Softmax
    ↓
Output: probability per breed
```

### Two-Phase Training Strategy

| Phase | Base Model | Learning Rate | Epochs | Expected Val Accuracy |
|---|---|---|---|---|
| **1** Feature Extraction | All frozen | `1e-3` | 15 | ~75–80% |
| **2** Fine-Tuning | Top 30 layers unfrozen | `1e-5` | 20 | ~85–92% |

### Data Augmentation

Applied only on training images to prevent overfitting:

- ↔️ Random horizontal flip
- 🔄 Random rotation ±10°
- 🔍 Random zoom ±10%
- ☀️ Random brightness ±10%
- 🎨 Random contrast ±10%

### Output Files (saved to Google Drive)

| File | Size (approx.) | Purpose |
|---|---|---|
| `cat_breed_model.h5` | ~15 MB | Model weights — loaded by web app |
| `class_names.json` | < 1 KB | Maps index → breed name |
| `breed_info.json` | ~20 KB | Origin, lifespan, temperament, fun facts |
| `training_history.png` | ~200 KB | Accuracy & loss curves (for README) |
| `confusion_matrix.png` | ~300 KB | Confusion matrix heatmap |

---

## 💻 Part 2 — Streamlit Web App

**File**: `app.py`

### Features

- 📤 **Upload a cat photo** (JPG, PNG, WEBP)
- 🔮 **Top-3 breed predictions** with confidence bars
- 📖 **Breed info card** for the top prediction:
  - 🌍 Country of origin
  - ⏳ Lifespan
  - 💬 Temperament
  - 💡 Fun fact
- 📸 **Camera input** option (take photo directly)
- 📱 **Responsive layout** — works on mobile

### UI Layout

```
┌─────────────────────────────────────────────┐
│  🐱 Cat Breed Identifier                     │
│  Upload a photo — we'll tell you the breed! │
├───────────────────┬─────────────────────────┤
│                   │  Top Predictions         │
│   [Cat Photo]     │  ████████ Bengal  87%   │
│                   │  ████     Ocicat  8%    │
│                   │  ██       Savanna 5%    │
├───────────────────┴─────────────────────────┤
│  📖 Bengal                                   │
│  Origin: United States  |  Lifespan: 12–16y │
│  Temperament: Active, Energetic, Playful     │
│  💡 Descended from Asian Leopard Cats!       │
└─────────────────────────────────────────────┘
```

### Tech Stack

```
Python 3.9+
tensorflow / keras     ← load .h5 model & run predictions
streamlit              ← web UI framework
Pillow                 ← image loading & resizing
numpy                  ← array operations
```

---

## 🚀 Part 3 — Deployment

### Option A — Streamlit Cloud (Recommended, Free)

1. Push project to a **GitHub repository**
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub → select repo → deploy
4. Add model files via **GitHub LFS** or load from a public URL

### Option B — Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ✅ Task Checklist

### Colab Notebook
- [x] Notebook generated (`cat_breed_classifier.ipynb`)
- [ ] Upload to Google Colab
- [ ] Run Section 1 — Setup & download dataset
- [ ] Run Section 2 — Load & organise data
- [ ] Run Section 3 — EDA (verify images look correct)
- [ ] Run Section 4 — Preprocessing & augmentation
- [ ] Run Section 5 — Build model
- [ ] Run Section 6 — Phase 1 training
- [ ] Run Section 6 — Phase 2 fine-tuning
- [ ] Run Section 7 — Evaluate results
- [ ] Run Section 8 — Save model to Google Drive
- [ ] Download `cat_breed_model.h5`, `class_names.json`, `breed_info.json`

### Web App
- [ ] Create `app.py` (Streamlit frontend)
- [ ] Create `requirements.txt`
- [ ] Test locally with downloaded model
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud

---

## 📊 Expected Results

| Metric | Expected Value |
|---|---|
| Top-1 Accuracy | 85–92% |
| Top-3 Accuracy | 95–98% |
| Training time (Colab T4) | ~25–35 minutes |
| Model file size | ~15 MB |
| Inference time (per image) | < 1 second |

> **Note**: Results depend on class balance. Rarer breeds with fewer training
> images (e.g. `sokoke`, `oregon_rex`) may have lower per-breed accuracy.
> The confusion matrix will highlight which breeds get mixed up.

---

## 🗓️ Suggested Timeline

| Day | Task |
|---|---|
| Day 1 | Run Colab notebook end-to-end, evaluate results |
| Day 2 | Build and test Streamlit web app locally |
| Day 3 | Polish UI, write README, deploy to Streamlit Cloud |

---

## 🔗 Resources

| Resource | Link |
|---|---|
| Dataset | [kaggle.com/datasets/nikolasgegenava/cat-breeds](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds) |
| Kaggle API token | [kaggle.com → Account → API](https://www.kaggle.com/settings) |
| MobileNetV2 paper | [arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381) |
| Streamlit docs | [docs.streamlit.io](https://docs.streamlit.io) |
| Streamlit Cloud | [streamlit.io/cloud](https://streamlit.io/cloud) |
| TensorFlow Transfer Learning guide | [tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning) |

---

*Generated: September 2026 · License: MIT*
