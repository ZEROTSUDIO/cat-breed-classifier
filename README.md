# 🐱 Cat Breed Classifier

> Upload a photo of your cat — AI will tell you the breed!

A deep learning image classifier trained on **67 cat breeds** using Transfer Learning (MobileNetV2). Built as a Google Colab project with a Streamlit web app frontend.

---

## 🚀 Live Demo

> *Coming soon — will be deployed on Streamlit Cloud after training*

---

## 📸 Features

- 🔮 **Top-3 breed predictions** with confidence scores
- 📖 **Breed info cards** — origin, lifespan, temperament, fun facts
- 📷 **Camera input** — take a photo directly in the browser
- 📱 **Mobile friendly** layout

---

## 🏗️ Project Structure

```
cat-breed-classifier/
│
├── cat_breed_classifier.ipynb  ← Google Colab training notebook
├── app.py                      ← Streamlit web app (coming soon)
├── requirements.txt            ← Python dependencies (coming soon)
├── PROJECT_PLAN.md             ← Full project plan & task checklist
└── .gitignore
```

> **Note**: Model files (`.h5`) are stored on Google Drive — too large for GitHub.

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Base Model | MobileNetV2 (pretrained on ImageNet) |
| Dataset | [Cat Breeds — Kaggle](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds) |
| Classes | 67 cat breeds |
| Images | 11,000+ |
| Expected Accuracy | 85–92% |
| Training Platform | Google Colab (T4 GPU, free tier) |

### Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Base (pretrained, then fine-tuned)
    ↓
Global Average Pooling
    ↓
Dense 256 → ReLU → Dropout
    ↓
Dense 67 → Softmax
    ↓
Output: probability per breed
```

### Two-Phase Training

| Phase | Strategy | LR | Epochs |
|---|---|---|---|
| 1 | Feature extraction (base frozen) | `1e-3` | 15 |
| 2 | Fine-tuning (top 30 layers unfrozen) | `1e-5` | 20 |

---

## 🛠️ How to Train (Google Colab)

1. Open `cat_breed_classifier.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Upload your `kaggle.json` API key when prompted
4. Run all cells — takes ~30 minutes
5. Model is saved to your Google Drive automatically

---

## 💻 How to Run the Web App (Coming Soon)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Results

> *Will be updated after training runs*

---

## 🔗 Resources

- [Dataset on Kaggle](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Streamlit Docs](https://docs.streamlit.io)

---

*Built with ❤️ · Google Colab + TensorFlow + Streamlit*
