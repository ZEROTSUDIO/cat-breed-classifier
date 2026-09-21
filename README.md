# 🐱 Cat Breed Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/ZEROTSUDIOS/cat-breed-classifier)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)

> Upload a photo of your cat — AI will identify the breed with confidence scores and detailed breed characteristics!

A deep learning image classifier trained on **67 cat breeds** using Transfer Learning with **EfficientNetV2S**. Featuring an interactive **Streamlit** web application, automatic weight downloads via **Hugging Face Hub**, and a complete training pipeline for **Google Colab**.

---

## 📸 Features

- 🔮 **Top-3 Breed Predictions**: Displays the most likely breeds with confidence percentage bars.
- 📖 **Comprehensive Breed Cards**: Shows origin, lifespan, temperament, and fun facts for identified breeds.
- 📷 **Dual Input Modes**: Upload an image file (JPG/PNG) or take a photo directly with your device's camera.
- ⚡ **Lightweight & Cloud-Ready**: Model weights (~130 MB) are automatically fetched on-demand from Hugging Face Hub, keeping the Git repository small and fast to clone.
- 📱 **Responsive UI**: Optimized for both mobile and desktop screens.

---

## 🧠 Model Details

| Detail | Specification |
|---|---|
| **Base Architecture** | **EfficientNetV2S** (Pretrained on ImageNet) |
| **Dataset** | [Cat Breeds Dataset — Kaggle](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds) |
| **Classes** | 67 distinct cat breeds |
| **Input Resolution** | 224 × 224 × 3 |
| **Head Architecture** | `GlobalAveragePooling2D` → `Dense(256, ReLU)` → `BatchNormalization` → `Dropout(0.4)` → `Dense(67, Softmax)` |
| **Training Pipeline** | Two-phase transfer learning (Feature extraction + Fine-tuning) |
| **Model Storage** | Hosted on [Hugging Face Hub](https://huggingface.co/ZEROTSUDIOS/cat-breed-classifier) |

---

## 🏗️ Project Structure

```text
cat-breed-classifier/
├── .devcontainer/
│   └── devcontainer.json       # Development container configuration
├── model/
│   ├── breed_info.json         # Breed metadata, temperaments, origins, and fun facts
│   └── class_names.json        # 67 target breed class labels
├── app.py                      # Streamlit web application frontend
├── cat_breed_classifier.ipynb  # Google Colab training notebook
├── generate_notebook.py        # Python script to programmatically build the training notebook
├── requirements.txt            # Application dependencies (TensorFlow, Streamlit, Hugging Face Hub)
├── PROJECT_PLAN.md             # Project roadmap and checklist
└── README.md                   # Project documentation
```

---

## 💻 Running the Web App Locally

### 1. Clone the Repository
```bash
git clone https://github.com/ZEROTSUDIO/cat-breed-classifier.git
cd cat-breed-classifier
```

### 2. Set Up Environment & Install Dependencies
We recommend using Python 3.10 or 3.11:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
> **Note**: On the first run, the app will automatically download the trained model (`cat_breed_model.h5`, ~130 MB) into the `model/` directory from Hugging Face Hub. Subsequent runs will use the cached model.

---

## 🛠️ Model Training (Google Colab)
 
### Option A: 67 Cat Breeds (`cat_breed_classifier.ipynb`)
- **Dataset**: [nikolasgegenava/cat-breeds](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds) (67 classes)
- Broadest breed coverage across common and exotic breeds.

### Option B: 20 Refined Cat Breeds with In-Notebook Upload Demo (`cat_breed_classifier_refined7k.ipynb`)
- **Dataset**: [doctrinek/catbreedsrefined-7k](https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k) (20 balanced classes, 350 images each)
- High accuracy (~85–90%+), uniform class distribution, and faster training.
- **Interactive In-Notebook Testing Demo**: Section 9 features an interactive file uploader (`google.colab.files.upload()`) allowing you to upload any cat picture from your computer to get immediate top-3 breed predictions and rich characteristic cards (origin, temperament, lifespan, and fun facts) without leaving Google Colab!

#### Running in Google Colab:
1. Open either notebook (`cat_breed_classifier.ipynb` or `cat_breed_classifier_refined7k.ipynb`) in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** > **Change runtime type** > **T4 GPU**.
3. Set your Kaggle API token in Section 1 to automatically fetch the dataset.
4. Run all cells through the two-phase training and evaluation pipeline.
5. In `cat_breed_classifier_refined7k.ipynb`, scroll to **Section 9** to test your own cat photos!

---

## 📦 Dependencies

Key libraries used in this project:
- **TensorFlow / Keras** (`>=2.16.0`)
- **Streamlit** (`>=1.30.0`)
- **huggingface_hub** (`>=0.20.0`)
- **Pillow** (`>=10.0.0`)
- **NumPy**

---

## 🔗 Resources & Acknowledgments

- **Dataset**: [Cat Breeds on Kaggle](https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds) by Nikolas Gegenava
- **Model Backbone**: [EfficientNetV2: Smaller Models and Faster Training (Tan & Le, 2021)](https://arxiv.org/abs/2104.00298)
- **Model Repository**: [Hugging Face Hub - ZEROTSUDIOS/cat-breed-classifier](https://huggingface.co/ZEROTSUDIOS/cat-breed-classifier)
- **UI Framework**: [Streamlit](https://streamlit.io/)
