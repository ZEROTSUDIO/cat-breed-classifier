"""
Cat Breed Classifier — Streamlit Web App
Upload a cat photo and get the breed predicted by a MobileNetV2 model.
"""

import json
import os
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cat Breed Identifier",
    page_icon="🐱",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH       = "model/cat_breed_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"
BREED_INFO_PATH  = "model/breed_info.json"
IMG_SIZE         = (224, 224)

# ── Load model & data (cached so it only runs once) ───────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_metadata():
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    with open(BREED_INFO_PATH, encoding="utf-8") as f:
        breed_info = json.load(f)
    return class_names, breed_info

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(img: Image.Image, model, class_names, breed_info, top_k=3):
    img_rgb = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs   = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_idx:
        breed = class_names[idx]
        info  = breed_info.get(breed, {})
        results.append({
            "breed":       breed.replace("_", " ").title(),
            "raw_key":     breed,
            "confidence":  float(probs[idx]),
            "origin":      info.get("origin",      "Unknown"),
            "lifespan":    info.get("lifespan",    "Unknown"),
            "temperament": info.get("temperament", "Unknown"),
            "fun_fact":    info.get("fun_fact",    ""),
        })
    return results

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐱 Cat Breed Identifier")
st.markdown("Upload a photo of your cat — AI will tell you the breed!")
st.divider()

# Check model files exist
if not os.path.exists(MODEL_PATH):
    st.error(
        f"⚠️ Model file not found at `{MODEL_PATH}`.\n\n"
        "Please download the model files from Google Drive and place them in a `model/` folder:\n"
        "```\n"
        "cat-breed-classifier/\n"
        "└── model/\n"
        "    ├── cat_breed_model.h5\n"
        "    ├── class_names.json\n"
        "    └── breed_info.json\n"
        "```"
    )
    st.stop()

# Load model and metadata
model                   = load_model()
class_names, breed_info = load_metadata()

# ── Input: upload or camera ───────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["📤 Upload Photo", "📷 Take Photo"])

uploaded_file = None
camera_file   = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a cat photo",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

with tab_camera:
    camera_file = st.camera_input("Take a photo of your cat")

img_source = uploaded_file or camera_file

# ── Predict ───────────────────────────────────────────────────────────────────
if img_source:
    img = Image.open(img_source)

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img, caption="Your cat 🐾", use_container_width=True)

    with col_results:
        with st.spinner("Analysing breed..."):
            results = predict(img, model, class_names, breed_info)

        st.subheader("Top Predictions")

        for i, r in enumerate(results):
            label = f"{'🥇' if i==0 else '🥈' if i==1 else '🥉'} {r['breed']}"
            st.progress(r["confidence"], text=f"{label} — {r['confidence']*100:.1f}%")

    # ── Breed info card ───────────────────────────────────────────────────────
    st.divider()
    top = results[0]

    st.subheader(f"📖 About the {top['breed']}")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("🌍 Origin",   top["origin"])
    info_col2.metric("⏳ Lifespan", top["lifespan"])
    info_col3.metric("😸 Vibe",     top["temperament"].split(",")[0].strip())

    st.markdown(f"**Temperament:** {top['temperament']}")

    if top["fun_fact"]:
        st.info(f"💡 **Fun fact:** {top['fun_fact']}")

    # ── Confidence table ──────────────────────────────────────────────────────
    with st.expander("View all top-3 details"):
        for r in results:
            st.markdown(
                f"**{r['breed']}** — {r['confidence']*100:.2f}%  \n"
                f"Origin: {r['origin']} · Lifespan: {r['lifespan']}  \n"
                f"Temperament: {r['temperament']}"
            )
            st.divider()

else:
    # Placeholder when nothing is uploaded yet
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 0; color: #888;">
            <div style="font-size: 4rem;">🐱</div>
            <p style="font-size: 1.1rem;">Upload a photo above to identify the breed!</p>
            <p style="font-size: 0.85rem;">Supports JPG, PNG, WEBP · 67 cat breeds</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with TensorFlow · MobileNetV2 · Streamlit · Trained on 67 cat breeds")
