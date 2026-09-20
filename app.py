"""
Cat Breed Classifier — Streamlit Web App
Upload a cat photo and get the breed predicted by an EfficientNetV2S model.
Model weights are downloaded automatically from Hugging Face Hub on first run.
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
HF_REPO_ID       = "ZEROTSUDIOS/cat-breed-classifier"   # ← your HF repo
MODEL_FILENAME   = "cat_breed_model.h5"
MODEL_PATH       = os.path.join("model", MODEL_FILENAME)
CLASS_NAMES_PATH = "model/class_names.json"
BREED_INFO_PATH  = "model/breed_info.json"
IMG_SIZE         = (224, 224)

# ── Download model from Hugging Face if not cached locally ───────────────────
def ensure_model_downloaded():
    """Download the model from HF Hub if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        from huggingface_hub import hf_hub_download
        os.makedirs("model", exist_ok=True)
        with st.spinner("⬇️ Downloading model from Hugging Face (~130 MB, first run only)..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir="model",
                local_dir_use_symlinks=False,
            )
        st.success("✅ Model downloaded!")

# ── Load model & data (cached so it only runs once) ───────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input  # noqa: F401

    ensure_model_downloaded()

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception:
        # Fallback: rebuild EfficientNetV2S architecture and load weights
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            num_classes = len(json.load(f))
        base = tf.keras.applications.EfficientNetV2S(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None,
        )
        base.trainable = False
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.40)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.30)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.load_weights(MODEL_PATH)
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
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

    img_rgb    = img.convert("RGB").resize(IMG_SIZE)
    img_rgb_np = np.array(img_rgb, dtype=np.uint8)          # keep original for overlay

    arr = preprocess_input(img_rgb_np.astype(np.float32))   # ✅ correct scaling
    arr = np.expand_dims(arr, axis=0)                        # (1, 224, 224, 3)

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
    return results, arr, img_rgb_np, int(top_idx[0])

# ── Grad-CAM helpers ──────────────────────────────────────────────────────────
def _get_last_conv_layer(model):
    """Return the last Conv2D layer, searching inside nested sub-models first."""
    import tensorflow as tf
    for layer in model.layers:
        if hasattr(layer, "layers"):                      # sub-model (e.g. EfficientNetV2S)
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub
    for layer in reversed(model.layers):                  # fallback: flat model
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def compute_grad_cam(model, preprocessed_arr, img_rgb_np, class_idx):
    """
    Generate a Grad-CAM heatmap blended onto the original image.

    Parameters
    ----------
    preprocessed_arr : np.ndarray, shape (1, H, W, 3) — model-ready input
    img_rgb_np       : np.ndarray, shape (H, W, 3) uint8 — original RGB for overlay
    class_idx        : int — target class for gradient computation

    Returns
    -------
    np.ndarray (H, W, 3) uint8 overlay, or None on failure
    """
    import cv2
    import tensorflow as tf

    last_conv = _get_last_conv_layer(model)
    if last_conv is None:
        return None

    # Sub-model that exposes both the last conv activations and final logits
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )

    img_tensor = tf.cast(preprocessed_arr, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]               # scalar for target class

    grads       = tape.gradient(loss, conv_outputs)    # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_outputs[0]                         # (h, w, C)
    heatmap  = conv_out @ pooled_grads[..., tf.newaxis]   # (h, w, 1)
    heatmap  = tf.squeeze(heatmap)                     # (h, w)
    heatmap  = tf.nn.relu(heatmap)
    heatmap  = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match the display image
    h, w = img_rgb_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap_np, (w, h))

    # Jet colormap (blue → green → red)
    heatmap_u8  = np.uint8(255 * heatmap_resized)
    colored     = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Blend: 45 % heatmap + 55 % original
    overlay = (colored_rgb * 0.45 + img_rgb_np * 0.55).astype(np.uint8)
    return overlay

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐱 Cat Breed Identifier")
st.markdown("Upload a photo of your cat — AI will tell you the breed!")
st.divider()

# Check JSON metadata exists (model will be auto-downloaded)
if not os.path.exists(CLASS_NAMES_PATH) or not os.path.exists(BREED_INFO_PATH):
    st.error(
        "⚠️ Metadata files not found. Please make sure `model/class_names.json` "
        "and `model/breed_info.json` are present."
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
            results, preprocessed_arr, img_rgb_np, top_class_idx = predict(
                img, model, class_names, breed_info
            )

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

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔥 What the AI saw — Grad-CAM")
    st.caption(
        "Highlights the regions that most influenced the prediction. "
        "🔴 Red = highly important · 🟡 Yellow = moderately important · 🔵 Blue = less important"
    )

    with st.spinner("Generating heatmap..."):
        cam_overlay = compute_grad_cam(model, preprocessed_arr, img_rgb_np, top_class_idx)

    if cam_overlay is not None:
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.image(img_rgb_np, caption="Original image", use_container_width=True)
        with cam_col2:
            st.image(cam_overlay, caption=f"Grad-CAM — {top['breed']}", use_container_width=True)
    else:
        st.warning("⚠️ Could not generate Grad-CAM heatmap for this model.")

    # ── Confidence table ──────────────────────────────────────────────────────
    st.divider()
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
st.caption("Built with TensorFlow · EfficientNetV2S · Streamlit · Trained on 67 cat breeds")
