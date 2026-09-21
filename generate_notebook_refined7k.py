"""
Generates cat_breed_classifier_refined7k.ipynb — a Google Colab notebook for training
a 20-class cat breed image classifier on the CatBreedsRefined-7k dataset using EfficientNetV2S.

Key Features:
  - Dataset: doctrinek/catbreedsrefined-7k (7,000 images, 20 balanced classes, 350 imgs/class)
  - Backbone: EfficientNetV2S (Transfer Learning + Fine-tuning)
  - Preprocessing: preprocess_input for EfficientNetV2
  - Augmentation: RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, RandomTranslation
  - Evaluation: Full 20x20 Confusion Matrix, Classification Report, Test Accuracy
  - Interactive Demo: Section 9 has an in-notebook image upload widget (google.colab.files.upload)
    that plots side-by-side predictions and rich HTML Breed Information Cards inspired by app.py.
"""

import json

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

# ─────────────────────────────────────────────
# CELLS
# ─────────────────────────────────────────────
cells = []

# ── Title ──────────────────────────────────────────────────────────────
cells.append(md(
"""# 🐱 Cat Breed Classifier (CatBreedsRefined-7k)
### Transfer Learning with EfficientNetV2S · 20 Refined Cat Breeds

**Dataset**: [CatBreedsRefined-7k — Kaggle](https://www.kaggle.com/datasets/doctrinek/catbreedsrefined-7k)  
**Goal**: Train an accurate image classifier on 20 refined, balanced cat breeds (350 images/class, 7,000 total).  
**Interactive Demo**: Includes an in-notebook upload test cell (Section 9) to test your own cat pictures with instant predictions and breed cards!

**Pipeline Highlights**:
- ⚖️ **Curated & Balanced**: 20 distinct breeds, exactly 350 images per class (7,000 images total)
- 🔥 **EfficientNetV2S Backbone**: High-capacity feature extraction with ImageNet pre-trained weights
- ✅ **Proper Preprocessing**: Scaled through `preprocess_input`
- 🎨 **Data Augmentation**: Flips, rotation, zoom, brightness, contrast, and translations
- 📊 **Complete 20x20 Confusion Matrix**: Full readable evaluation across all 20 classes
- 📤 **In-Notebook Upload Demo**: Upload any cat photo directly in Colab and inspect top-3 confidence scores & breed characteristic cards!

---
> ⚠️ **Before running**: Runtime → Change runtime type → **T4 GPU**"""
))

# ── GPU check ──────────────────────────────────────────────────────────
cells.append(code(
"""import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    print("⚠️  No GPU found — go to Runtime > Change runtime type > T4 GPU")"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 🔧 Section 1 — Setup & Installation"))

cells.append(code(
"""# Install kaggle API
!pip install kaggle -q
print("✅ kaggle installed")"""
))

cells.append(code(
"""# Mount Google Drive (model and metadata will be saved here)
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/cat_breed_classifier_refined7k'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 Save directory: {SAVE_DIR}")"""
))

cells.append(code(
"""# ─── Paste your Kaggle API token here ─────────────────────────────────
# Go to kaggle.com → Settings → API → Create New Token
# Copy the token string (starts with KGAT_...) and paste below

import os

KAGGLE_TOKEN = 'KGAT_9740d09bffdde6753ab9223292ebc5e2'  # <-- your token

# Save token to the path kaggle CLI reads automatically
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/access_token', 'w') as f:
    f.write(KAGGLE_TOKEN)
os.chmod('/root/.kaggle/access_token', 0o600)

# Quick test — should print dataset info without errors
!kaggle datasets list --max-size 1 -q 2>&1 | head -3
print("✅ Kaggle API token configured!")"""
))

cells.append(code(
"""# Download and extract the CatBreedsRefined-7k dataset
print("🐱 Downloading CatBreedsRefined-7k dataset (≈464 MB)...")
!kaggle datasets download -d doctrinek/catbreedsrefined-7k --force -q
!unzip -q catbreedsrefined-7k.zip -d /content/catbreedsrefined-7k
print("✅ Dataset ready at /content/catbreedsrefined-7k")
!find /content/catbreedsrefined-7k -maxdepth 3 -type d | head -25"""
))

cells.append(code(
"""# ─── All imports ───────────────────────────────────────────────────────
import os, json, random, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from IPython.display import display, HTML

print(f"TensorFlow  : {tf.__version__}")
print(f"GPU detected: {bool(tf.config.list_physical_devices('GPU'))}")
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 📂 Section 2 — Load & Organise Data"))

cells.append(code(
"""# ─── Scan dataset folders ──────────────────────────────────────────────
DATASET_ROOT = Path('/content/catbreedsrefined-7k')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

paths, labels = [], []

# Walk every subdirectory — collect images grouped by folder name (= breed)
for folder in sorted(DATASET_ROOT.rglob('*')):
    if folder.is_dir():
        imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
        if imgs:
            # Normalise breed label: lowercase, spaces/dashes to underscores
            breed = folder.name.lower().strip().replace(' ', '_').replace('-', '_')
            for img in imgs:
                paths.append(str(img))
                labels.append(breed)

df = pd.DataFrame({'path': paths, 'breed': labels})
print(f"✅ Found {len(df):,} images across {df['breed'].nunique()} breed classes")
print("\\nClass counts:")
print(df['breed'].value_counts().to_string())"""
))

cells.append(code(
"""# ─── Encode labels & split ─────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['breed'])
class_names = list(le.classes_)          # sorted alphabetically
NUM_CLASSES = len(class_names)
print(f"📋 {NUM_CLASSES} classes encoded:")
for i, c in enumerate(class_names):
    print(f"  [{i:2d}] {c}")

# Stratified 80 / 10 / 10 split
train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['label']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
)

print(f"\\n📊 Split sizes:")
print(f"  Train : {len(train_df):,} images ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val   : {len(val_df):,} images ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test  : {len(test_df):,} images ({len(test_df)/len(df)*100:.1f}%)")"""
))

cells.append(code(
"""# ─── Save class names to Drive ─────────────────────────────────────────
class_names_path = os.path.join(SAVE_DIR, 'class_names.json')
with open(class_names_path, 'w', encoding='utf-8') as f:
    json.dump(class_names, f, indent=2)
print(f"✅ class_names.json saved → {class_names_path}")"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — EDA
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 🔍 Section 3 — Explore the Data (EDA)"))

cells.append(code(
"""# ─── Sample images grid (All 20 Breeds) ────────────────────────────────
fig, axes = plt.subplots(4, 5, figsize=(18, 14))
fig.suptitle('🐱 Sample Image for Each of the 20 Cat Breeds', fontsize=18, fontweight='bold', y=1.02)

for ax, breed in zip(axes.flatten(), class_names):
    breed_rows = df[df['breed'] == breed]
    if len(breed_rows) > 0:
        row = breed_rows.sample(1, random_state=42).iloc[0]
        try:
            img = Image.open(row['path']).convert('RGB').resize((180, 180))
            ax.imshow(img)
        except Exception:
            ax.set_facecolor('#ddd')
    ax.set_title(breed.replace('_', ' ').title(), fontsize=10, pad=4, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'sample_breeds.png'), dpi=120, bbox_inches='tight')
plt.show()"""
))

cells.append(code(
"""# ─── Class distribution (Uniform Balance Check) ────────────────────────
counts = df['breed'].value_counts()
avg = counts.mean()

fig, ax = plt.subplots(figsize=(15, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(counts)))
bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(avg, color='crimson', linestyle='--', linewidth=2, label=f'Target Average: {avg:.0f} imgs/class')

ax.set_xticks(range(len(counts)))
ax.set_xticklabels([b.replace('_', ' ').title() for b in counts.index], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Image Count', fontsize=11, fontweight='bold')
ax.set_title('📊 Distribution Across All 20 Cat Breeds', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.15)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 4, f'{int(height)}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'class_distribution.png'), dpi=120, bbox_inches='tight')
plt.show()

print(f"📈 Dataset Statistics:")
print(f"  Total Images : {counts.sum():,}")
print(f"  Total Breeds : {len(counts)}")
print(f"  Min Images   : {counts.min()} ({counts.idxmin()})")
print(f"  Max Images   : {counts.max()} ({counts.idxmax()})")
print(f"  Mean Images  : {avg:.1f}")"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — PREPROCESS & AUGMENT
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 🧹 Section 4 — Preprocess & Data Augmentation"))

cells.append(code(
"""# ─── Config ────────────────────────────────────────────────────────────
IMG_SIZE   = 224   # EfficientNetV2S default resolution
BATCH_SIZE = 32

# ─── Augmentation pipeline (applied ONLY during training) ──────────────
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.15),
    layers.RandomTranslation(0.10, 0.10),
], name='augmentation')

# ─── Image loader ──────────────────────────────────────────────────────
def load_image(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)   # Scales internally for EfficientNetV2
    return img, label

def make_dataset(dataframe, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (dataframe['path'].values, dataframe['label'].values)
    )
    if shuffle:
        ds = ds.shuffle(len(dataframe), seed=42)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, augment=True,  shuffle=True)
val_ds   = make_dataset(val_df,   augment=False, shuffle=False)
test_ds  = make_dataset(test_df,  augment=False, shuffle=False)

print("✅ tf.data pipelines successfully created")
print(f"  Train batches : {len(train_ds)}")
print(f"  Val batches   : {len(val_ds)}")
print(f"  Test batches  : {len(test_ds)}")"""
))

cells.append(code(
"""# ─── Visualise augmentation samples ────────────────────────────────────
sample_path = train_df['path'].iloc[0]
raw = tf.io.read_file(sample_path)
img = tf.image.decode_image(raw, channels=3, expand_animations=False)
img = tf.cast(tf.image.resize(img, [IMG_SIZE, IMG_SIZE]), tf.float32)
img = preprocess_input(img)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('🎨 Augmentation Variations for a Training Sample', fontsize=14, fontweight='bold')

def to_display(t):
    t = (t - t.numpy().min()) / (t.numpy().max() - t.numpy().min() + 1e-7)
    return t.numpy()

axes[0, 0].imshow(to_display(img))
axes[0, 0].set_title('Original Sample', fontweight='bold')
axes[0, 0].axis('off')

for ax in list(axes.flatten())[1:]:
    aug = data_augmentation(tf.expand_dims(img, 0), training=True)[0]
    ax.imshow(to_display(aug))
    ax.axis('off')

plt.tight_layout()
plt.show()"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — BUILD MODEL
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 🧠 Section 5 — Build the Model (Transfer Learning)"))

cells.append(code(
"""# ─── EfficientNetV2S + Custom Classification Head ─────────────────────
def build_model(num_classes: int, img_size: int = 224):
    base = EfficientNetV2S(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="CatBreedClassifier_Refined7k")
    return model, base

model, base_model = build_model(NUM_CLASSES)
model.summary()

total   = model.count_params()
train_p = sum(tf.size(w).numpy() for w in model.trainable_weights)
frozen  = total - train_p
print(f"\\n🔢 Parameter Breakdown:")
print(f"   Total params      : {total:,}")
print(f"   Trainable params  : {train_p:,}  (Classification Head)")
print(f"   Frozen params     : {frozen:,}  (Pretrained EfficientNetV2S)")"""
))

cells.append(code(
"""# ─── Compile Phase 1 ──────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)
print("✅ Model compiled — ready for Phase 1 (Feature Extraction)")"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAIN
# ═══════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 🏋️ Section 6 — Two-Phase Training

| Phase | Base Backbone | Epochs | Learning Rate | Expected Val Accuracy |
|---|---|---|---|---|
| **1: Feature Extraction** | Frozen | 15 | `1e-3` | ~80–85% |
| **2: Fine-Tuning** | Top 80 layers unfrozen | 20 | `5e-6` | ~86–92%+ |"""
))

cells.append(code(
"""# ─── Phase 1 — Feature Extraction (Base Frozen) ───────────────────────
ckpt_p1 = os.path.join(SAVE_DIR, 'best_phase1.h5')

callbacks_p1 = [
    ModelCheckpoint(ckpt_p1, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
]

print("🏋️  Starting Phase 1 — training classification head only...")
print("=" * 60)

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_p1,
    verbose=1
)

best_p1 = max(history_p1.history['val_accuracy'])
print(f"\\n✅ Phase 1 complete! Best val accuracy: {best_p1:.4f} ({best_p1*100:.2f}%)")"""
))

cells.append(code(
"""# ─── Phase 2 — Fine-Tuning (Top 80 Layers Unfrozen) ───────────────────
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

n_trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"🔓 Unfrozen {n_trainable} layers in EfficientNetV2S backbone")
print(f"   Trainable params now: {model.count_params():,}")

# Compile with conservative LR for gentle fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-6),
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

ckpt_final = os.path.join(SAVE_DIR, 'best_model_final.h5')

callbacks_p2 = [
    ModelCheckpoint(ckpt_final, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-9, verbose=1),
]

print("\\n🏋️  Starting Phase 2 — fine-tuning top 80 backbone layers...")
print("=" * 60)

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_p2,
    verbose=1
)

best_p2 = max(history_p2.history['val_accuracy'])
print(f"\\n✅ Phase 2 complete! Best val accuracy: {best_p2:.4f} ({best_p2*100:.2f}%)")"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 📊 Section 7 — Comprehensive Evaluation"))

cells.append(code(
"""# ─── Training History Curves ──────────────────────────────────────────
def plot_history(h1, h2):
    acc      = h1.history['accuracy']     + h2.history['accuracy']
    val_acc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss     = h1.history['loss']         + h2.history['loss']
    val_loss = h1.history['val_loss']     + h2.history['val_loss']
    split    = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Training & Validation Curves across Phases', fontsize=14, fontweight='bold')

    # Accuracy plot
    ax1.plot(acc, label='Train Accuracy', color='#2980b9', linewidth=2)
    ax1.plot(val_acc, label='Val Accuracy', color='#e67e22', linewidth=2)
    ax1.axvline(split, color='crimson', linestyle='--', linewidth=1.5, label=f'Fine-Tuning Start (Ep {split})')
    ax1.set_title('Accuracy History', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss plot
    ax2.plot(loss, label='Train Loss', color='#2980b9', linewidth=2)
    ax2.plot(val_loss, label='Val Loss', color='#e67e22', linewidth=2)
    ax2.axvline(split, color='crimson', linestyle='--', linewidth=1.5, label=f'Fine-Tuning Start (Ep {split})')
    ax2.set_title('Loss History', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=120, bbox_inches='tight')
    plt.show()

plot_history(history_p1, history_p2)"""
))

cells.append(code(
"""# ─── Test Set Evaluation ───────────────────────────────────────────────
print("📊 Evaluating on unseen TEST set (700 images)...")
test_loss, test_acc, test_top3 = model.evaluate(test_ds, verbose=0)

print(f"\\n{'='*45}")
print(f"  Top-1 Accuracy : {test_acc*100:.2f}%")
print(f"  Top-3 Accuracy : {test_top3*100:.2f}%")
print(f"  Test Loss      : {test_loss:.4f}")
print(f"{'='*45}")"""
))

cells.append(code(
"""# ─── Per-Breed Classification Report ───────────────────────────────────
print("📋 Computing classification report across all 20 breeds...")

y_true, y_pred = [], []
for images, labels_batch in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels_batch.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true_names = [class_names[i] for i in y_true]
y_pred_names = [class_names[i] for i in y_pred]

print(classification_report(y_true_names, y_pred_names, zero_division=0))"""
))

cells.append(code(
"""# ─── Full 20x20 Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
labels_display = [b.replace('_', ' ').title() for b in class_names]

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels_display, yticklabels=labels_display,
    linewidths=0.5, linecolor='white', cbar=True
)
plt.title('Confusion Matrix — All 20 Cat Breeds', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('True Breed', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Breed', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.show()"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — SAVE MODEL & METADATA
# ═══════════════════════════════════════════════════════════════════════
cells.append(md("---\n## 💾 Section 8 — Save Model & Breed Characteristics"))

cells.append(code(
"""# ─── Comprehensive Breed Information Dictionary ────────────────────────
BREED_INFO = {
    "abyssinian":          {"origin": "Ethiopia", "lifespan": "9–15 yrs", "temperament": "Active, Curious, Playful, Intelligent", "fun_fact": "Often depicted in ancient Egyptian art — one of the oldest known breeds."},
    "american_bobtail":    {"origin": "United States", "lifespan": "11–15 yrs", "temperament": "Friendly, Intelligent, Adaptable", "fun_fact": "They love to play fetch — very dog-like personality!"},
    "american_curl":       {"origin": "United States", "lifespan": "12–16 yrs", "temperament": "Affectionate, Social, Gentle", "fun_fact": "Born with straight ears that curl backward within days of birth."},
    "american_shorthair":  {"origin": "United States", "lifespan": "15–20 yrs", "temperament": "Easygoing, Adaptable, Gentle", "fun_fact": "One of the most popular cat breeds in the USA, descended from working cats."},
    "american_wirehair":   {"origin": "United States", "lifespan": "14–18 yrs", "temperament": "Adaptable, Affectionate, Playful", "fun_fact": "Their unique wiry coat is a natural mutation first seen in 1966."},
    "balinese":            {"origin": "United States", "lifespan": "12–20 yrs", "temperament": "Vocal, Intelligent, Affectionate", "fun_fact": "Sometimes called the 'long-haired Siamese' — same vocal personality!"},
    "bengal":              {"origin": "United States", "lifespan": "12–16 yrs", "temperament": "Active, Energetic, Curious, Playful", "fun_fact": "Descended from Asian Leopard Cats — wild look, domestic heart."},
    "birman":              {"origin": "Burma (Myanmar)", "lifespan": "12–16 yrs", "temperament": "Gentle, Affectionate, Social", "fun_fact": "Legend says Birmans were sacred temple cats blessed by a golden goddess."},
    "bombay":              {"origin": "United States", "lifespan": "12–18 yrs", "temperament": "Affectionate, Playful, Social", "fun_fact": "Bred to look like a miniature black panther — nicknamed 'parlor panther'."},
    "british_shorthair":   {"origin": "United Kingdom", "lifespan": "12–20 yrs", "temperament": "Calm, Easygoing, Affectionate", "fun_fact": "The Cheshire Cat from Alice in Wonderland was inspired by this breed!"},
    "burmese":             {"origin": "Burma (Myanmar)", "lifespan": "16–18 yrs", "temperament": "Social, Affectionate, Energetic", "fun_fact": "All Burmese worldwide trace back to a single female cat brought to the USA in 1930."},
    "chartreux":           {"origin": "France", "lifespan": "11–15 yrs", "temperament": "Quiet, Gentle, Loyal, Playful", "fun_fact": "French monks at the Grande Chartreuse monastery are said to have bred this cat."},
    "chausie":             {"origin": "United States", "lifespan": "12–14 yrs", "temperament": "Active, Loyal, Intelligent", "fun_fact": "Descended from the Jungle Cat (Felis chaus) — one of the closest to wild cats."},
    "cornish_rex":         {"origin": "Cornwall, UK", "lifespan": "11–15 yrs", "temperament": "Active, Playful, Sociable, Mischievous", "fun_fact": "Has only the soft undercoat — no guard hairs, giving a uniquely curly look."},
    "cymric":              {"origin": "Canada / Isle of Man", "lifespan": "8–14 yrs", "temperament": "Gentle, Playful, Intelligent", "fun_fact": "Essentially a long-haired Manx — they share the taillessness gene."},
    "cyprus":              {"origin": "Cyprus", "lifespan": "12–15 yrs", "temperament": "Adaptable, Friendly, Independent", "fun_fact": "One of the oldest domesticated cat breeds — evidence dates back 9,500 years!"},
    "devon_rex":           {"origin": "Devon, UK", "lifespan": "9–15 yrs", "temperament": "Mischievous, Playful, People-oriented", "fun_fact": "Called 'pixie cats' due to their large ears and elfin face."},
    "donskoy":             {"origin": "Russia", "lifespan": "12–15 yrs", "temperament": "Affectionate, Loyal, Curious", "fun_fact": "Hairlessness is caused by a dominant gene — unlike Sphynx which is recessive."},
    "egyptian_mau":        {"origin": "Egypt", "lifespan": "12–15 yrs", "temperament": "Active, Loyal, Playful", "fun_fact": "The only naturally spotted domestic cat breed — spots are in the coat, not just the skin!"},
    "european_shorthair":  {"origin": "Europe", "lifespan": "15–20 yrs", "temperament": "Adaptable, Playful, Independent", "fun_fact": "The most common cat in Europe, descended from cats the Romans spread across the continent."},
    "exotic_shorthair":    {"origin": "United States", "lifespan": "12–15 yrs", "temperament": "Calm, Gentle, Playful, Affectionate", "fun_fact": "Nicknamed 'the lazy man's Persian' — same look, easier-to-maintain coat."},
    "german_rex":          {"origin": "Germany", "lifespan": "12–15 yrs", "temperament": "Affectionate, Playful, Curious, Social", "fun_fact": "One of the first curly-coated breeds, discovered in Germany after World War II."},
    "havana_brown":        {"origin": "United Kingdom", "lifespan": "10–15 yrs", "temperament": "Affectionate, Curious, Playful, Social", "fun_fact": "The only all-brown cat breed — their rich mahogany color inspired the name 'Havana'."},
    "himalayan":           {"origin": "United States", "lifespan": "9–15 yrs", "temperament": "Calm, Gentle, Sweet, Quiet", "fun_fact": "A Persian × Siamese cross — Persian body with Siamese color points."},
    "japanese_bobtail":    {"origin": "Japan", "lifespan": "15–18 yrs", "temperament": "Active, Vocal, Sociable, Friendly", "fun_fact": "The 'Maneki-neko' lucky waving cat figurine is modeled after this breed!"},
    "karelian_bobtail":    {"origin": "Russia / Finland", "lifespan": "12–15 yrs", "temperament": "Calm, Friendly, Adaptable", "fun_fact": "A rare natural breed from Karelian forests with a pom-pom tail."},
    "khao_manee":          {"origin": "Thailand", "lifespan": "10–12 yrs", "temperament": "Curious, Playful, Affectionate, Social", "fun_fact": "An ancient Thai royal cat — once exclusive to royalty."},
    "korat":               {"origin": "Thailand", "lifespan": "15+ yrs", "temperament": "Gentle, Loyal, Playful, Affectionate", "fun_fact": "In Thailand, Korats are considered good luck and gifted to newlyweds."},
    "korean_bobtail":      {"origin": "Korea", "lifespan": "10–14 yrs", "temperament": "Active, Curious, Friendly, Independent", "fun_fact": "A natural breed from Korea with a distinctive short, pom-pom tail."},
    "kurilian_bobtail":    {"origin": "Russia (Kuril Islands)", "lifespan": "15–20 yrs", "temperament": "Gentle, Social, Playful, Adaptable", "fun_fact": "Natural hunters from the remote Kuril Islands — excellent fishers!"},
    "laperm":              {"origin": "United States", "lifespan": "10–15 yrs", "temperament": "Affectionate, Gentle, Curious, Active", "fun_fact": "Born bald, they grow a unique curly coat — each cat's curls are different!"},
    "lykoi":               {"origin": "United States", "lifespan": "12–15 yrs", "temperament": "Loyal, Playful, Friendly, Curious", "fun_fact": "Called the 'Werewolf Cat' — their patchy coat gives a spooky, Halloween appearance."},
    "maine_coon":          {"origin": "United States (Maine)", "lifespan": "12–15 yrs", "temperament": "Friendly, Playful, Gentle, Dog-like", "fun_fact": "The largest domestic cat breed — males can reach 18 lbs! They love water."},
    "manx":                {"origin": "Isle of Man", "lifespan": "14–16 yrs", "temperament": "Gentle, Playful, Social, Loyal", "fun_fact": "Naturally tailless due to a genetic mutation — can be fully tailless or have a stub."},
    "mekong_bobtail":      {"origin": "Southeast Asia", "lifespan": "12–15 yrs", "temperament": "Affectionate, Loyal, Social, Gentle", "fun_fact": "Named after the Mekong River — historically kept by royalty across Southeast Asia."},
    "munchkin":            {"origin": "United States", "lifespan": "12–15 yrs", "temperament": "Playful, Outgoing, Affectionate, Curious", "fun_fact": "Their short legs are a natural mutation — they can still run surprisingly fast!"},
    "nebelung":            {"origin": "United States", "lifespan": "15–18 yrs", "temperament": "Gentle, Shy, Loyal, Affectionate", "fun_fact": "Named after German mythology ('creature of the mist') — their shimmery coat inspired it."},
    "norwegian_forest_cat":{"origin": "Norway", "lifespan": "12–16 yrs", "temperament": "Independent, Gentle, Adventurous", "fun_fact": "Vikings likely kept these cats on ships to catch mice during long voyages!"},
    "ocicat":              {"origin": "United States", "lifespan": "12–18 yrs", "temperament": "Active, Sociable, Curious, Playful", "fun_fact": "Looks wild but is entirely domestic — created by accident from Siamese × Abyssinian crosses."},
    "oregon_rex":          {"origin": "United States (Oregon)", "lifespan": "12–15 yrs", "temperament": "Playful, Affectionate, Gentle, Social", "fun_fact": "An extremely rare curly-coated breed — now considered an extinct separate variety."},
    "oriental_shorthair":  {"origin": "United Kingdom", "lifespan": "12–15 yrs", "temperament": "Vocal, Sociable, Affectionate, Demanding", "fun_fact": "Comes in over 300 color and pattern combinations — more than any other breed!"},
    "persian":             {"origin": "Iran (Persia)", "lifespan": "12–17 yrs", "temperament": "Gentle, Quiet, Calm, Affectionate", "fun_fact": "Their luxurious coat can grow up to 6 inches long and needs daily brushing."},
    "peterbald":           {"origin": "Russia (St. Petersburg)", "lifespan": "12–15 yrs", "temperament": "Affectionate, Curious, Energetic, Social", "fun_fact": "Created in 1994 by crossing a Donskoy with an Oriental Shorthair."},
    "pixie_bob":           {"origin": "United States", "lifespan": "13–15 yrs", "temperament": "Sociable, Loyal, Gentle, Dog-like", "fun_fact": "Rumored to descend from bobcats — DNA tests show they are fully domestic."},
    "ragamuffin":          {"origin": "United States", "lifespan": "12–16 yrs", "temperament": "Gentle, Calm, Affectionate, Patient", "fun_fact": "They go completely limp when held — nicknamed 'puppy cats'."},
    "ragdoll":             {"origin": "United States", "lifespan": "12–17 yrs", "temperament": "Gentle, Relaxed, Affectionate, Calm", "fun_fact": "Named 'Ragdoll' because they go limp and floppy when you pick them up!"},
    "russian_blue":        {"origin": "Russia", "lifespan": "15–20 yrs", "temperament": "Gentle, Reserved, Loyal, Intelligent", "fun_fact": "Their double coat has silver-tipped guard hairs giving a shimmering blue glow."},
    "safari":              {"origin": "United States", "lifespan": "12–16 yrs", "temperament": "Active, Curious, Affectionate, Energetic", "fun_fact": "A rare hybrid from domestic cats × wild Geoffroy's Cat from South America."},
    "savannah":            {"origin": "United States", "lifespan": "12–20 yrs", "temperament": "Active, Adventurous, Loyal, Curious", "fun_fact": "The tallest domestic cat — crossed with Serval, they can jump over 8 feet high!"},
    "scottish_fold":       {"origin": "Scotland", "lifespan": "11–14 yrs", "temperament": "Adaptable, Sweet, Calm, Gentle", "fun_fact": "All Scottish Folds trace back to one barn cat named Susie, found in Scotland in 1961."},
    "selkirk_rex":         {"origin": "United States", "lifespan": "14–15 yrs", "temperament": "Tolerant, Loving, Playful, Patient", "fun_fact": "Named after the Selkirk Mountains — the only curly-coated breed named after a person!"},
    "serengeti":           {"origin": "United States", "lifespan": "10–15 yrs", "temperament": "Active, Vocal, Sociable, Energetic", "fun_fact": "Bred to look like a Serval — from Bengal × Oriental Shorthair crosses, no wild blood."},
    "siamese":             {"origin": "Thailand", "lifespan": "12–20 yrs", "temperament": "Vocal, Social, Intelligent, Affectionate", "fun_fact": "One of the oldest recognized Asian cat breeds, famous for striking blue almond eyes."},
    "siberian":            {"origin": "Russia", "lifespan": "11–18 yrs", "temperament": "Playful, Affectionate, Adventurous, Loyal", "fun_fact": "Russia's national cat — centuries-old forest breed, and semi-hypoallergenic!"},
    "singapura":           {"origin": "Singapore", "lifespan": "11–15 yrs", "temperament": "Curious, Playful, Affectionate, Sociable", "fun_fact": "The smallest domestic cat breed — adults weigh only 4–8 pounds!"},
    "sokoke":              {"origin": "Kenya", "lifespan": "9–15 yrs", "temperament": "Active, Sociable, Intelligent, Dog-like", "fun_fact": "One of the rarest breeds — discovered living wild in Kenya's Arabuko-Sokoke Forest."},
    "somali":              {"origin": "United States / Canada", "lifespan": "12–16 yrs", "temperament": "Active, Curious, Playful, Sociable", "fun_fact": "Called the 'fox cat' — essentially a long-haired Abyssinian with a fluffy tail."},
    "sphynx":              {"origin": "Canada", "lifespan": "8–14 yrs", "temperament": "Energetic, Mischievous, Warm, Social", "fun_fact": "Despite looking hairless, they're covered in fine peach-fuzz and feel like warm suede!"},
    "thai":                {"origin": "Thailand", "lifespan": "12–16 yrs", "temperament": "Vocal, Social, Intelligent, Curious", "fun_fact": "The traditional round-faced Siamese, preserved by Thai cat fanciers."},
    "tonkinese":           {"origin": "Canada", "lifespan": "14–16 yrs", "temperament": "Playful, Social, Affectionate, Vocal", "fun_fact": "A Siamese × Burmese cross — they inherited the best traits of both!"},
    "toyger":              {"origin": "United States", "lifespan": "10–15 yrs", "temperament": "Relaxed, Friendly, Intelligent, Easygoing", "fun_fact": "Bred to look like a miniature tiger — 'Toyger' = 'Toy Tiger'. Still being developed!"},
    "turkish_angora":      {"origin": "Turkey (Ankara)", "lifespan": "12–18 yrs", "temperament": "Playful, Affectionate, Intelligent, Energetic", "fun_fact": "The Ankara Zoo has a special program to preserve this ancient Turkish breed."},
    "turkish_van":         {"origin": "Turkey (Lake Van)", "lifespan": "12–17 yrs", "temperament": "Active, Playful, Independent, Affectionate", "fun_fact": "Known as 'the swimming cat' — they genuinely love water and are strong swimmers!"},
    "ukrainian_levkoy":    {"origin": "Ukraine", "lifespan": "12–15 yrs", "temperament": "Gentle, Sociable, Curious, Playful", "fun_fact": "Created in Ukraine in 2004 — a Donskoy × Scottish Fold cross with folded ears and no fur."},
    "ural_rex":            {"origin": "Russia (Ural Mountains)", "lifespan": "12–15 yrs", "temperament": "Gentle, Calm, Sociable, Loyal", "fun_fact": "A natural curly breed from the Ural Mountains, independently evolved from other Rex breeds."},
    "vankedisi":           {"origin": "Turkey", "lifespan": "12–17 yrs", "temperament": "Playful, Active, Independent, Curious", "fun_fact": "The all-white version of the Turkish Van — 'Vankedisi' means 'Van cat' in Turkish."},
}

breed_info_path = os.path.join(SAVE_DIR, 'breed_info.json')
with open(breed_info_path, 'w', encoding='utf-8') as f:
    json.dump(BREED_INFO, f, indent=2, ensure_ascii=False)
print(f"✅ breed_info.json saved → {breed_info_path}")"""
))

cells.append(code(
"""# ─── Save Final Trained Model ──────────────────────────────────────────
final_model_path = os.path.join(SAVE_DIR, 'cat_breed_model_refined7k.h5')
model.save(final_model_path)
print(f"✅ Model saved → {final_model_path}")

# Verify model reload
model_reloaded = keras.models.load_model(final_model_path)
loss, acc, top3 = model_reloaded.evaluate(test_ds, verbose=0)
print(f"🔍 Reload validation — Test accuracy: {acc*100:.2f}% | Top-3: {top3*100:.2f}%")"""
))

cells.append(code(
"""# ─── Core Prediction Helper Function ───────────────────────────────────
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_preprocess

def predict_breed(image_input, model, class_names, breed_info, top_k=3):
    \"\"\"
    Predict cat breed from a PIL Image or file path.
    Returns: list of dicts with breed, confidence, and metadata.
    \"\"\"
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    img_resized = img.resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32)
    arr = eff_preprocess(arr)
    arr = np.expand_dims(arr, axis=0)

    probs   = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_idx:
        raw_breed = class_names[idx]
        norm_breed = raw_breed.lower().strip().replace(' ', '_').replace('-', '_')
        info = breed_info.get(norm_breed, {})
        results.append({
            'raw_name':    raw_breed,
            'breed':       raw_breed.replace('_', ' ').title(),
            'confidence':  float(probs[idx]),
            'origin':      info.get('origin', 'Unknown'),
            'lifespan':    info.get('lifespan', 'Unknown'),
            'temperament': info.get('temperament', 'Friendly, Curious, Loyal'),
            'fun_fact':    info.get('fun_fact', 'A wonderful and unique feline companion!'),
        })
    return results

print("✅ predict_breed() function defined and ready!")"""
))

# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — IN-NOTEBOOK UPLOAD DEMO
# ═══════════════════════════════════════════════════════════════════════
cells.append(md(
"""---
## 🐾 Section 9 — Interactive Cat Image Upload & Testing Demo

Upload any cat photo from your computer (JPG, PNG, WEBP) to test the trained model directly in Google Colab!  
The cell will display:
1. **Side-by-Side Visual Plot**: Your uploaded image alongside a top-3 horizontal confidence bar chart.
2. **Rich Breed Information Card**: Formatted card displaying Origin, Lifespan, Temperament, and Fun Facts (styled like the web app)."""
))

cells.append(code(
"""# ─── Upload Cat Photo & Run Inference ──────────────────────────────────
import io
from IPython.display import display, HTML, clear_output

print("📤 Click 'Choose Files' below to upload your cat picture (.jpg, .png, .webp)...\\n")

# Colab upload dialog (with graceful fallback for local notebooks)
try:
    from google.colab import files
    uploaded = files.upload()
except ImportError:
    print("ℹ️  Running locally (outside Colab). Provide image path:")
    path_input = input("Enter path to cat image: ").strip('"\\\' ')
    if path_input and os.path.exists(path_input):
        with open(path_input, 'rb') as f:
            uploaded = {os.path.basename(path_input): f.read()}
    else:
        uploaded = {}

if not uploaded:
    print("⚠️  No image file was uploaded. Please run the cell again to choose an image.")
else:
    for filename, content in uploaded.items():
        try:
            pil_img = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as e:
            print(f"❌ Could not decode image '{filename}': {e}")
            continue

        # Run model inference
        predictions = predict_breed(pil_img, model, class_names, BREED_INFO, top_k=3)
        top1 = predictions[0]

        # ── 1. Matplotlib Plot (Image + Horizontal Top-3 Confidence Bars) ────
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.2]})
        fig.suptitle(f"🐱 Breed Identification for: {filename}", fontsize=15, fontweight='bold', y=0.98)

        # Left: Original Image
        ax_img.imshow(pil_img)
        match_color = '#27ae60' if top1['confidence'] >= 0.50 else '#d35400'
        ax_img.set_title(
            f"Top Match: {top1['breed']} ({top1['confidence']*100:.1f}%)",
            fontsize=12, fontweight='bold', color=match_color, pad=8
        )
        ax_img.axis('off')

        # Right: Horizontal Bars
        breed_labels = [p['breed'] for p in predictions]
        conf_values  = [p['confidence'] * 100 for p in predictions]
        bar_colors   = ['#2ecc71', '#3498db', '#f39c12']

        y_positions = range(len(predictions))[::-1]
        bars = ax_bar.barh(y_positions, conf_values, color=bar_colors, height=0.45, edgecolor='white', linewidth=1.2)
        ax_bar.set_yticks(y_positions)
        ax_bar.set_yticklabels(breed_labels, fontsize=11, fontweight='bold')
        ax_bar.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
        ax_bar.set_title('Top-3 Breed Probabilities', fontsize=12, fontweight='bold', pad=8)
        ax_bar.set_xlim(0, 100)
        ax_bar.grid(axis='x', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, conf_values):
            ax_bar.text(
                bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold', color='#2c3e50'
            )

        plt.tight_layout()
        plt.show()

        # ── 2. Rich HTML Breed Information Card (matching app.py aesthetics) ─
        badge_bg = '#27ae60' if top1['confidence'] >= 0.60 else ('#f39c12' if top1['confidence'] >= 0.35 else '#7f8c8d')
        other_breeds_html = " &nbsp;|&nbsp; ".join(
            [f"#{i+1} <b>{p['breed']}</b> ({p['confidence']*100:.1f}%)" for i, p in enumerate(predictions[1:])]
        )

        card_html = f\"\"\"
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    max-width: 680px; border-radius: 12px; border: 1px solid #e2e8f0;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.06); padding: 22px; margin: 16px 0;
                    background: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center;
                        border-bottom: 2px solid #f1f5f9; padding-bottom: 14px; margin-bottom: 16px;">
                <div>
                    <h2 style="margin: 0; color: #0f172a; font-size: 22px; font-weight: 700;">🐾 {top1['breed']}</h2>
                    <span style="color: #64748b; font-size: 13px;">Top Model Prediction</span>
                </div>
                <span style="background: {badge_bg}; color: #ffffff; padding: 6px 14px;
                            border-radius: 20px; font-weight: bold; font-size: 14px; letter-spacing: 0.3px;">
                    {top1['confidence']*100:.1f}% Confidence
                </span>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">
                <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">🌍 Origin</div>
                    <div style="color: #1e293b; font-size: 15px; font-weight: 600; margin-top: 3px;">{top1['origin']}</div>
                </div>
                <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 12px; border-radius: 8px;">
                    <div style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">⏳ Lifespan</div>
                    <div style="color: #1e293b; font-size: 15px; font-weight: 600; margin-top: 3px;">{top1['lifespan']}</div>
                </div>
            </div>

            <div style="background: #f8fafc; border: 1px solid #f1f5f9; padding: 12px; border-radius: 8px; margin-bottom: 14px;">
                <div style="font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase;">🎭 Temperament</div>
                <div style="color: #1e293b; font-size: 14px; font-weight: 500; margin-top: 4px;">{top1['temperament']}</div>
            </div>

            <div style="background: #eff6ff; border-left: 4px solid #3b82f6; padding: 14px; border-radius: 6px; margin-bottom: 14px;">
                <div style="font-size: 13px; font-weight: 700; color: #1d4ed8;">💡 Fun Fact</div>
                <div style="color: #1e3a8a; font-size: 14px; margin-top: 4px; line-height: 1.5;">{top1['fun_fact']}</div>
            </div>

            <div style="border-top: 1px solid #f1f5f9; padding-top: 12px; font-size: 13px; color: #64748b;">
                <strong>Alternative Matches:</strong> {other_breeds_html}
            </div>
        </div>
        \"\"\"
        display(HTML(card_html))"""
))

# ── Conclusion ────────────────────────────────────────────────────────
cells.append(md(
"""---
## 🎉 Summary & Next Steps

All training outputs and model weights have been exported to Google Drive:
- `cat_breed_model_refined7k.h5`: Trained EfficientNetV2S weights (~130 MB)
- `class_names.json`: The 20 breed category labels
- `breed_info.json`: Breed origins, lifespan, temperament, and fun facts
- `training_history.png` & `confusion_matrix.png`: Model performance graphs

To test another cat image at any time, simply re-run **Section 9**!"""
))

# ─────────────────────────────────────────────
# ASSEMBLE NOTEBOOK
# ─────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "collapsed_sections": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

output_path = "cat_breed_classifier_refined7k.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook successfully written to: {output_path}")
print(f"     Total cells: {len(cells)}")
