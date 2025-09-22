
import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models

# ─────────── Config ───────────
st.set_page_config(page_title="Water Turbidity Classifier", layout="centered")

CLASSES = [
    "Ultra cloudy",
    "very cloudy",
    "cloudy",
    "lightly cloudy",
    "lightly clear",
    "clear",
]
TURBIDITY_RANGES = {
    "Ultra cloudy":   (3336.0, 3844.0),
    "very cloudy":    (1300.0, 2520.0),
    "cloudy":         (600.0, 1200.0),
    "lightly cloudy": (150.0, 450.0),
    "lightly clear":  (25.0, 90.0),
    "clear":          (1.47, 17.13),
}
MODEL_PATH = "best.pth"

# ─────────── Preprocessing (match training val_tf + gray-world white balance) ───────────
def gray_world(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    mean = arr.reshape(-1, 3).mean(0) + 1e-6
    arr *= mean.mean() / mean
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

preprocess = transforms.Compose([
    transforms.Lambda(lambda im: ImageOps.exif_transpose(im)),
    transforms.Lambda(gray_world),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────── Model builder (ResNet-34 with same head) ───────────
def build_model(num_classes: int = len(CLASSES)) -> nn.Module:
    try:
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    except Exception:
        m = models.resnet34(pretrained=True)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return m

@st.cache_resource(show_spinner=False)
def load_model(path: str) -> nn.Module:
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found. Place best_turbidity_model.pth in this folder.")
        st.stop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASSES)).to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # Remove 'module.' prefix if accidentally saved from DDP
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        st.error("Failed to load model state_dict with exact match. Error:\n" + str(e))
        st.stop()
    model.eval()
    return model

def predict(image: Image.Image, model: nn.Module):
    device = next(model.parameters()).device
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs

# ─────────── UI ───────────
st.title("Water Turbidity Classifier")
st.write("Take or upload a photo of a water sample; the model will estimate its turbidity class and NTU range.")

model = load_model(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    img_file = st.camera_input("📷 Capture water sample")
with col2:
    st.write("— or —")
    img_file2 = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

image = None
if img_file is not None:
    try:
        image = Image.open(img_file).convert("RGB")
    except Exception:
        st.error("Could not read camera image.")
elif img_file2 is not None:
    try:
        image = Image.open(img_file2).convert("RGB")
    except Exception:
        st.error("Could not read uploaded image.")

if image is not None:
    st.image(image, caption="Input image", use_column_width=True)
    pred_class, pred_conf, all_probs = predict(image, model)
    ntu_min, ntu_max = TURBIDITY_RANGES[pred_class]

    st.subheader("Prediction")
    st.markdown(f"**Class:** `{pred_class}`")
    st.markdown(f"**Confidence:** `{pred_conf:.2%}`")
    st.markdown(f"**Estimated Turbidity Range:** `{ntu_min} – {ntu_max} NTU`")

    st.markdown("### Probabilities")
    for cls, p in zip(CLASSES, all_probs):
        prob = float(p)  # ensure native float
        st.write(f"{cls}: **{prob:.2%}**")
        st.progress(min(max(prob, 0.0), 1.0))

    st.caption("Ranges come from your mapping; shoot samples with consistent lighting for best reliability.")
else:
    st.info("Awaiting image input: use the camera or upload a file.")
