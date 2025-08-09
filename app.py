from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import cv2
import numpy as np
import json

app = FastAPI()

# ====== Load labels ======
with open("labels.json", "r") as f:
    LABELS = json.load(f)

# ====== Load model ======
from torchvision import models
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# ====== Segmentation & Preprocessing ======
def segment_megakaryocyte(img_bgr):
    # HSV thresholds (adjust to your training values)
    nucleus_lower = np.array([120, 40, 40])
    nucleus_upper = np.array([160, 255, 255])
    cytoplasm_lower = np.array([20, 20, 20])
    cytoplasm_upper = np.array([180, 255, 255])

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_nuc = cv2.inRange(img_hsv, nucleus_lower, nucleus_upper)
    mask_cyt = cv2.inRange(img_hsv, cytoplasm_lower, cytoplasm_upper)
    mask = cv2.bitwise_or(mask_nuc, mask_cyt)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest) * 255
    else:
        mask[:] = 0
    return mask

def crop_center_on_mask(img_bgr, mask, pad=16):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img_bgr  # fallback
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = max(x2 - x1 + 1, y2 - y1 + 1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, x2 = cx - w // 2, cx + w // 2
    y1, y2 = cy - w // 2, cy + w // 2
    x1 = max(x1 - pad, 0); y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, img_bgr.shape[1] - 1)
    y2 = min(y2 + pad, img_bgr.shape[0] - 1)
    return img_bgr[y1:y2+1, x1:x2+1]

def preprocess(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    mask = segment_megakaryocyte(img_bgr)
    centered = crop_center_on_mask(img_bgr, mask)

    # Resize + normalize like training
    rgb = cv2.cvtColor(centered, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    rgb = rgb.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb = (rgb - mean) / std
    rgb = np.transpose(rgb, (2, 0, 1))
    return torch.tensor(rgb).unsqueeze(0)

# ====== API ======
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess(image_bytes)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.argmax(probs).item()
    return JSONResponse({
        "prediction": LABELS[top],
        "confidence": float(probs[top])
    })

# Serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")
