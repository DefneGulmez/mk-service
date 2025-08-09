import gradio as gr
import torch
import torch.nn as nn
import cv2
import numpy as np
import json

# ===== Load labels =====
LABELS = ["Type1", "Type2", "Type3", "Type4", "Type5", "Type6", "Type7"]  # change to your actual labels

# ===== Load model =====
from torchvision import models
model = torch.load("best_model.pth", map_location="cpu", weights_only=False)
model.eval()

# ===== Segmentation functions =====
def segment_megakaryocyte(img_bgr):
    nucleus_lower = np.array([120, 40, 40])
    nucleus_upper = np.array([160, 255, 255])
    cytoplasm_lower = np.array([20, 20, 20])
    cytoplasm_upper = np.array([180, 255, 255])
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_nuc = cv2.inRange(img_hsv, nucleus_lower, nucleus_upper)
    mask_cyt = cv2.inRange(img_hsv, cytoplasm_lower, cytoplasm_upper)
    mask = cv2.bitwise_or(mask_nuc, mask_cyt)
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
        return img_bgr
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

# ===== Prediction pipeline =====
def predict_image(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = segment_megakaryocyte(img_bgr)
    centered = crop_center_on_mask(img_bgr, mask)
    rgb = cv2.cvtColor(centered, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    rgb = rgb.astype(np.float32) / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    rgb = (rgb - mean) / std
    rgb = np.transpose(rgb, (2, 0, 1))
    x = torch.tensor(rgb).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.argmax(probs).item()
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# ===== Gradio Interface =====
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=7),
    title="Megakaryocyte Classifier",
    description="Upload an image of a blood smear containing a megakaryocyte to classify it into 1 of 7 types."
)

if __name__ == "__main__":
    demo.launch()
