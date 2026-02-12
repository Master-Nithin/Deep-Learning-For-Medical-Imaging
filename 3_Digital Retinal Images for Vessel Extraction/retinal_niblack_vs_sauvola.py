import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Get script directory
base_dir = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(base_dir, "Image.tif")
mask_path = os.path.join(base_dir, "Mask.tif")

# Read image (OpenCV)
img = cv2.imread(img_path)

if img is None:
    raise ValueError("Retinal image not found")

# Read mask (PIL - safe for GIF/TIF)
mask_pil = Image.open(mask_path).convert("L")
mask = np.array(mask_pil)

# Convert mask to binary
mask = (mask > 0).astype(np.uint8)

# Extract green channel
green = img[:, :, 1]

# Preprocessing
green = cv2.GaussianBlur(green, (5, 5), 0)
green_f = green.astype(np.float32)

# Parameters
window = 25
k_niblack = -0.2
k_sauvola = 0.5
R = 128

# Local statistics
mean = cv2.boxFilter(green_f, -1, (window, window))
mean_sq = cv2.boxFilter(green_f ** 2, -1, (window, window))
std = np.sqrt(mean_sq - mean ** 2)

# Niblack thresholding
niblack_thresh = mean + k_niblack * std
niblack = (green_f < niblack_thresh).astype(np.uint8)

# Sauvola thresholding
sauvola_thresh = mean * (1 + k_sauvola * ((std / R) - 1))
sauvola = (green_f < sauvola_thresh).astype(np.uint8)

# Sensitivity metric
def sensitivity(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp / (tp + fn + 1e-7)

print("Niblack Sensitivity:", round(sensitivity(niblack, mask), 4))
print("Sauvola Sensitivity:", round(sensitivity(sauvola, mask), 4))

# Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 4, 1)
plt.imshow(green, cmap="gray")
plt.title("Green Channel")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(niblack, cmap="gray")
plt.title("Niblack")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(sauvola, cmap="gray")
plt.title("Sauvola")
plt.axis("off")

plt.tight_layout()
plt.show()
