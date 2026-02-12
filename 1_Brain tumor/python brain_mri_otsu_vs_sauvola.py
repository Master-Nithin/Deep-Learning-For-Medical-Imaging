import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("1_image.png.tif", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("1_mask.png.tif", cv2.IMREAD_GRAYSCALE)

if img is None or mask is None:
    raise ValueError("Image or mask file not found")

mask = (mask > 0).astype(np.uint8)

# ---------- OTSU ----------
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu = (otsu > 0).astype(np.uint8)

# ---------- SAUVOLA (MANUAL IMPLEMENTATION) ----------
window = 25
k = 0.5
R = 128

img_f = img.astype(np.float32)

mean = cv2.boxFilter(img_f, -1, (window, window))
mean_sq = cv2.boxFilter(img_f**2, -1, (window, window))
std = np.sqrt(mean_sq - mean**2)

sauvola_thresh = mean * (1 + k * ((std / R) - 1))
sauvola = (img_f > sauvola_thresh).astype(np.uint8)

# ---------- METRICS ----------
def dice(pred, gt):
    return 2 * np.sum(pred * gt) / (np.sum(pred) + np.sum(gt) + 1e-7)

def jaccard(pred, gt):
    inter = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - inter
    return inter / (union + 1e-7)

print("Otsu Dice Score:", round(dice(otsu, mask), 4))
print("Otsu Jaccard Index:", round(jaccard(otsu, mask), 4))
print("Sauvola Dice Score:", round(dice(sauvola, mask), 4))
print("Sauvola Jaccard Index:", round(jaccard(sauvola, mask), 4))

# ---------- DISPLAY ----------
plt.figure(figsize=(10,4))

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("MRI Image")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(mask, cmap='gray')
plt.title("Ground Truth")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(otsu, cmap='gray')
plt.title("Otsu")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(sauvola, cmap='gray')
plt.title("Sauvola")
plt.axis('off')

plt.tight_layout()
plt.show()
