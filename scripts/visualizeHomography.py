import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIGURATION ===
base_path = Path("/mnt/ssd2tb/boat/thesis/superpoint_dataset/HPatches/wl_omanga1")
id1, id2 = 1, 2
image1_path = base_path / f"{id1}.ppm"
image2_path = base_path / f"{id2}.ppm"
homography_path = base_path / f"H_{id1}_{id2}"

# === LOAD IMAGES ===
img1 = cv2.imread(str(image1_path))
img2 = cv2.imread(str(image2_path))

# Convert to RGB
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# === LOAD HOMOGRAPHY ===
with open(homography_path, 'r') as f:
    H = np.array([[float(num) for num in line.strip().split()] for line in f])

# === WARP IMAGE 1 ===
height, width = img2.shape[:2]
warped_img1 = cv2.warpPerspective(img1_rgb, H, (width, height))

# === CREATE STRIPED MASK ===
stripe_width = 10  # Width of each stripe
mask = np.zeros((height, width), dtype=np.uint8)

for x in range(0, width, 2 * stripe_width):
    mask[:, x:x + stripe_width] = 1  # Create alternating vertical stripes

# Expand to 3 channels
mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

# === APPLY MASK TO COMBINE IMAGES ===
combined = img2_rgb.copy()
combined[mask_3ch == 1] = warped_img1[mask_3ch == 1]

# === DISPLAY ===
plt.figure(figsize=(12, 6))
plt.title("Overlay with Alternating Stripe Transparency (Image 1 on Image 2)")
plt.imshow(combined)
plt.axis('off')
plt.show()
