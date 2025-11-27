import numpy as np
import matplotlib.pyplot as plt
import cv2

# 💬 ==========================
# 📌 USER PARAMETERS (Adjust here)
# 💬 ==========================
KEYPOINTS_FILE = "/home/boat/proxyISP/pytorch-superpoint/temp_log/homoadapt_pnts_online"         # path to keypoints txt file
IMAGE_FILE = "/home/boat/proxyISP/pytorch-superpoint/temp_log/images.png"                 # path to image
OUTPUT_FILE = "/home/boat/proxyISP/pytorch-superpoint/temp_log/plottedPnts.png"    # path to save result
POINT_COLOR = 'lime'                     # color of points
EDGE_COLOR = 'red'                       # edge color of points
POINT_SIZE = 30                         # size of scatter points
MIN_CONFIDENCE = 0.0                     # min score to keep keypoints
SHOW_FIGURE = False                      # set True to visualize

# 💬 ==========================
# 🚀 1. Load keypoints from text file
# 💬 ==========================
with open(KEYPOINTS_FILE, "r") as f:
    raw_text = f.read().strip()

# Remove brackets and line breaks
clean_text = raw_text.replace('[', '').replace(']', '').replace('\n', ' ')

# Convert to numpy array (float)
data = np.fromstring(clean_text, sep=' ')
keypoints = data.reshape(-1, 3)  # (N, 3)

print(f"✅ Loaded {keypoints.shape[0]} keypoints from: {KEYPOINTS_FILE}")

# Filter keypoints by confidence
if MIN_CONFIDENCE > 0:
    keypoints = keypoints[keypoints[:, 2] >= MIN_CONFIDENCE]

# 💬 ==========================
# 🖼️ 2. Load image
# 💬 ==========================
img = cv2.imread(IMAGE_FILE)
if img is None:
    raise FileNotFoundError(f"❌ Image not found: {IMAGE_FILE}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 💬 ==========================
# 🪄 3. Plot image + keypoints
# 💬 ==========================
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.scatter(
    keypoints[:, 0],
    keypoints[:, 1],
    s=POINT_SIZE,
    c=POINT_COLOR,
    edgecolors=EDGE_COLOR
)
plt.title("Keypoints Visualization")
plt.axis('off')

# 💬 ==========================
# 💾 4. Save (and optionally show)
# 💬 ==========================
plt.savefig(OUTPUT_FILE, bbox_inches='tight', pad_inches=0)
if SHOW_FIGURE:
    plt.show()
plt.close()

print(f"✅ Keypoints plotted and saved to: {OUTPUT_FILE}")