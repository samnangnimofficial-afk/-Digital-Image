#pip install scikit-image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

# =====================================
# Read Image
# =====================================
I = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if I is None:
    raise ValueError("Image not found! Make sure 'test.jpg' is in the current folder.")

# =====================================
# Add Gaussian Noise
# =====================================
noise = np.random.normal(0, 25, I.shape)

I_noise = I.astype(np.float32) + noise

I_noise = np.clip(I_noise, 0, 255).astype(np.uint8)

# =====================================
# Display Figure
# =====================================
plt.figure(figsize=(18, 10))

# =====================================
# Original Noisy Image
# =====================================
plt.subplot(2, 4, 1)
plt.imshow(I_noise, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

# =====================================
# Sobel
# =====================================
sobelx = cv2.Sobel(I_noise, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(I_noise, cv2.CV_64F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobelx, sobely)

plt.subplot(2, 4, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel")
plt.axis("off")

# =====================================
# Prewitt
# =====================================
kernelx = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

kernely = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
], dtype=np.float32)

prewittx = cv2.filter2D(I_noise, cv2.CV_64F, kernelx)
prewitty = cv2.filter2D(I_noise, cv2.CV_64F, kernely)

prewitt = np.hypot(prewittx, prewitty)

plt.subplot(2, 4, 3)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt")
plt.axis("off")

# =====================================
# Roberts
# =====================================
robertsx = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

robertsy = np.array([
    [0, 1],
    [-1, 0]
], dtype=np.float32)

rx = cv2.filter2D(I_noise, cv2.CV_64F, robertsx)
ry = cv2.filter2D(I_noise, cv2.CV_64F, robertsy)

roberts = np.hypot(rx, ry)

plt.subplot(2, 4, 4)
plt.imshow(roberts, cmap="gray")
plt.title("Roberts")
plt.axis("off")

# =====================================
# Laplacian of Gaussian (LoG)
# =====================================
blur = cv2.GaussianBlur(I_noise, (5, 5), 0)

log = cv2.Laplacian(blur, cv2.CV_64F)

plt.subplot(2, 4, 5)
plt.imshow(log, cmap="gray")
plt.title("LoG")
plt.axis("off")

# =====================================
# Zero Crossing (using Canny from skimage)
# =====================================
zero = feature.canny(I_noise, sigma=1)

plt.subplot(2, 4, 6)
plt.imshow(zero, cmap="gray")
plt.title("Zero Crossing")
plt.axis("off")

# =====================================
# Canny
# =====================================
canny = cv2.Canny(I_noise, 100, 200)

plt.subplot(2, 4, 7)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.axis("off")

# =====================================
# Save Result
# =====================================
plt.tight_layout()

plt.savefig("edge_detection_comparison.png", dpi=300)

plt.show()