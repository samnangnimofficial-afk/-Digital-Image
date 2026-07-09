import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read local image
img = cv2.imread("Noise.jpg")

# Check if image is loaded
if img is None:
    raise ValueError("Image not found! Make sure 'Noise.jpg' exists.")

# Convert BGR to RGB for matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply Median Filter
dst = cv2.medianBlur(img, 5)

# Save the result
cv2.imwrite("Median_Filter_Result.jpg", cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))

# Display
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dst)
plt.title("Median Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
