import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Read Image
# ==========================
image = cv2.imread("Image.png")

if image is None:
    raise ValueError("Image not found! Check Image.png path.")

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split RGB channels
R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

# ==========================
# Flatten Channels
# ==========================
R_flatten = R.ravel()
G_flatten = G.ravel()
B_flatten = B.ravel()

# ==========================
# Correlation Matrix
# ==========================
correlation_matrix = np.corrcoef(
    [R_flatten, G_flatten, B_flatten]
)

print("Correlation Matrix:")
print("       R          G          B")
print(correlation_matrix)

# ==========================
# Display Correlation Matrix
# ==========================
plt.figure(figsize=(6,5))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    interpolation="none"
)

plt.colorbar()

plt.xticks(
    [0,1,2],
    ["R","G","B"]
)

plt.yticks(
    [0,1,2],
    ["R","G","B"]
)

plt.title("RGB Correlation Matrix")

plt.show()