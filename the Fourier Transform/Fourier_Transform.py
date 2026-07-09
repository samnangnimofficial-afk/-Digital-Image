import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read image in grayscale
image_path = "images.jpg"
image = cv2.imread(image_path, 0)

# Check if image loaded successfully
if image is None:
    raise ValueError("Image not found. Check the image path.")

# Compute DFT
DFT = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift zero frequency to center
shift = np.fft.fftshift(DFT)

# Create Low-Pass Filter Mask
row, col = image.shape
center_row, center_col = row // 2, col // 2

mask = np.zeros((row, col, 2), np.uint8)
mask[center_row-30:center_row+30,
     center_col-30:center_col+30] = 1

# Apply mask
filtered = shift * mask

# Shift back
fft_ifft_shift = np.fft.ifftshift(filtered)

# Inverse DFT
imageThen = cv2.idft(fft_ifft_shift)

# Compute magnitude
imageThen = cv2.magnitude(imageThen[:, :, 0], imageThen[:, :, 1])

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Input Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(imageThen, cmap='gray')
plt.title("Low Pass Filter Output")
plt.axis("off")

plt.show()

