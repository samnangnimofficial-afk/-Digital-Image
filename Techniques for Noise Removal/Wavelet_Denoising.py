import numpy as np
import cv2
from matplotlib import pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Image URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRmKTFI_7r2TKR0sknfK_7GJX8sHItxbf-zh_jOJIBde2-L69K29IAzFLrD&s"

# Fetch the image from the URL
response = requests.get(url)
img_data = response.content

# Open the image using PIL
img = Image.open(BytesIO(img_data))

# Convert the PIL image to a grayscale numpy array
img = np.array(img.convert('L'))  # Convert to grayscale

# Check if the image is loaded correctly
if img is None:
    print("Error: Image file not found or unable to read. Please check the file path and format.")
    exit()

# Function to perform wavelet-like denoising
def wavelet_approx_denoising(image, levels=2):
    # Create Gaussian pyramid
    gaussian_pyramid = [image]
    for i in range(levels):
        gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))

    # Create Laplacian pyramid
    laplacian_pyramid = []
    for i in range(levels, 0, -1):
        # Upsample to the exact size of the next Gaussian layer
        upsampled = cv2.pyrUp(gaussian_pyramid[i])
        upsampled = cv2.resize(upsampled, (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i-1], upsampled)
        laplacian_pyramid.append(laplacian)

    # Apply thresholding to reduce noise in the Laplacian layers
    threshold = np.sqrt(2 * np.log(image.size))  # Universal threshold approximation
    denoised_laplacian_pyramid = [cv2.threshold(l, threshold, 255, cv2.THRESH_TOZERO)[1] for l in laplacian_pyramid]

    # Reconstruct the denoised image
    reconstructed = gaussian_pyramid[-1]
    for i in range(levels-1, -1, -1):
        # Upsample to the exact size of the Laplacian layer
        upsampled = cv2.pyrUp(reconstructed)
        upsampled = cv2.resize(upsampled, (denoised_laplacian_pyramid[i].shape[1], denoised_laplacian_pyramid[i].shape[0]))
        reconstructed = cv2.add(upsampled, denoised_laplacian_pyramid[i])

    return np.clip(reconstructed, 0, 255).astype(np.uint8)

# Applying wavelet-like denoising
dst = wavelet_approx_denoising(img, levels=2)

# Plotting the source and destination images
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Denoised Image (Wavelet filter)')
plt.show()