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

# Convert the PIL image to a numpy array
img = np.array(img)

# If the image is in RGB, no need for conversion, else convert from BGR to RGB
if img.shape[2] == 3:  # Check if it's RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Removing noise in image using Non-Local Means Denoising
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# Plotting the source and destination images
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img), plt.title('Original Image')
plt.subplot(122), plt.imshow(dst), plt.title('Denoised Image (Non-Local Means Denoising)')
plt.show()