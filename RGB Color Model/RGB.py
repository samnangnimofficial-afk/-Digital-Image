import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Image.png')
plt.imshow(image)
plt.title('Image in BGR Mode')
plt.axis('off')
plt.show()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title('Image in RGB Mode')
plt.axis('off')
plt.show()
height, width, num_channels = image.shape
print("Height:", height)
print("Width:", width)
print("No. of channels:", num_channels)
print("Total no. of pixels in the image:", height * width)
flat_image = image.reshape(-1, 3)
print("First 10 pixels (flattened):")
for i in range(10):
    print(f"Pixel {i+1} (R value, G value, B value): {flat_image[i]}")

# Extract the RGB channels
r, g, b = cv2.split(image)

plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.imshow(r, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(g, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(b, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.show()

