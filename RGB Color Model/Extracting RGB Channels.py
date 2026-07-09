import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Image.png')
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
