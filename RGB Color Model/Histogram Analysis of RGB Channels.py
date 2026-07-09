import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("Image.png")

# Check image
if image is None:
    raise ValueError("Image not found! Check Image.png path.")

# Convert BGR (OpenCV) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split RGB channels
R = image_rgb[:, :, 0]
G = image_rgb[:, :, 1]
B = image_rgb[:, :, 2]

# Plot RGB Histograms
plt.figure(figsize=(15, 5))

colors = ['red', 'green', 'blue']
channels = [R, G, B]

for i, (channel, color) in enumerate(zip(channels, colors)):

    plt.subplot(1, 3, i+1)

    plt.hist(
        channel.ravel(),
        bins=256,
        color=color,
        alpha=0.8
    )

    plt.title(f'{color.capitalize()} Channel Histogram')
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()