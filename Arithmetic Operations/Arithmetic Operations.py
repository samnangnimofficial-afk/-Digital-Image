#pip install opencv - python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('input1.png')
img2 = cv2.imread('input2.png')
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

line_thickness = 5
height = img1.shape[0]
line = np.full((height, line_thickness, 3), (0, 0, 255), dtype=np.uint8)

side_by_side = np.hstack((img1, line, img2))

side_by_side_rgb = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)
added = cv2.add(img1, img2)
added_rgb = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)

plt.imshow(added_rgb)
plt.title('Addition (cv2.add)')
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(side_by_side_rgb)
plt.title('input1 input2')
plt.axis('off')
plt.show()