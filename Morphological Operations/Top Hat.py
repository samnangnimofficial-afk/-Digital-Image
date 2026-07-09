import cv2, numpy as np, matplotlib.pyplot as plt
img = cv2.imread("noise_car.png", 0)

bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
k = np.ones((13, 13), np.uint8)
top = cv2.morphologyEx(bin, cv2.MORPH_TOPHAT, k)
plt.imshow(top, cmap='gray'), plt.axis('off'), plt.show()