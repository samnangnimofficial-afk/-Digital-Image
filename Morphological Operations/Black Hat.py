import cv2, numpy as np, matplotlib.pyplot as plt
img = cv2.imread("noise_car.png", 0)

bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
inv = cv2.bitwise_not(bin)
k = np.ones((5, 5), np.uint8)
bh = cv2.morphologyEx(inv, cv2.MORPH_BLACKHAT, k)
plt.imshow(bh, cmap='gray'), plt.axis('off'), plt.show()