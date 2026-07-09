import cv2, numpy as np, matplotlib.pyplot as plt

img = cv2.imread(r"noise_car.png", 0)  
bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]  

k = np.ones((3, 3), np.uint8)  
opened = cv2.morphologyEx(bin, cv2.MORPH_OPEN, k)
plt.imshow(opened, cmap='gray'), plt.axis('off'), plt.show()