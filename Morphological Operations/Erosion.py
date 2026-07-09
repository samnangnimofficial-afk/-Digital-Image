import cv2, numpy as np, matplotlib.pyplot as plt
img = cv2.imread(r"test.jpg", 0) 
bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]  

k = np.ones((5, 5), np.uint8)  
inv = cv2.bitwise_not(bin)                                 
out = cv2.erode(inv, k, 1)              
plt.imshow(out, cmap='gray'), plt.axis('off'), plt.show()