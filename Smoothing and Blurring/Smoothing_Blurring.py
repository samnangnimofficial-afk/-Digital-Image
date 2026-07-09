# Importing the modules
import cv2
import numpy as np

# Reading the image
image = cv2.imread('image.jpg')

# Creating the kernel with numpy
kernel2 = np.ones((5, 5), np.float32)/25

# Applying the filter
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

# showing the image
cv2.imshow('Original', image)
cv2.imshow('Kernel Blur', img)

cv2.waitKey()
cv2.destroyAllWindows()
# Applying the filter bilateral

bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Original', image)
cv2.imshow('Bilateral blur', bilateral)

cv2.waitKey()
cv2.destroyAllWindows()

# Applying the filter averageBlur
averageBlur = cv2.blur(image, (5, 5))

# Showing the image
cv2.imshow('Original', image)
cv2.imshow('Average blur', averageBlur)

cv2.waitKey()
cv2.destroyAllWindows()
# Applying the filter medianBlur
medianBlur = cv2.medianBlur(image, 9)

cv2.imshow('Original', image)
cv2.imshow('Median blur', medianBlur)

cv2.waitKey()
cv2.destroyAllWindows()
# Applying the filter Gaussian
gaussian = cv2.GaussianBlur(image, (3, 3), 0)

# Showing the image
cv2.imshow('Original', image)
cv2.imshow('Gaussian blur', gaussian)

cv2.waitKey()
cv2.destroyAllWindows()