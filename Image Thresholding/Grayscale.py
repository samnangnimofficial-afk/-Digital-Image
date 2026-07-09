import cv2

img = cv2.imread("images.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Binary Thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ToZero Thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

cv2.imshow("ToZero Threshold", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ToZero Inverse Thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow("ToZero Inverse", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()