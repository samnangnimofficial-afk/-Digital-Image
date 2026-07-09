#pip install requests

import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRmKTFI_7r2TKR0sknfK_7GJX8sHItxbf-zh_jOJIBde2-L69K29IAzFLrD&s"

response = requests.get(url)
img_data = response.content

img = Image.open(BytesIO(img_data))

img_gray = np.array(img.convert('L'))  

f = np.fft.fft2(img_gray)
fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift))

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img_gray, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
cv2.imwrite('magnitude_spectrum.jpg', magnitude_spectrum)
plt.show()