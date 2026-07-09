import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# =====================================================
# Read Image
# =====================================================
img = cv2.imread("sample.jpg", 0)

if img is None:
    raise ValueError("Image not found! Check the file path.")

# =====================================================
# Show Original Image
# =====================================================
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# =====================================================
# FFT
# =====================================================
f = np.fft.fft2(img)

magnitude = 20 * np.log(np.abs(f) + 1)

plt.figure(figsize=(6,6))
plt.imshow(magnitude, cmap="gray")
plt.title("Magnitude Spectrum")
plt.axis("off")
plt.show()

# =====================================================
# Center FFT
# =====================================================
fshift = np.fft.fftshift(f)

magnitude_shift = 20 * np.log(np.abs(fshift) + 1)

plt.figure(figsize=(6,6))
plt.imshow(magnitude_shift, cmap="gray")
plt.title("Centered Spectrum")
plt.axis("off")
plt.show()

# =====================================================
# Distance Function
# =====================================================
def distance(point1, point2):
    return math.sqrt(
        (point1[0]-point2[0])**2 +
        (point1[1]-point2[1])**2
    )

# =====================================================
# Ideal Low Pass Filter
# =====================================================
def idealFilterLP(D0, imgShape):

    rows, cols = imgShape[:2]

    base = np.zeros((rows, cols))

    center = (rows/2, cols/2)

    for x in range(cols):
        for y in range(rows):

            if distance((y, x), center) <= D0:
                base[y, x] = 1

    return base

# =====================================================
# Ideal High Pass Filter
# =====================================================
def idealFilterHP(D0, imgShape):

    rows, cols = imgShape[:2]

    base = np.ones((rows, cols))

    center = (rows/2, cols/2)

    for x in range(cols):
        for y in range(rows):

            if distance((y, x), center) <= D0:
                base[y, x] = 0

    return base

# =====================================================
# Gaussian Low Pass Filter
# =====================================================
def gaussianLP(D0, imgShape):

    rows, cols = imgShape[:2]

    base = np.zeros((rows, cols))

    center = (rows/2, cols/2)

    for x in range(cols):
        for y in range(rows):

            base[y, x] = math.exp(
                -(distance((y, x), center) ** 2) /
                (2 * (D0 ** 2))
            )

    return base

# =====================================================
# Gaussian High Pass Filter
# =====================================================
def gaussianHP(D0, imgShape):

    rows, cols = imgShape[:2]

    base = np.zeros((rows, cols))

    center = (rows/2, cols/2)

    for x in range(cols):
        for y in range(rows):

            base[y, x] = 1 - math.exp(
                -(distance((y, x), center) ** 2) /
                (2 * (D0 ** 2))
            )

    return base

# =====================================================
# Display Filters
# =====================================================
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0,0].imshow(idealFilterLP(50, img.shape), cmap="gray")
ax[0,0].set_title("Ideal Low Pass")
ax[0,0].axis("off")

ax[0,1].imshow(idealFilterHP(50, img.shape), cmap="gray")
ax[0,1].set_title("Ideal High Pass")
ax[0,1].axis("off")

ax[1,0].imshow(gaussianLP(50, img.shape), cmap="gray")
ax[1,0].set_title("Gaussian Low Pass")
ax[1,0].axis("off")

ax[1,1].imshow(gaussianHP(50, img.shape), cmap="gray")
ax[1,1].set_title("Gaussian High Pass")
ax[1,1].axis("off")

plt.tight_layout()
plt.savefig("filters.png")
plt.show()

# =====================================================
# Apply Ideal Low Pass Filter
# =====================================================
LPF = idealFilterLP(50, img.shape)

filtered_frequency = fshift * LPF

plt.figure(figsize=(6,6))
plt.imshow(np.log(1 + np.abs(filtered_frequency)), cmap="gray")
plt.title("Filtered Frequency Domain")
plt.axis("off")
plt.savefig("filtered_frequency_domain.png")
plt.show()

# =====================================================
# Inverse FFT
# =====================================================
f_ishift = np.fft.ifftshift(filtered_frequency)

img_back = np.fft.ifft2(f_ishift)

img_back = np.abs(img_back)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(img_back, cmap="gray")
plt.title("Recovered Image")
plt.axis("off")

plt.savefig("filtered_image_inverse_fft.png")
plt.show()

# =====================================================
# Save Images
# =====================================================
cv2.imwrite("original_image.jpg", img)
cv2.imwrite("filtered_image.jpg", np.uint8(img_back))

# =====================================================
# Frequency Transformation Function
# =====================================================
def Freq_Trans(image, filter_used):

    img_fft = np.fft.fft2(image)

    centered = np.fft.fftshift(img_fft)

    filtered = centered * filter_used

    inverse_shift = np.fft.ifftshift(filtered)

    final = np.fft.ifft2(inverse_shift)

    return (
        img_fft,
        centered,
        filter_used,
        filtered,
        inverse_shift,
        final
    )

# =====================================================
# Compare All Filters
# =====================================================
filters = [

    (idealFilterLP,50),
    (idealFilterLP,100),
    (idealFilterLP,150),

    (idealFilterHP,50),
    (idealFilterHP,100),
    (idealFilterHP,150),

    (gaussianLP,50),
    (gaussianLP,100),
    (gaussianLP,150),

    (gaussianHP,50),
    (gaussianHP,100),
    (gaussianHP,150)

]

titles = [

    "Original",
    "Spectrum",
    "Centered Spectrum",
    "Filter",
    "Filtered Spectrum",
    "Inverse Shift",
    "Recovered"

]

fig, axs = plt.subplots(12, 7, figsize=(28, 50))

for row, (filter_func, diameter) in enumerate(filters):

    result = Freq_Trans(img, filter_func(diameter, img.shape))

    images = [

        img,
        np.log(1 + np.abs(result[0])),
        np.log(1 + np.abs(result[1])),
        result[2],
        np.log(1 + np.abs(result[3])),
        np.log(1 + np.abs(result[4])),
        np.abs(result[5])

    ]

    for col in range(7):

        axs[row, col].imshow(images[col], cmap="gray")

        if col == 3:
            axs[row, col].set_title(
                f"{filter_func.__name__}\nD={diameter}"
            )
        else:
            axs[row, col].set_title(titles[col])

        axs[row, col].axis("off")

plt.tight_layout()

plt.savefig("all_processes.png")

plt.show()

print("===================================")
print("Program Finished Successfully!")
print("Images Saved:")
print(" - original_image.jpg")
print(" - filtered_image.jpg")
print(" - filters.png")
print(" - filtered_frequency_domain.png")
print(" - filtered_image_inverse_fft.png")
print(" - all_processes.png")
print("===================================")