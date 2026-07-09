import cv2
import numpy as np
import time

# ==========================
# Video Paths
# ==========================
VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "edge_output.mp4"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ==========================
# Resize Frame
# ==========================
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height),
                      interpolation=cv2.INTER_AREA)

# ==========================
# Convert to Grayscale
# ==========================
def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# ==========================
# CLAHE
# ==========================
def apply_clahe(gray):
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)

# ==========================
# Bilateral Filter
# ==========================
def bilateral_smooth(gray):
    return cv2.bilateralFilter(gray, 9, 75, 75)

# ==========================
# Dynamic Canny
# ==========================
def dynamic_canny(smooth):

    sigma = np.std(smooth)

    lower = max(20, int(0.66 * sigma))
    upper = min(200, int(1.33 * sigma))

    return cv2.Canny(smooth, lower, upper)

# ==========================
# Sobel
# ==========================
def sobel_gradient(gray):

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    gradient = np.sqrt(sx**2 + sy**2)

    return cv2.convertScaleAbs(gradient)

# ==========================
# Laplacian
# ==========================
def laplacian_edge(gray):

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return cv2.convertScaleAbs(lap)

# ==========================
# Fuse Edges
# ==========================
def fuse_edges(canny, lap, sobel):

    fused = cv2.addWeighted(
        canny, 0.6,
        lap, 0.3,
        0
    )

    fused = cv2.addWeighted(
        fused, 0.7,
        sobel, 0.3,
        0
    )

    return fused

# ==========================
# Morphology
# ==========================
def morphology_close(fused):

    kernel = np.ones((3,3), np.uint8)

    return cv2.morphologyEx(
        fused,
        cv2.MORPH_CLOSE,
        kernel
    )

# ==========================
# Temporal Smoothing
# ==========================
prev_fused = None

def temporal_smooth(fused):

    global prev_fused

    if prev_fused is None:
        prev_fused = fused.copy()

    fused = cv2.addWeighted(
        fused,
        0.7,
        prev_fused,
        0.3,
        0
    )

    prev_fused = fused.copy()

    return fused

# ==========================
# Overlay Edges
# ==========================
def overlay_edges(frame, fused):

    overlay = frame.copy()

    overlay[fused > 40] = [0,0,255]

    return overlay

# ==========================
# Process Frame
# ==========================
def process_frame(frame):

    gray = to_grayscale(frame)

    clahe = apply_clahe(gray)

    smooth = bilateral_smooth(clahe)

    canny = dynamic_canny(smooth)

    lap = laplacian_edge(smooth)

    sobel = sobel_gradient(smooth)

    fused = fuse_edges(canny, lap, sobel)

    fused = morphology_close(fused)

    fused = temporal_smooth(fused)

    output = overlay_edges(frame, fused)

    return output

# ==========================
# Open Video
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError("Cannot open video!")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

prev_time = time.time()
frame_counter = 0

# ==========================
# Main Loop
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Finished processing video.")
        break

    frame = resize_frame(
        frame,
        FRAME_WIDTH,
        FRAME_HEIGHT
    )

    output = process_frame(frame)

    out.write(output)

    cv2.imshow("Edge Detection", output)

    frame_counter += 1

    curr_time = time.time()

    if curr_time - prev_time >= 1:

        fps_live = frame_counter / (curr_time - prev_time)

        print(f"Live FPS: {fps_live:.2f}")

        prev_time = curr_time

        frame_counter = 0

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# Release Resources
# ==========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as:", OUTPUT_PATH)