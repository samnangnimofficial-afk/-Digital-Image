"""
Face + Hand + Pose landmark detection using the MediaPipe TASKS API.

Why this version exists:
MediaPipe removed the old `mp.solutions.holistic` / `mp.solutions.drawing_utils`
API in recent releases. Those are only available in mediapipe<=0.10.21, which
in turn only ships wheels for Python <=3.12. If you're on Python 3.13/3.14,
you get whatever the newest mediapipe is (0.10.30+), which no longer has
`mp.solutions` at all -- hence the AttributeError you were hitting.

This script uses the modern replacement: three separate "Tasks" models
(HandLandmarker, FaceLandmarker, PoseLandmarker) run together, since there is
currently no single combined "Holistic" task in the new API. Landmarks are
drawn manually with OpenCV, since `mp_drawing` doesn't exist anymore either.

Requirements:
    pip install mediapipe opencv-python

First run will auto-download ~15MB of model files into a "models" folder
next to this script.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# =====================================
# Model download (first run only)
# =====================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}


def ensure_models():
    paths = {}
    for filename, url in MODEL_URLS.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, path)
        paths[filename] = path
    return paths


# =====================================
# Landmark connection topology
# (mp_drawing.draw_landmarks handled this for us before; now we do it
#  ourselves using the same standard connection sets MediaPipe used to
#  ship in mediapipe.python.solutions.*_connections)
# =====================================

HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),   # palm
    (1, 2), (2, 3), (3, 4),                               # thumb
    (5, 6), (6, 7), (7, 8),                               # index
    (9, 10), (10, 11), (11, 12),                          # middle
    (13, 14), (14, 15), (15, 16),                         # ring
    (17, 18), (18, 19), (19, 20),                         # pinky
]

POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
    (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18),
    (16, 20), (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30),
    (28, 32), (29, 31), (30, 32),
]

HAND_LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def draw_landmarks(image, landmark_list, connections, point_color, line_color,
                    point_radius=2, line_thickness=1):
    h, w = image.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]

    for a, b in connections:
        if a < len(points) and b < len(points):
            cv2.line(image, points[a], points[b], line_color, line_thickness)

    for p in points:
        cv2.circle(image, p, point_radius, point_color, -1)


def draw_face_points(image, landmark_list, color=(255, 0, 255), radius=1):
    h, w = image.shape[:2]
    for lm in landmark_list:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), radius, color, -1)


# =====================================
# Set up the three landmarkers
# =====================================

model_paths = ensure_models()

BaseOptions = mp_python.BaseOptions
VisionRunningMode = vision.RunningMode

hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_paths["pose_landmarker_lite.task"]),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# =====================================
# Open Camera
# =====================================

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise RuntimeError(
        "Could not open webcam (index 0). Check that it's connected and not in use by another app."
    )

previousTime = 0
start_time = time.time()

try:
    while capture.isOpened():

        ret, frame = capture.read()
        if not ret:
            print("Cannot receive frame")
            break

        frame = cv2.resize(frame, (800, 600))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        image = frame  # already BGR, ready for display/drawing

        # Pose
        for pose_landmarks in pose_result.pose_landmarks:
            draw_landmarks(
                image, pose_landmarks, POSE_CONNECTIONS,
                point_color=(245, 117, 66), line_color=(245, 66, 230),
                point_radius=3, line_thickness=2,
            )

        # Face (dots only -- full FACEMESH_TESSELATION has ~2500 edges,
        # omitted here for simplicity; the 468 points still give you the
        # face shape)
        for face_landmarks in face_result.face_landmarks:
            draw_face_points(image, face_landmarks, color=(255, 0, 255), radius=1)

        # Hands
        for hand_landmarks in hand_result.hand_landmarks:
            draw_landmarks(
                image, hand_landmarks, HAND_CONNECTIONS,
                point_color=(0, 255, 255), line_color=(0, 255, 0),
                point_radius=3, line_thickness=2,
            )

        # =====================================
        # FPS Calculation
        # =====================================
        currentTime = time.time()
        fps = 0
        if currentTime != previousTime:
            fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(
            image, f"{int(fps)} FPS", (10, 70),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2,
        )

        cv2.imshow("Face and Hand Landmarks", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

finally:
    capture.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    face_landmarker.close()
    pose_landmarker.close()


# =====================================
# Print Hand Landmark Index (reference)
# =====================================
if __name__ == "__main__":
    for i, name in enumerate(HAND_LANDMARK_NAMES):
        print(name, i)

    print("Wrist index:", HAND_LANDMARK_NAMES.index("WRIST"))