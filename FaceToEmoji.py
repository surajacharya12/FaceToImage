import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# --- Setup Face Landmarker with BLENDSHAPES for expression detection ---
model_path = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,  # Enable facial expression detection!
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# --- Load emoji images ---
cap = cv2.VideoCapture(0)

smile_img = cv2.imread("smile.webp", cv2.IMREAD_UNCHANGED)
cry_img = cv2.imread("cry.webp", cv2.IMREAD_UNCHANGED)
okey_img = cv2.imread("okey.webp", cv2.IMREAD_UNCHANGED)

frame_timestamp_ms = 0


def overlay_image(background, overlay, x, y, size):
    """Overlay an image onto the background at position (x, y) with given size."""
    if overlay is None or size <= 0:
        return

    h_bg, w_bg = background.shape[:2]
    overlay = cv2.resize(overlay, (size, size))

    # Calculate visible region (handle edges)
    x1_bg = max(x, 0)
    y1_bg = max(y, 0)
    x2_bg = min(x + size, w_bg)
    y2_bg = min(y + size, h_bg)

    x1_ov = x1_bg - x
    y1_ov = y1_bg - y
    x2_ov = x2_bg - x
    y2_ov = y2_bg - y

    if x1_bg >= x2_bg or y1_bg >= y2_bg:
        return

    if overlay.shape[2] == 4:
        alpha = overlay[y1_ov:y2_ov, x1_ov:x2_ov, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)

        background[y1_bg:y2_bg, x1_bg:x2_bg] = (
            alpha * overlay[y1_ov:y2_ov, x1_ov:x2_ov, :3]
            + (1 - alpha) * background[y1_bg:y2_bg, x1_bg:x2_bg]
        ).astype(np.uint8)
    else:
        background[y1_bg:y2_bg, x1_bg:x2_bg] = overlay[y1_ov:y2_ov, x1_ov:x2_ov]


def get_blendshape_score(blendshapes, name):
    """Get the score for a specific blendshape by name."""
    for bs in blendshapes:
        if bs.category_name == name:
            return bs.score
    return 0.0


def detect_expression(blendshapes):
    """
    Detect facial expression using Face Blendshapes.
    Returns (emoji_image, expression_label, scores_dict)
    """
    # Key blendshape scores for expression detection
    smile_left = get_blendshape_score(blendshapes, "mouthSmileLeft")
    smile_right = get_blendshape_score(blendshapes, "mouthSmileRight")
    frown_left = get_blendshape_score(blendshapes, "mouthFrownLeft")
    frown_right = get_blendshape_score(blendshapes, "mouthFrownRight")
    brow_down_left = get_blendshape_score(blendshapes, "browDownLeft")
    brow_down_right = get_blendshape_score(blendshapes, "browDownRight")
    brow_inner_up = get_blendshape_score(blendshapes, "browInnerUp")
    eye_squint_left = get_blendshape_score(blendshapes, "eyeSquintLeft")
    eye_squint_right = get_blendshape_score(blendshapes, "eyeSquintRight")
    mouth_press_left = get_blendshape_score(blendshapes, "mouthPressLeft")
    mouth_press_right = get_blendshape_score(blendshapes, "mouthPressRight")
    mouth_lower_down_left = get_blendshape_score(blendshapes, "mouthLowerDownLeft")
    mouth_lower_down_right = get_blendshape_score(blendshapes, "mouthLowerDownRight")

    # Calculate combined scores
    smile_score = (smile_left + smile_right) / 2
    frown_score = (frown_left + frown_right) / 2
    brow_down_score = (brow_down_left + brow_down_right) / 2
    eye_squint_score = (eye_squint_left + eye_squint_right) / 2
    mouth_press_score = (mouth_press_left + mouth_press_right) / 2
    mouth_lower_score = (mouth_lower_down_left + mouth_lower_down_right) / 2

    # Sad: frown + inner brow raise + pressed lips + lowered mouth corners
    sad_score = (
        frown_score * 0.35
        + brow_inner_up * 0.25
        + mouth_press_score * 0.15
        + brow_down_score * 0.15
        + mouth_lower_score * 0.10
    )

    # Happy: smile + eye squint (genuine smile / Duchenne smile)
    happy_score = smile_score * 0.7 + eye_squint_score * 0.3

    scores = {
        "Happy": happy_score,
        "Sad": sad_score,
        "Frown": frown_score,
        "BrowUp": brow_inner_up,
    }

    # Determine expression — check sad FIRST since it's harder to trigger
    if happy_score > 0.4:
        return smile_img, "HAPPY :)", scores
    elif sad_score > 0.12 or frown_score > 0.15:
        return cry_img, "SAD :(", scores
    else:
        return okey_img, "NEUTRAL :|", scores


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # Convert to mediapipe Image
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    frame_timestamp_ms += 33  # ~30 FPS
    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    current_emoji = None
    expression_label = ""

    if results.face_landmarks and results.face_blendshapes:
        for i, face_landmarks in enumerate(results.face_landmarks):
            # Draw face outline rectangle
            top = face_landmarks[10]
            bottom = face_landmarks[152]
            left = face_landmarks[234]
            right = face_landmarks[454]

            x1, y1 = int(top.x * w), int(top.y * h)
            x2, y2 = int(bottom.x * w), int(bottom.y * h)
            lx = int(left.x * w)
            rx = int(right.x * w)
            cv2.rectangle(img, (lx, y1), (rx, y2), (0, 255, 0), 2)

            # Detect expression using BLENDSHAPES (real facial expressions!)
            blendshapes = results.face_blendshapes[i]
            current_emoji, expression_label, scores = detect_expression(blendshapes)

            # Show expression scores on screen
            y_pos = 30
            cv2.putText(
                img,
                f"Expression: {expression_label}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_pos += 30
            for name, score in scores.items():
                # Draw score bar
                bar_width = int(score * 200)
                color = (0, 255, 255)
                if name == "Happy" and score > 0.4:
                    color = (0, 255, 0)
                elif name == "Sad" and score > 0.3:
                    color = (0, 0, 255)

                cv2.putText(
                    img,
                    f"{name}: {score:.2f}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                cv2.rectangle(
                    img, (130, y_pos - 12), (130 + bar_width, y_pos), color, -1
                )
                y_pos += 25

    # Show emoji in the BOTTOM-RIGHT corner
    if current_emoji is not None:
        emoji_size = 550 
        ex = w - emoji_size - 20
        ey = h - emoji_size - 20
        overlay_image(img, current_emoji, ex, ey, emoji_size)

    cv2.imshow("FaceToEmoji", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
