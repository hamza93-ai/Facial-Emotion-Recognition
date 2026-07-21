"""
============================================================
Face Emotion Recognition - Images + Real-Time Webcam
============================================================
FER-2013 | 7 Emotions | CNN (scratch_cnn)
Usage:
  python main.py --webcam
  python main.py --images img1.jpg img2.png
  python main.py --folder ./test_images
  python main.py --images img.jpg --save output.jpg
============================================================
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # suppress TF noise

import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# ── TensorFlow import ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")              # suppress TF warnings
    print(f"[INFO] TensorFlow {tf.__version__} loaded.")
except ImportError:
    print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH    = "fixed_model.keras"
IMG_SIZE      = 48
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# BGR colors for each emotion
EMOTION_COLORS = {
    'angry':    (0,   0,   255),
    'disgust':  (0,   140, 0  ),
    'fear':     (180, 0,   180),
    'happy':    (0,   220, 0  ),
    'neutral':  (180, 180, 180),
    'sad':      (255, 100, 0  ),
    'surprise': (0,   200, 200),
}

# Emoji for fun overlay
EMOTION_EMOJI = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😄', 'neutral': '😐', 'sad': '😢', 'surprise': '😲',
}

# ── Model loader ─────────────────────────────────────────────────────────────
def load_model(path: str):
    """Load .keras model with clear error messages."""
    if not os.path.exists(path):
        print(f"[ERROR] Model not found: '{path}'")
        print("       Make sure fixed_model.keras is in the same folder as main.py")
        sys.exit(1)
    print(f"[INFO] Loading model from '{path}' ...")
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    print(f"[INFO] Model loaded | input: {model.input_shape} | output: {model.output_shape}")
    return model

# ── Face cascade loader ──────────────────────────────────────────────────────
def load_face_cascade():
    """Load Haar cascade with fallback error message."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("[ERROR] Haar cascade not found. Reinstall OpenCV.")
        sys.exit(1)
    return cascade

# ── Preprocessing ─────────────────────────────────────────────────────────────
# FIX 1: Training used `load_and_preprocess_gray` which does:
#   decode JPEG (1 channel) → resize(48,48) → float32 / 255.0
# So inference must match exactly — simple grayscale + normalize, NO CLAHE.
def preprocess(face_bgr: np.ndarray) -> np.ndarray:
    """
    Match the training pipeline (load_and_preprocess_gray in notebook):
      BGR → Gray → resize(48,48) → float32/255 → shape (1,48,48,1)
    """
    gray   = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    norm   = resized.astype("float32") / 255.0
    return norm[np.newaxis, ..., np.newaxis]          # (1, 48, 48, 1)

# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, x: int, y: int, w: int, h: int,
                 emotion: str, conf: float, all_preds: np.ndarray) -> np.ndarray:
    """
    Draw bounding box, label, confidence bar, and mini bar chart
    for all 7 emotions on the right side.
    FIX 2: label background height was hardcoded (y-40) — now dynamic.
    FIX 3: confidence bar had no white border in webcam mode — added.
    FIX 4: added all-emotion mini chart for better readability.
    """
    color = EMOTION_COLORS[emotion]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # --- Bounding box ---
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # --- Label background (dynamic height) ---
    label     = f"{emotion.upper()}  {conf * 100:.1f}%"
    scale     = 0.75
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 8
    label_y1  = max(0, y - th - pad * 2)
    label_y2  = y
    cv2.rectangle(frame, (x, label_y1), (x + tw + pad * 2, label_y2), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + pad, y - pad),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # --- Confidence bar (below face box) ---
    bar_h  = 14
    bar_y1 = y + h + 8
    bar_y2 = bar_y1 + bar_h
    filled_w = int(w * conf)
    if bar_y2 < frame.shape[0]:                       # FIX 5: bounds check
        cv2.rectangle(frame, (x, bar_y1), (x + w, bar_y2), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, bar_y1), (x + filled_w, bar_y2), color, -1)
        cv2.rectangle(frame, (x, bar_y1), (x + w, bar_y2), (255, 255, 255), 1)

    # --- Mini emotion chart (right of face box) ---
    chart_x   = x + w + 12
    chart_bar_w = 60
    chart_bar_h = 12
    chart_gap   = 4
    chart_top   = y

    for i, (emo, prob) in enumerate(zip(EMOTION_LABELS, all_preds)):
        ey1 = chart_top + i * (chart_bar_h + chart_gap)
        ey2 = ey1 + chart_bar_h
        if ey2 >= frame.shape[0] or chart_x + chart_bar_w + 55 >= frame.shape[1]:
            break
        bar_len = int(chart_bar_w * prob)
        emo_color = EMOTION_COLORS[emo]
        cv2.rectangle(frame, (chart_x, ey1), (chart_x + chart_bar_w, ey2), (40, 40, 40), -1)
        cv2.rectangle(frame, (chart_x, ey1), (chart_x + bar_len, ey2), emo_color, -1)
        cv2.putText(frame, f"{emo[:3]} {prob*100:.0f}%",
                    (chart_x + chart_bar_w + 4, ey2 - 2),
                    font, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    return frame

# ── Single image processing ──────────────────────────────────────────────────
def process_image(model, face_cascade, path: str):
    """
    Detect faces in an image and annotate with emotion.
    FIX 6: returns (annotated_frame, emotion_str) instead of just frame
            so callers can print results.
    FIX 7: shows message on frame when no face is detected.
    """
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[WARN] Cannot read image: {path}")
        return None, None

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # FIX 8: equalizeHist on detection-gray improves detection in dark images
    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return frame, None

    best_result = None
    best_score  = -1

    for (x, y, w, h) in faces:
        pad = int(w * 0.15)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame.shape[1], x + w + pad)
        y2  = min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        inp   = preprocess(face_crop)
        preds = model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(preds))
        conf  = float(preds[idx])
        score = conf * (w * h)

        if score > best_score:
            best_score  = score
            best_result = (x, y, w, h, EMOTION_LABELS[idx], conf, preds)

    if best_result is None:
        return frame, None

    x, y, w, h, emotion, conf, all_preds = best_result
    frame = draw_overlay(frame, x, y, w, h, emotion, conf, all_preds)
    return frame, emotion

# ── Real-time webcam ─────────────────────────────────────────────────────────
def run_webcam(model, face_cascade):
    """
    Real-time emotion detection from webcam.
    FIX 9:  model.predict() inside loop is slow — now uses a frame-skip
            counter so prediction runs every N frames for better FPS.
    FIX 10: added FPS counter overlay.
    FIX 11: added 'S' key to save screenshot.
    FIX 12: added ESC key as alternative quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not found or already in use.")
        return

    # Try to set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam started.")
    print("       Q / ESC  → quit")
    print("       S        → save screenshot")

    import time
    PREDICT_EVERY = 3          # run model every N frames (boosts FPS)
    frame_count   = 0
    cached_result = None       # last prediction result
    fps_timer     = time.time()
    fps           = 0.0
    screenshot_n  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — webcam may have disconnected.")
            break

        frame = cv2.flip(frame, 1)   # ← yeh add karo

        frame_count += 1

        # ── FPS calculation ──
        now = time.time()
        if now - fps_timer >= 1.0:
            fps       = frame_count / (now - fps_timer + 1e-6)
            fps_timer = now
            frame_count = 0

        # ── Detect + predict every N frames ──
        if frame_count % PREDICT_EVERY == 0:
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            faces  = face_cascade.detectMultiScale(
                gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )

            best_result = None
            best_score  = -1
            for (x, y, w, h) in faces:
                pad = int(w * 0.15)
                x1  = max(0, x - pad)
                y1  = max(0, y - pad)
                x2  = min(frame.shape[1], x + w + pad)
                y2  = min(frame.shape[0], y + h + pad)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                preds = model.predict(preprocess(face_crop), verbose=0)[0]
                idx   = int(np.argmax(preds))
                conf  = float(preds[idx])
                score = conf * (w * h)
                if score > best_score:
                    best_score  = score
                    best_result = (x, y, w, h, EMOTION_LABELS[idx], conf, preds)
            cached_result = best_result

        # ── Draw cached result ──
        if cached_result:
            x, y, w, h, emotion, conf, all_preds = cached_result
            frame = draw_overlay(frame, x, y, w, h, emotion, conf, all_preds)
        else:
            cv2.putText(frame, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # ── FPS overlay ──
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1, cv2.LINE_AA)

        cv2.imshow("FER - Real Time  |  Q/ESC: quit  |  S: screenshot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):            # Q or ESC
            break
        if key == ord('s'):
            screenshot_n += 1
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")

# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Face Emotion Recognition — FER-2013 | 7 classes"
    )
    parser.add_argument("--model",  default=MODEL_PATH,
                        help="Path to .keras model file (default: fixed_model.keras)")
    parser.add_argument("--images", nargs="+", metavar="IMG",
                        help="One or more image paths")
    parser.add_argument("--folder", metavar="DIR",
                        help="Folder of images (.jpg/.png/.jpeg)")
    parser.add_argument("--webcam", action="store_true",
                        help="Run real-time webcam mode")
    parser.add_argument("--save",   metavar="OUT",
                        help="Save annotated output (works with --images, single image)")
    args = parser.parse_args()

    # ── Load model + cascade ──
    model        = load_model(args.model)
    face_cascade = load_face_cascade()

    # ── Webcam mode ──
    if args.webcam:
        run_webcam(model, face_cascade)
        return

    # ── Collect image paths ──
    image_paths = []
    if args.images:
        for p in args.images:
            if os.path.isfile(p):
                image_paths.append(p)
            else:
                print(f"[WARN] File not found, skipping: {p}")

    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"[ERROR] Folder not found: {args.folder}")
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            found = [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
            if not found:
                print(f"[WARN] No images found in '{args.folder}'")
            image_paths.extend(found)

    if not image_paths:
        print("[ERROR] No images provided. Use --images, --folder, or --webcam.")
        parser.print_help()
        return

    # ── Process images ──
    for i, path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing: {path}")
        img, emotion = process_image(model, face_cascade, path)
        if img is None:
            continue
        if emotion:
            print(f"         → Detected: {emotion.upper()} {EMOTION_EMOJI.get(emotion,'')}")
        else:
            print("         → No face detected")

        # FIX 13: save only for first image when --save is used;
        #          for folder mode save with auto-name
        if args.save:
            if len(image_paths) == 1:
                out_path = args.save
            else:
                stem = Path(path).stem
                out_path = str(Path(args.save).parent / f"{stem}_fer{Path(args.save).suffix}")
            cv2.imwrite(out_path, img)
            print(f"         → Saved: {out_path}")

        win_title = f"[{i+1}/{len(image_paths)}] {Path(path).name} — press any key"
        cv2.imshow(win_title, img)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:          # ESC to stop early
            print("[INFO] Stopped by user.")
            break

    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
