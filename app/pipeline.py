"""Model wrappers: face_recognition (dlib HOG) detection + 128-d encodings (CPU), and an optional YOLO object detector.

Colour convention: every image handed around this app is BGR (that is what cv2 produces via
imdecode/imread/cvtColor and what the quality metrics and the YOLO guard expect). face_recognition
expects C-contiguous uint8 RGB, so the engine converts BGR->RGB exactly once, right before calling
the library. Uploaded enrolment photos are loaded with face_recognition.load_image_file (which
returns RGB) and converted once to BGR here, so the net result the library sees is the original
RGB — channels are never swapped twice.
"""
from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from .config import settings as S
from .quality import Analysis, head_pose


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to the C-contiguous uint8 RGB that face_recognition requires."""
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_upload_image(data: bytes, max_bytes: int = 2_000_000, max_width: int = 960) -> np.ndarray | None:
    """Decode an uploaded enrolment photo with face_recognition.load_image_file (RGB).

    Returns a BGR image like decode_image() in main.py (the app-wide convention), or None for
    missing / oversized / unreadable data. load_image_file already returns RGB, so the image is
    only re-scaled here; the engine converts it back to RGB internally before encoding.
    """
    if not data or len(data) > max_bytes:
        return None
    import face_recognition  # lazy import so pure logic stays testable without the models

    fd, path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        img = face_recognition.load_image_file(path)          # RGB, no channel conversion needed
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        return None
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    return bgr_to_rgb(np.ascontiguousarray(img, dtype=np.uint8))   # RGB -> BGR for the app-wide convention


def five_pose_points(landmarks: dict) -> np.ndarray:
    """Derive the 5 pose landmarks quality.head_pose() expects from dlib's 68-point model.

    Layout: [eye(img-left), eye(img-right), nose tip, mouth(img-left), mouth(img-right)].
    """
    le = np.mean(landmarks["left_eye"], axis=0)       # the PERSON'S left eye = image-right when facing the camera
    re = np.mean(landmarks["right_eye"], axis=0)
    eye_l, eye_r = (le, re) if le[0] <= re[0] else (re, le)
    nose = np.mean(landmarks["nose_tip"], axis=0)
    lips = np.asarray(landmarks["top_lip"] + landmarks["bottom_lip"], dtype=np.float64)
    mouth_l, mouth_r = lips[lips[:, 0].argmin()], lips[lips[:, 0].argmax()]
    return np.asarray([eye_l, eye_r, nose, mouth_l, mouth_r], dtype=np.float64)


class FaceEngine:
    """Detection + 128-d encodings from the `face_recognition` library (dlib), CPU only (hog).

    Keeps the analyze()/embed() contract the rest of the app relies on: analyze() fills a
    quality.Analysis for the single significant face (largest first), embed() encodes THAT face —
    so session.py, quality.py and the ObjectDetector need no changes.
    """

    def __init__(self) -> None:
        # imported lazily so the pure logic (quality.py / session.py / tests) can be tested without the models
        import face_recognition

        self._fr = face_recognition

    # ----------------------------------------------------------------- detect
    def analyze(self, frame: np.ndarray) -> Analysis:
        """Detect the main face in a BGR frame. Cheap path: HOG on a downscaled copy."""
        h, w = frame.shape[:2]
        a = Analysis(frame_w=w, frame_h=h)
        rgb = bgr_to_rgb(frame)
        scale = max(0.1, min(1.0, S.detect_scale))
        small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else rgb

        # hog for CPU (cnn only pays off with a GPU, which this project does not use)
        locations = self._fr.face_locations(small, model="hog", number_of_times_to_upsample=1)
        if not locations:
            return a

        # Ignore small faces far behind the user (other gym members): only count faces at least half as
        # wide as the biggest one.
        boxes = [(t / scale, r / scale, b / scale, l / scale) for (t, r, b, l) in locations]
        widths = [x2 - x1 for (x1, _, x2, _) in boxes]
        main = int(np.argmax(widths))
        a.n_faces = sum(1 for wd in widths if wd >= 0.5 * widths[main])
        if a.n_faces != 1:
            return a

        top, right, bottom, left = (int(round(v)) for v in boxes[main])
        a.bbox = (float(left), float(top), float(right), float(bottom))
        a.det_score = 1.0     # dlib's HOG detector exposes no confidence; a detection is a pass
        # landmarks on the SMALL frame (the shape predictor's cost scales with the box), then scale back
        lm = self._fr.face_landmarks(small, [locations[main]])   # only the main face
        if lm:
            a.kps = five_pose_points(lm[0]) / scale
            a.yaw, a.pitch, a.roll = head_pose(a.kps)
        self._image_metrics(frame, a)
        return a

    @staticmethod
    def _image_metrics(frame: np.ndarray, a: Analysis) -> None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in a.bbox)
        mx, my = int(0.1 * (x2 - x1)), int(0.1 * (y2 - y1))
        crop = frame[max(0, y1 - my): min(h, y2 + my), max(0, x1 - mx): min(w, x2 + mx)]
        if crop.size == 0:
            return
        gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)  # fixed size => comparable scores
        a.blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        a.brightness = float(gray.mean())
        a.contrast = float(gray.std())

    # ------------------------------------------------------------------ embed
    def embed(self, frame: np.ndarray, a: Analysis, num_jitters: int = 1) -> np.ndarray:
        """128-d dlib encoding of the (already detected) face in a BGR frame.

        num_jitters=1 keeps live scanning fast; the enrolment endpoint passes S.enroll_jitters
        for more robust encodings.
        """
        if a.bbox is None:
            raise RuntimeError("embed() needs a detected face (call analyze() first)")
        rgb = bgr_to_rgb(frame)
        x1, y1, x2, y2 = a.bbox
        loc = (int(round(y1)), int(round(x2)), int(round(y2)), int(round(x1)))   # (top, right, bottom, left)
        encodings = self._fr.face_encodings(rgb, known_face_locations=[loc], num_jitters=num_jitters)
        if not encodings:
            raise RuntimeError("face_encodings returned no encoding for an existing detection")
        return np.asarray(encodings[0], dtype=np.float32)


class ObjectDetector:
    """Optional: flags phones / screens / books / laptops held in front of the face (COCO classes).

    Needs `pip install ultralytics` and OBJECT_DETECTION=1. Costs ~60-100 ms per checked frame on CPU.
    This is a coarse guard; for gym-specific objects fine-tune YOLO on your own images.
    """

    BLOCK = {62: "screen", 63: "laptop", 67: "phone", 73: "book"}

    def __init__(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(S.yolo_model)

    def find(self, frame: np.ndarray, bbox) -> str | None:
        res = self.model.predict(
            frame, imgsz=320, conf=S.object_conf, classes=list(self.BLOCK), verbose=False, device="cpu"
        )[0]
        fx1, fy1, fx2, fy2 = bbox
        farea = max(1.0, (fx2 - fx1) * (fy2 - fy1))
        fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            ix = max(0.0, min(x2, fx2) - max(x1, fx1))
            iy = max(0.0, min(y2, fy2) - max(y1, fy1))
            if ix * iy / farea >= 0.05 or (x1 <= fcx <= x2 and y1 <= fcy <= y2):
                return self.BLOCK[int(b.cls[0])]
        return None
