"""Pure-numpy quality gates and head-pose geometry. No model code here, so it is easy to unit test."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import settings as S


@dataclass
class Analysis:
    frame_w: int = 640
    frame_h: int = 480
    n_faces: int = 0                       # number of *significant* faces (small background faces are ignored)
    bbox: Optional[Tuple[float, float, float, float]] = None
    kps: Optional[np.ndarray] = None       # 5 landmarks: eye(img-left), eye(img-right), nose, mouth(img-left), mouth(img-right)
    det_score: float = 0.0
    yaw: float = 0.0                       # >0 means the user turned toward THEIR left
    pitch: float = 0.6
    roll: float = 0.0                      # degrees
    blur: float = 0.0
    brightness: float = 128.0
    contrast: float = 50.0

    @property
    def width_frac(self) -> float:
        if not self.bbox:
            return 0.0
        return (self.bbox[2] - self.bbox[0]) / self.frame_w

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if not self.bbox:
            return None
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2 / self.frame_w, (y1 + y2) / 2 / self.frame_h)

    def metrics(self) -> dict:
        c = self.center or (0, 0)
        return {
            "faces": self.n_faces, "det": round(self.det_score, 2), "width": round(self.width_frac, 2),
            "cx": round(c[0], 2), "cy": round(c[1], 2), "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2), "roll": round(self.roll, 1), "blur": round(self.blur, 1),
            "bright": round(self.brightness, 1), "contrast": round(self.contrast, 1),
        }


def head_pose(kps: np.ndarray) -> Tuple[float, float, float]:
    """Cheap pose proxies from the 5 detector landmarks.

    yaw   : (nose_x - eye_mid_x) / inter_eye_distance. A real 3D head shifts the protruding nose sideways
            when it turns, while a flat photo only foreshortens uniformly, so the ratio stays ~0 for a photo.
    pitch : where the nose sits between the eye line (0) and mouth line (1).
    roll  : angle of the eye line in degrees.
    """
    le, re, nose, lm, rm = [np.asarray(p, dtype=np.float64) for p in kps]
    eye_mid = (le + re) / 2
    mouth_mid = (lm + rm) / 2
    iod = max(float(np.linalg.norm(re - le)), 1e-6)
    yaw = float((nose[0] - eye_mid[0]) / iod)
    em = float(mouth_mid[1] - eye_mid[1])
    pitch = float((nose[1] - eye_mid[1]) / em) if em > 1e-6 else 0.6
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    return yaw, pitch, roll


def quality_problem(a: Analysis, prev_center: Optional[Tuple[float, float]] = None, strict: bool = True) -> Optional[str]:
    """Return a user-facing message for the first failed gate, or None if the frame is good.

    strict=True  : live scan (frontal face, must hold still)
    strict=False : enrolment (slight head turns allowed, no motion check)
    """
    if a.n_faces == 0:
        return "No face detected. Look at the camera."
    if a.n_faces > 1:
        return "Only one person should be in the frame."
    if a.det_score < S.min_det_score:
        return "Your face looks partly hidden. Remove mask, glasses, cap or hand."

    wf = a.width_frac
    if wf < S.min_face_frac:
        return "Move closer to the camera."
    if wf > S.max_face_frac:
        return "Move back a little."
    cx, cy = a.center
    if abs(cx - 0.5) > S.max_center_offset or abs(cy - 0.5) > S.max_center_offset:
        return "Center your face in the oval."

    yaw_tol = S.frontal_yaw if strict else S.enroll_yaw
    p_min = S.pitch_min if strict else S.enroll_pitch_min
    p_max = S.pitch_max if strict else S.enroll_pitch_max
    roll_tol = S.max_roll if strict else S.enroll_max_roll
    if abs(a.yaw) > yaw_tol:
        return "Look straight at the camera."
    if a.pitch < p_min:
        return "Lower your chin slightly."
    if a.pitch > p_max:
        return "Raise your chin slightly."
    if abs(a.roll) > roll_tol:
        return "Keep your head upright."

    if a.brightness < S.min_brightness:
        return "Too dark. Face a light source."
    if a.brightness > S.max_brightness:
        return "Too bright. Avoid strong light behind or on your face."
    if a.contrast < S.min_contrast:
        return "Lighting is too flat. Improve the lighting on your face."

    if strict and prev_center is not None:
        move = math.hypot(cx - prev_center[0], cy - prev_center[1])
        if move > S.max_motion:
            return "Hold still."
    if a.blur < S.min_sharpness:
        return "Image is blurry. Hold still and wipe the camera lens."
    return None
