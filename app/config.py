"""All tunable settings. Every value can be overridden with an environment variable.

IMPORTANT: the thresholds below are sensible starting points, not calibrated values.
Run scripts/calibrate.py on photos of YOUR members and set MATCH_THRESHOLD from the result.
"""
import os
from dataclasses import dataclass


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # ---- general -------------------------------------------------------
    data_dir: str = os.getenv("DATA_DIR", "./data")
    admin_key: str = os.getenv("ADMIN_KEY", "ADMIN")
    cors_origins: tuple = tuple(
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:10000,http://127.0.0.1:10000,http://0.0.0.0:10000").split(",") if o.strip()
    )
    debug: bool = os.getenv("DEBUG_METRICS", "0") == "1"  # adds raw metrics to WS replies (for tuning)

    # ---- models (CPU) --------------------------------------------------
    model_pack: str = os.getenv("MODEL_PACK", "buffalo_l")  # buffalo_s is ~2-3x faster, a bit less accurate
    det_size: int = _i("DET_SIZE", 320)
    det_thresh: float = _f("DET_THRESH", 0.5)
    max_concurrent_inference: int = _i("MAX_CONCURRENT_INFERENCE", 2)

    # ---- recognition (128-d face_recognition/dlib encodings, matched by Euclidean face_distance)
    tolerance: float = _f("TOLERANCE", 0.5)               # a face matches a member if distance <= TOLERANCE
                                                          # (0.6 is the library default; 0.5 avoids false positives)
    identify_margin: float = _f("IDENTIFY_MARGIN", 0.05)  # best member must beat the runner-up by this much
    detect_scale: float = _f("DETECT_SCALE", 0.5)         # frames are downscaled before (cheap) HOG detection
    enroll_jitters: int = _i("ENROLL_JITTERS", 5)         # face_encodings jitters at enrolment (scans use 1)
    min_frame_match_ratio: float = _f("MIN_FRAME_MATCH_RATIO", 0.8)  # share of frames that must identify the same member
    min_enroll_samples: int = _i("MIN_ENROLL_SAMPLES", 5)
    # legacy cosine-era thresholds, kept for compatibility with env files; matching now uses TOLERANCE/IDENTIFY_MARGIN
    match_threshold: float = _f("MATCH_THRESHOLD", 0.45)
    rival_margin: float = _f("RIVAL_MARGIN", 0.08)         # still used by the legacy 1:1 Gallery.decide()
    own_consistency: float = _f("OWN_CONSISTENCY", 0.30)

    # ---- quality gates -------------------------------------------------
    min_det_score: float = _f("MIN_DET_SCORE", 0.70)       # low score => occluded / odd face
    min_face_frac: float = _f("MIN_FACE_FRAC", 0.28)       # face width / frame width
    max_face_frac: float = _f("MAX_FACE_FRAC", 0.65)
    max_center_offset: float = _f("MAX_CENTER_OFFSET", 0.15)
    frontal_yaw: float = _f("FRONTAL_YAW", 0.10)           # nose offset / inter-eye distance (~13 deg)
    pitch_min: float = _f("PITCH_MIN", 0.45)               # nose position between eye line and mouth line
    pitch_max: float = _f("PITCH_MAX", 0.85)
    max_roll: float = _f("MAX_ROLL", 15.0)                 # degrees
    min_brightness: float = _f("MIN_BRIGHTNESS", 55.0)
    max_brightness: float = _f("MAX_BRIGHTNESS", 205.0)
    min_contrast: float = _f("MIN_CONTRAST", 22.0)
    min_sharpness: float = _f("MIN_SHARPNESS", 40.0)       # variance of Laplacian on a 112px face crop
    max_motion: float = _f("MAX_MOTION", 0.03)             # face centre movement per frame (fraction of frame)
    # relaxed limits used while enrolling (staff asks for slight head turns on purpose)
    enroll_yaw: float = _f("ENROLL_YAW", 0.25)
    enroll_pitch_min: float = _f("ENROLL_PITCH_MIN", 0.35)
    enroll_pitch_max: float = _f("ENROLL_PITCH_MAX", 0.95)
    enroll_max_roll: float = _f("ENROLL_MAX_ROLL", 25.0)

    # ---- scan session --------------------------------------------------
    stable_frames: int = _i("STABLE_FRAMES", 4)            # consecutive good frames needed per window
    challenge_steps: int = _i("CHALLENGE_STEPS", 0)        # head-turn liveness challenge; 0 = OFF (fast,
                                                           # still-face check-in). Set 2+ to re-enable the
                                                           # anti-photo/anti-screen defence.
    challenge_yaw: float = _f("CHALLENGE_YAW", 0.22)       # ~27 deg turn
    return_yaw: float = _f("RETURN_YAW", 0.12)
    challenge_timeout_s: float = _f("CHALLENGE_TIMEOUT_S", 15.0)
    session_timeout_s: float = _f("SESSION_TIMEOUT_S", 45.0)
    cooldown_min: int = _i("ATTENDANCE_COOLDOWN_MIN", 240)  # no second check-in within this window

    # ---- abuse protection ---------------------------------------------
    max_failures: int = _i("MAX_FAILURES", 5)
    failure_window_s: int = _i("FAILURE_WINDOW_S", 600)
    starts_per_min: int = _i("STARTS_PER_MIN", 20)

    # ---- optional object detector (phones, books, laptops in front of the face)
    object_detection: bool = os.getenv("OBJECT_DETECTION", "0") == "1"
    yolo_model: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    object_conf: float = _f("OBJECT_CONF", 0.40)


settings = Settings()
