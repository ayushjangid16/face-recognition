"""Minimal end-to-end test of the face_recognition identification pipeline (no server needed).

Folder layout:
    dataset/
        alice/ 1.jpg 2.jpg ...      # enrolled members: folder name = member_id
        bob/  1.jpg ...
        _unknown/ face.jpg          # a face that is present at scan time but NEVER enrolled
        _noface/  blank.jpg         # an image with no face at all (e.g. an empty/landscape photo)

Usage (from the backend/ folder):
    python -m scripts.test_identify dataset/

What it checks:
  1. every enrolment photo encodes to exactly 128 floats;
  2. scanning an image of an enrolled user returns that user's member_id (no userId passed!);
  3. scanning an image of an un-enrolled face returns "Unknown" (never a wrong member);
  4. an image with no face returns a clean "no face detected" result, without raising.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from app.config import settings as S
from app.pipeline import FaceEngine, load_upload_image
from app.store import Gallery, UNKNOWN


def encode_face(engine: FaceEngine, path: Path, jitters: int):
    """(analysis, encoding) for one photo, following the enrolment path. (None, None) = no face."""
    img = load_upload_image(path.read_bytes())               # RGB via face_recognition.load_image_file
    if img is None:
        raise ValueError(f"cannot decode {path}")
    a = engine.analyze(img)
    if a.n_faces == 0:
        return None, None
    if a.n_faces != 1:
        raise ValueError(f"{a.n_faces} significant faces in {path}; use a photo with one face")
    return a, engine.embed(img, a, jitters)


def identify_face(engine: FaceEngine, gallery: Gallery, path: Path):
    """Simulate a scan of one image: returns a label string, like the WebSocket flow would."""
    img = load_upload_image(path.read_bytes())
    if img is None:
        return "cannot decode image"
    a = engine.analyze(img)
    if a.n_faces == 0:
        return "no face detected"                            # clean result, no exception
    if a.n_faces != 1:
        return "only one person should be in the frame"
    emb = engine.embed(img, a, 1)                            # scanning uses num_jitters=1
    m = gallery.identify(emb)                                # 1:N — no userId anywhere
    if m.member_id is None:
        return UNKNOWN
    return f"{m.member_id} (confidence {m.confidence}%)"


def main(root: str) -> None:
    root = Path(root)
    engine = FaceEngine()
    gallery = Gallery(str(Path(tempfile.mkdtemp()) / "gallery.db"))   # throw-away DB, never the real one
    failures = 0

    people = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if len(people) < 2:
        sys.exit("Need at least 2 enrolled members under " + str(root))
    for person in people:
        gallery.add_member(person.name, person.name.title())
        for img_path in sorted(p for p in person.iterdir() if p.is_file()):
            _, enc = encode_face(engine, img_path, S.enroll_jitters)
            if enc is None:
                print(f"FAIL  enrol {person.name}/{img_path.name}: no face detected")
                failures += 1
                continue
            assert enc.shape == (128,), f"expected a 128-d encoding, got {enc.shape}"
            dup_sim, dup_id = gallery.best_other(person.name, enc)
            warn = f"  WARNING: looks like {dup_id}" if dup_sim >= 1.0 - S.tolerance else ""
            gallery.add_embedding(person.name, enc)
            print(f"ok    enrolled {person.name}/{img_path.name} (128-d encoding){warn}")

    # 1. an enrolled face must be identified as its own member — with no userId given
    for person in people:
        img_path = next(p for p in sorted(person.iterdir()) if p.is_file())
        label = identify_face(engine, gallery, img_path)
        if label.startswith(person.name):
            print(f"ok    scan {img_path.name} -> {label}")
        else:
            print(f"FAIL  scan {img_path.name} -> {label!r}, expected {person.name}")
            failures += 1

    # 2. an un-enrolled face must come back Unknown, never as some wrong member
    unknown_dir = root / "_unknown"
    for img_path in sorted(p for p in unknown_dir.iterdir() if p.is_file()) if unknown_dir.is_dir() else []:
        label = identify_face(engine, gallery, img_path)
        if label == UNKNOWN:
            print(f"ok    scan {img_path.name} -> {label}")
        else:
            print(f"FAIL  scan {img_path.name} -> {label!r}, expected {UNKNOWN}")
            failures += 1

    # 3. an image with no face must produce a clean result, not an exception
    noface_dir = root / "_noface"
    for img_path in sorted(p for p in noface_dir.iterdir() if p.is_file()) if noface_dir.is_dir() else []:
        label = identify_face(engine, gallery, img_path)
        if label == "no face detected":
            print(f"ok    scan {img_path.name} -> {label}")
        else:
            print(f"FAIL  scan {img_path.name} -> {label!r}, expected 'no face detected'")
            failures += 1

    print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
