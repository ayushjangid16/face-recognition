"""Calibrate TOLERANCE on YOUR data.

Folder layout (use real photos of real members, ideally taken with the gym's own webcam):
    dataset/
        member_001/ a.jpg b.jpg c.jpg ...
        member_002/ ...
Include look-alikes / siblings if you have them: they are exactly the cases that matter.

Usage (from the backend/ folder):
    python -m scripts.calibrate dataset/

It prints, for several target false-accept rates (FAR), the maximum face_recognition
face_distance to accept and the resulting false-reject rate (FRR) for genuine attempts.
"""
import itertools
import sys
from pathlib import Path

import face_recognition
import numpy as np

from app.config import settings as S
from app.pipeline import FaceEngine, load_upload_image
from app.quality import quality_problem


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance used by face_recognition (only matching rule in the system)."""
    return float(face_recognition.face_distance([np.asarray(b, dtype=np.float64)], a)[0])


def main(root: str) -> None:
    engine = FaceEngine()
    encs: dict[str, list] = {}
    skipped = 0
    for person in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for img_path in sorted(person.glob("*")):
            img = load_upload_image(img_path.read_bytes())       # same decode path as enrolment
            if img is None:
                skipped += 1
                continue
            a = engine.analyze(img)
            if quality_problem(a, None, strict=False):   # only evaluate usable photos, like enrolment does
                skipped += 1
                continue
            encs.setdefault(person.name, []).append(engine.embed(img, a, S.enroll_jitters))
    encs = {k: v for k, v in encs.items() if v}
    print(f"people: {len(encs)}  photos kept: {sum(len(v) for v in encs.values())}  skipped (bad quality): {skipped}")

    genuine, impostor = [], []
    for _, vs in encs.items():
        genuine += [distance(a, b) for a, b in itertools.combinations(vs, 2)]
    people = list(encs)
    for i, p in enumerate(people):
        for q in people[i + 1:]:
            impostor += [distance(a, b) for a in encs[p] for b in encs[q]]
    genuine, impostor = np.array(genuine), np.array(impostor)
    if len(genuine) == 0 or len(impostor) == 0:
        print("Need at least 2 people with 2+ usable photos each.")
        return

    print(f"genuine pairs: {len(genuine)}  impostor pairs: {len(impostor)}")
    print(f"genuine  mean={genuine.mean():.3f}  p95={np.percentile(genuine, 95):.3f}  max={genuine.max():.3f}")
    print(f"impostor mean={impostor.mean():.3f}  p5={np.percentile(impostor, 5):.3f}  min={impostor.min():.3f}\n")
    print(f"{'target FAR':>12} {'max distance':>13} {'FRR':>8}   note")
    for far in (1e-2, 1e-3, 1e-4, 1e-5):
        thr = float(np.quantile(impostor, far))          # accept when distance <= thr
        frr = float((genuine > thr).mean())
        note = "" if len(impostor) >= 10 / far else "(too few impostor pairs to trust this row)"
        print(f"{far:>12.0e} {thr:>13.3f} {frr:>8.2%}   {note}")
    print("\nPick the row matching your risk appetite, subtract a small safety buffer, and set TOLERANCE")
    print(f"(current value: {S.tolerance}).")
    print("Note: the live system is stricter than this table because it also applies the IDENTIFY_MARGIN,")
    print("the liveness challenge and multi-frame voting.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
