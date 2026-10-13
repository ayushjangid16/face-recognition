"""Model-free tests of the pose geometry and the scan state machine. Run:  python -m tests.test_logic"""
import time
from types import SimpleNamespace

import numpy as np

from app.config import settings as S
from app.quality import Analysis, head_pose, quality_problem
from app.session import FAIL_GENERIC, ScanSession, State
from app.store import Match


def good(yaw=0.0, cx=0.5, cy=0.5, **kw) -> Analysis:
    a = Analysis(frame_w=640, frame_h=480, n_faces=1,
                 bbox=(640 * (cx - 0.2), 480 * (cy - 0.3), 640 * (cx + 0.2), 480 * (cy + 0.3)),
                 kps=None, det_score=0.9, yaw=yaw, pitch=0.62, roll=0.0, blur=120, brightness=120, contrast=50)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class FakeEngine:
    def __init__(self):
        self.queue = []

    def analyze(self, frame):
        return self.queue.pop(0)

    def embed(self, frame, a):
        return np.ones(4, np.float32) / 2


class FakeGallery:
    """Stands in for the 1:N identification the real Gallery does with face_distance."""

    def __init__(self, ok=True):
        self.ok = ok                      # does the current face match ANY member at all?
        self.as_id = "m1"                 # which member the current face is identified as
        self.marked = []

    def identify(self, q):
        if not self.ok:
            return Match(None, None, 0.9)                       # unknown face (nothing within TOLERANCE)
        return Match(self.as_id, "Test Member", 0.3, "other", 0.8)

    def last_attendance_ts(self, member_id):
        return None

    def mark_attendance(self, member_id, score, margin, session):
        self.marked.append(member_id)
        return "2026-10-12T13:00:00+05:30"


def run(sess, ctx, analyses):
    out = None
    for a in analyses:
        ctx.engine.queue.append(a)
        out = sess.process(None, ctx)
    return out


def make(steps=("left", "right"), ok=True):
    ctx = SimpleNamespace(engine=FakeEngine(), gallery=FakeGallery(ok), objects=None)
    return ScanSession("m1", "Test Member", list(steps)), ctx


def test_pose_geometry():
    # frontal face: nose centred between the eyes
    le, re, nose, lm, rm = (260, 200), (380, 200), (320, 254), (275, 290), (365, 290)
    yaw, pitch, roll = head_pose(np.array([le, re, nose, lm, rm], float))
    assert abs(yaw) < 1e-6 and 0.5 < pitch < 0.7 and abs(roll) < 1e-6
    # nose shifted toward image-right => user turned toward THEIR left => positive yaw
    yaw_l, _, _ = head_pose(np.array([le, re, (350, 254), lm, rm], float))
    assert yaw_l > 0.2
    # flat photo rotated about vertical axis: all x shrink uniformly -> yaw proxy unchanged
    shrink = lambda p: (320 + (p[0] - 320) * 0.7, p[1])
    yaw_p, _, _ = head_pose(np.array([shrink(p) for p in (le, re, nose, lm, rm)], float))
    assert abs(yaw_p) < 1e-6


def test_gates():
    assert quality_problem(good()) is None
    assert "closer" in quality_problem(good(bbox=(300, 220, 340, 260)))
    assert quality_problem(Analysis(n_faces=2)).startswith("Only one")
    assert "Look straight" in quality_problem(good(yaw=0.3))
    assert "blurry" in quality_problem(good(blur=5))
    assert "Too dark" in quality_problem(good(brightness=20))
    assert "Hold still" in quality_problem(good(cx=0.58), prev_center=(0.5, 0.5))
    assert quality_problem(good(yaw=0.2), strict=False) is None      # enrolment allows slight turns


def test_happy_path():
    sess, ctx = make()
    n = S.stable_frames
    out = run(sess, ctx, [good()] * n)
    assert sess.state == State.CHALLENGE, out
    # steps are left then right; yaw > 0 is "left"
    out = run(sess, ctx, [good(yaw=0.0), good(yaw=0.3), good(yaw=0.05)])
    assert sess.step_idx == 1, out
    out = run(sess, ctx, [good(yaw=-0.3), good(yaw=0.0)])
    assert sess.state == State.VERIFYING, out
    out = run(sess, ctx, [good()] * n)
    assert sess.state == State.MARKED and out["member"]["id"] == "m1" and ctx.gallery.marked == ["m1"]
    assert out["confidence"] > 0                                       # identification confidence is returned
    assert "score" not in out                                          # raw scores must not reach the client


def test_no_challenge_flow_is_fast():
    sess, ctx = make(steps=[])                                  # CHALLENGE_STEPS=0 disables the liveness challenge
    out = run(sess, ctx, [good()] * S.stable_frames)
    assert sess.state == State.VERIFYING, out                    # straight from positioning to verifying
    out = run(sess, ctx, [good()] * S.stable_frames)
    assert sess.state == State.MARKED and ctx.gallery.marked == ["m1"]


def test_wrong_direction_does_not_advance():
    sess, ctx = make(steps=("left",))
    run(sess, ctx, [good()] * S.stable_frames)
    out = run(sess, ctx, [good(yaw=-0.35)] * 5)                        # turned RIGHT when LEFT was asked
    assert sess.step_idx == 0 and sess.step_phase == "turn" and "Other way" in out["message"]


def test_photo_never_passes_liveness():
    sess, ctx = make(steps=("left",))
    run(sess, ctx, [good()] * S.stable_frames)
    run(sess, ctx, [good(yaw=0.0)] * 20)                               # a flat photo: yaw proxy stays ~0
    assert sess.state == State.CHALLENGE
    sess.challenge_started = time.monotonic() - S.challenge_timeout_s - 1
    ctx.engine.queue.append(good())
    out = sess.process(None, ctx)
    assert sess.state == State.FAILED and "timed out" in out["message"]


def test_identity_mismatch_fails_and_is_counted():
    sess, ctx = make(ok=False)
    out = run(sess, ctx, [good()] * S.stable_frames)
    assert sess.state == State.FAILED and out["message"] == FAIL_GENERIC and sess.verification_failed


def test_swap_after_challenge_is_caught():
    sess, ctx = make(steps=("left",))
    run(sess, ctx, [good()] * S.stable_frames)                         # real member identified and adopted
    run(sess, ctx, [good(yaw=0.3), good(yaw=0.0)])
    assert sess.state == State.VERIFYING and sess.member_id == "m1"
    ctx.gallery.as_id = "m2"                                           # a different person finishes the scan
    out = run(sess, ctx, [good()] * S.stable_frames)
    assert sess.state == State.FAILED and not ctx.gallery.marked


def test_broken_stability_window_resets():
    sess, ctx = make()
    run(sess, ctx, [good()] * (S.stable_frames - 1))
    out = run(sess, ctx, [good(blur=1)])                               # one bad frame
    assert sess.good_streak == 0 and sess.pre_embs == [] and not out["ok"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all passed")
