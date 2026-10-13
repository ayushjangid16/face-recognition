"""Per-scan state machine.

POSITIONING -> (CHALLENGE ->) VERIFYING -> MARKED
                   \\_________________\\__> FAILED

POSITIONING : N consecutive good frontal frames. Encodings are collected and identified 1:N against the
              whole gallery (face_distance, TOLERANCE + IDENTIFY_MARGIN). No userId is provided by the
              client: the identified member is ADOPTED by the session.
CHALLENGE   : OPTIONAL (CHALLENGE_STEPS=0 — the default — skips it): server-chosen random head turns,
              a liveness check that flat photos/screens cannot pass. When enabled the user must
              perform every turn before VERIFYING starts.
VERIFYING   : N more consecutive good frontal frames. These must identify the SAME member as before the
              challenge, which proves the person who finished the scan is the person who started it.
MARKED      : attendance written for the identified member. Confidence/scores are logged but the raw
              distances are never sent to the client (don't help attackers tune).
"""
from __future__ import annotations

import logging
import secrets
import time
from enum import Enum
from typing import List, Optional

import numpy as np

from .config import settings as S
from .quality import Analysis, quality_problem

log = logging.getLogger("attendance.session")


class State(str, Enum):
    POSITIONING = "POSITIONING"
    CHALLENGE = "CHALLENGE"
    VERIFYING = "VERIFYING"
    MARKED = "MARKED"
    FAILED = "FAILED"


TERMINAL = {State.MARKED, State.FAILED}
FAIL_GENERIC = "We couldn't verify you. Please see the front desk."


class ScanSession:
    def __init__(self, member_id: str, name: str, steps: List[str]):
        self.token = secrets.token_urlsafe(24)
        self.member_id = member_id
        self.name = name
        self.steps = steps                       # e.g. ["left", "right"]
        self.created = time.monotonic()
        self.state = State.POSITIONING
        self.attached = False                    # a token can be used by one websocket only
        self.verification_failed = False         # true only for identity mismatch (counted by the abuse guard)

        self.good_streak = 0
        self.prev_center = None
        self.pre_embs: List[np.ndarray] = []
        self.post_embs: List[np.ndarray] = []
        self.identified: Optional[tuple] = None       # (member_id, name, confidence) of the last passed window

        self.step_idx = 0
        self.step_phase = "turn"                 # "turn" -> "return"
        self.challenge_started: Optional[float] = None
        self._last: Optional[Analysis] = None

    # ------------------------------------------------------------------ utils
    @property
    def expired(self) -> bool:
        return time.monotonic() - self.created > S.session_timeout_s

    def _reply(self, ok: bool, message: str, progress: float = 0.0, **extra) -> dict:
        direction = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
        out = {
            "state": self.state.value,
            "ok": ok,
            "message": message,
            "progress": round(max(0.0, min(1.0, progress)), 3),
            "challenge": {
                "step": min(self.step_idx + 1, len(self.steps)),
                "total": len(self.steps),
                "direction": direction,
                "phase": self.step_phase,
            },
        }
        if S.debug and self._last is not None:
            out["metrics"] = self._last.metrics()
        out.update(extra)
        return out

    def _fail(self, message: str, mismatch: bool = False) -> dict:
        self.state = State.FAILED
        self.verification_failed = mismatch
        return self._reply(False, message)

    def _verify(self, embs: List[np.ndarray], ctx):
        """Identify one stable window of encodings 1:N (no userId involved).

        Every encoding is matched against the whole gallery via Gallery.identify() (face_distance,
        TOLERANCE, IDENTIFY_MARGIN). The window passes only if at least min_frame_match_ratio of the
        frames identify the SAME member. During VERIFYING that member must also be the one adopted at
        POSITIONING, which proves the person who finished the challenge is the person who started it.

        Side effect: self.identified holds (member_id, name, confidence) of the winning member when
        (and only when) the window passes.
        """
        matches = [ctx.gallery.identify(e) for e in embs]
        self.identified = None
        ids = [m.member_id for m in matches if m.member_id]
        if not ids:
            return False, -1.0, 0.0, 0.0, []
        top = max(set(ids), key=ids.count)                       # most frequent identification wins
        hits = [m for m in matches if m.member_id == top]
        ratio = len(hits) / max(len(matches), 1)
        score = float(np.mean([m.similarity for m in hits]))
        finite = [m.margin for m in hits if np.isfinite(m.rival_distance)]
        margin = float(np.mean(finite)) if finite else 0.0
        rivals = sorted({m.rival_id for m in matches if m.rival_id and m.member_id != top})
        expected = self.member_id if self.state == State.VERIFYING else None
        ok = ratio >= S.min_frame_match_ratio and (expected is None or top == expected)
        if ok:
            self.identified = (top, hits[0].name or "", float(np.mean([m.confidence for m in hits])))
        return ok, score, margin, ratio, rivals

    # ------------------------------------------------------------------- main
    def process(self, frame, ctx) -> dict:
        """Handle one frame. ctx needs .engine (analyze/embed), .gallery (decide/mark_attendance), .objects."""
        if self.state in TERMINAL:
            return self._reply(self.state == State.MARKED, "Done.")
        now = time.monotonic()
        if now - self.created > S.session_timeout_s:
            return self._fail("Timed out. Please try again.")

        a = ctx.engine.analyze(frame)
        self._last = a
        if self.state == State.CHALLENGE:
            return self._challenge(a, now)
        return self._frontal(frame, a, ctx)

    # ------------------------------------------------ POSITIONING / VERIFYING
    def _frontal(self, frame, a: Analysis, ctx) -> dict:
        collecting = self.pre_embs if self.state == State.POSITIONING else self.post_embs
        problem = quality_problem(a, self.prev_center, strict=True)
        self.prev_center = a.center

        if problem is None and ctx.objects is not None:
            obj = ctx.objects.find(frame, a.bbox)
            if obj:
                problem = f"Please remove the {obj} from view."

        if problem:
            self.good_streak = 0
            collecting.clear()                   # the stable window must be unbroken
            return self._reply(False, problem)

        collecting.append(ctx.engine.embed(frame, a))
        self.good_streak += 1
        progress = self.good_streak / S.stable_frames
        if self.good_streak < S.stable_frames:
            return self._reply(True, "Hold still...", progress)

        # stable window complete -> identify the face (1:N)
        ok, score, margin, ratio, rivals = self._verify(collecting, ctx)
        phase = "pre" if self.state == State.POSITIONING else "post"
        log.info("member=%s phase=%s ok=%s score=%.3f margin=%.3f frame_ratio=%.2f rivals=%s",
                 self.member_id or "unidentified", phase, ok, score, margin, ratio, rivals)
        if not ok:
            return self._fail(FAIL_GENERIC, mismatch=True)

        if self.state == State.POSITIONING:
            # adopt the identified member; apply the guards the old claimed-ID flow applied at scan start
            ident_id, ident_name, conf = self.identified or (None, None, 0.0)
            if ident_id is None:                                 # defensive: a passing window implies identified
                return self._fail(FAIL_GENERIC, mismatch=True)
            guard = getattr(ctx, "guard", None)
            if guard is not None and guard.locked(ident_id):
                return self._fail("Too many failed attempts. Please see the front desk.")
            last = ctx.gallery.last_attendance_ts(ident_id)
            if last and time.time() - last < S.cooldown_min * 60:
                self.state = State.FAILED
                return self._reply(False, "You're already checked in (at %s). No need to scan again."
                                          % time.strftime("%H:%M", time.localtime(last)))
            self.member_id, self.name = ident_id, ident_name
            log.info("adopted member=%s confidence=%.2f", ident_id, conf)
            self.good_streak = 0
            self.prev_center = None
            if not self.steps:                                   # CHALLENGE_STEPS=0: no liveness challenge
                self.state = State.VERIFYING
                return self._reply(True, "Recognised. Please keep looking straight and hold still.")
            self.state = State.CHALLENGE
            self.challenge_started = time.monotonic()
            return self._reply(True, f"Great. Now slowly turn your head to your {self.steps[0]}.")

        # post-challenge identification passed -> mark attendance for the identified member
        all_embs = self.pre_embs + self.post_embs
        _, score, margin, _, _ = self._verify(all_embs, ctx)
        ts = ctx.gallery.mark_attendance(self.member_id, score, margin, self.token)
        self.state = State.MARKED
        confidence = round(float(self.identified[2]), 2) if self.identified else 0.0
        return self._reply(True, "Attendance marked.", 1.0,
                           member={"id": self.member_id, "name": self.name}, time=ts, confidence=confidence)

    # --------------------------------------------------------------- CHALLENGE
    def _challenge(self, a: Analysis, now: float) -> dict:
        total = len(self.steps)
        if now - (self.challenge_started or now) > S.challenge_timeout_s:
            return self._fail("Liveness check timed out. Please try again.")
        if a.n_faces == 0:
            return self._reply(False, "Keep your face in view.", self._challenge_progress())
        if a.n_faces > 1:
            return self._reply(False, "Only one person should be in the frame.", self._challenge_progress())
        if a.det_score < S.det_thresh:
            return self._reply(False, "Keep your face in view.", self._challenge_progress())

        direction = self.steps[self.step_idx]
        sign = 1.0 if direction == "left" else -1.0      # yaw > 0 == user turned to THEIR left
        y = a.yaw

        if self.step_phase == "turn":
            if y * sign >= S.challenge_yaw:
                self.step_phase = "return"
                return self._reply(True, "Good. Now look straight at the camera.", self._challenge_progress())
            if -y * sign >= S.challenge_yaw:
                return self._reply(True, f"Other way: turn to your {direction}.", self._challenge_progress())
            return self._reply(True, f"Slowly turn your head to your {direction}.", self._challenge_progress())

        # phase == "return"
        if abs(y) <= S.return_yaw:
            self.step_idx += 1
            self.step_phase = "turn"
            if self.step_idx >= total:
                self.state = State.VERIFYING
                self.good_streak = 0
                self.prev_center = None
                return self._reply(True, "Look straight and hold still.", 0.0)
            return self._reply(True, f"Now turn your head to your {self.steps[self.step_idx]}.", self._challenge_progress())
        return self._reply(True, "Look straight at the camera.", self._challenge_progress())

    def _challenge_progress(self) -> float:
        total = max(len(self.steps), 1)
        return (self.step_idx + (0.5 if self.step_phase == "return" else 0.0)) / total
