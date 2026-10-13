"""SQLite-backed gallery of face encodings + attendance log, with an in-memory cache for fast matching.

1000 members x ~7 encodings x 128 floats is only ~3.5 MB, so a plain numpy distance
computation against `face_recognition.face_distance` is instant on CPU. The cache (_snap) is
loaded once at startup and refreshed after every enrolment / update / delete, so scans never
re-encode stored images.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from .config import settings as S

log = logging.getLogger("attendance.store")

EMB_DIM = 128          # face_recognition / dlib encodings are 128-d float32 vectors
UNKNOWN = "Unknown"    # label used when no stored face is within TOLERANCE


@dataclass
class Decision:
    ok: bool
    score: float            # similarity to the claimed member (1 - distance of their closest encoding)
    rival: float            # similarity to the best-matching OTHER member (1 - distance)
    margin: float           # score - rival
    rival_id: Optional[str] = None


@dataclass
class Match:
    """Result of 1:N identification for one face encoding."""

    member_id: Optional[str]     # None => "Unknown" (nothing within TOLERANCE / no margin over the runner-up)
    name: Optional[str]          # member name, when matched
    distance: float              # face_recognition.face_distance to the closest encoding (inf if none)
    rival_id: Optional[str] = None       # runner-up member, when there is one
    rival_distance: float = float("inf")

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance

    @property
    def margin(self) -> float:
        """How much farther the runner-up member's closest encoding is (>= IDENTIFY_MARGIN is required)."""
        return self.rival_distance - self.distance

    @property
    def confidence(self) -> float:
        """confidence = (1 - distance) * 100, rounded to 2 decimals (0 when unmatched)."""
        if self.member_id is None or not np.isfinite(self.distance):
            return 0.0
        return round((1.0 - self.distance) * 100, 2)

    @property
    def label(self) -> str:
        return self.name or (self.member_id or UNKNOWN)


class Gallery:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.lock = threading.RLock()
        self._create_tables()
        self._snap: Tuple[np.ndarray, np.ndarray] = (np.zeros((0, EMB_DIM), np.float32), np.array([], dtype=str))
        self._reload()

    # ------------------------------------------------------------------ setup
    def _create_tables(self) -> None:
        with self.lock, self.db:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS members(
                    member_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS embeddings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_id TEXT NOT NULL REFERENCES members(member_id) ON DELETE CASCADE,
                    vec BLOB NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS attendance(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, member_id TEXT NOT NULL,
                    ts REAL NOT NULL, score REAL, margin REAL, session TEXT);
                CREATE INDEX IF NOT EXISTS idx_att_member_ts ON attendance(member_id, ts);
                """
            )

    def _reload(self) -> None:
        with self.lock:
            rows = self.db.execute("SELECT member_id, vec FROM embeddings ORDER BY id").fetchall()
            names = dict(self.db.execute("SELECT member_id, name FROM members").fetchall())
        good = [r for r in rows if len(r[1]) == EMB_DIM * 4]
        if len(good) != len(rows):
            log.warning("skipped %d embedding(s) with a foreign dimension (re-enrol those members)",
                        len(rows) - len(good))
        if good:
            E = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in good]).astype(np.float32)
            owners = np.array([r[0] for r in good], dtype=str)
        else:
            E, owners = np.zeros((0, EMB_DIM), np.float32), np.array([], dtype=str)
        self._names = names
        self._snap = (E, owners)  # atomic swap; readers always see a consistent pair

    # ---------------------------------------------------------------- members
    def get_member(self, member_id: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT member_id, name FROM members WHERE member_id=?", (member_id,)).fetchone()
        return {"member_id": r[0], "name": r[1]} if r else None

    def add_member(self, member_id: str, name: str) -> bool:
        """True if created, False if it already existed (existing record is left untouched)."""
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO members(member_id, name, created_at) VALUES(?,?,?)",
                (member_id, name, time.time()),
            )
            return cur.rowcount == 1

    def delete_member(self, member_id: str) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM members WHERE member_id=?", (member_id,))
        self._reload()

    def list_members(self) -> List[dict]:
        with self.lock:
            rows = self.db.execute("SELECT member_id, name FROM members ORDER BY created_at DESC").fetchall()
        _, owners = self._snap
        uniq, counts = (np.unique(owners, return_counts=True) if owners.size else ([], []))
        samples = dict(zip(uniq.tolist(), counts.tolist())) if owners.size else {}
        return [
            {"member_id": r[0], "name": r[1], "samples": samples.get(r[0], 0),
             "ready": samples.get(r[0], 0) >= S.min_enroll_samples}
            for r in rows
        ]

    # ------------------------------------------------------------- embeddings
    def count(self, member_id: str) -> int:
        """Number of USABLE (128-d) encodings stored for the member.

        Counts from the in-memory snapshot rather than raw rows, so legacy embeddings of a foreign
        dimension (skipped at load time) are not counted — a member re-enrolling after an upgrade
        reaches `ready` with the right number of usable photos.
        """
        _, owners = self._snap
        return int((owners == member_id).sum())

    def enrolled_count(self, min_samples: int = 1) -> int:
        """Members with at least `min_samples` USABLE encodings, i.e. members a scan can actually find."""
        _, owners = self._snap
        if owners.size == 0:
            return 0
        _, counts = np.unique(owners, return_counts=True)
        return int((counts >= min_samples).sum())

    def add_embedding(self, member_id: str, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).flatten()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO embeddings(member_id, vec, created_at) VALUES(?,?,?)", (member_id, v.tobytes(), time.time())
            )
        self._reload()

    def reset_embeddings(self, member_id: str) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM embeddings WHERE member_id=?", (member_id,))
        self._reload()

    def own_similarity(self, member_id: str, q: np.ndarray) -> Optional[float]:
        """Best similarity (1 - face_distance) between q and the member's OWN encodings."""
        E, owners = self._snap
        mask = owners == member_id
        if not mask.any():
            return None
        import face_recognition
        dists = face_recognition.face_distance(E[mask], np.asarray(q, dtype=np.float64).flatten())
        return 1.0 - float(np.min(dists))

    def best_other(self, member_id: str, q: np.ndarray) -> Tuple[float, Optional[str]]:
        """Closest OTHER member: (similarity = 1 - face_distance, other_id); (-1.0, None) if there is none."""
        E, owners = self._snap
        mask = owners != member_id
        if not mask.any():
            return -1.0, None
        import face_recognition
        dists = face_recognition.face_distance(E[mask], np.asarray(q, dtype=np.float64).flatten())
        j = int(np.argmin(dists))
        return 1.0 - float(dists[j]), str(owners[mask][j])

    # --------------------------------------------------------------- matching
    def _distances(self, q: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """face_recognition.face_distance from q to every (optionally masked) cached encoding."""
        import face_recognition
        E = self._snap[0][mask] if mask is not None else self._snap[0]
        return face_recognition.face_distance(E, np.asarray(q, dtype=np.float64).flatten())

    def identify(self, q: np.ndarray) -> Match:
        """1:N identification: who does this face belong to? No userId involved.

        Groups the cached encodings by userId and keeps the minimum distance per member. The face
        matches a member only if its closest encoding is within TOLERANCE AND beats the runner-up
        member by IDENTIFY_MARGIN; otherwise the result is "Unknown". An empty gallery returns an
        unmatched result instead of raising.
        """
        E, owners = self._snap
        if E.shape[0] == 0:
            return Match(None, None, float("inf"))
        dists = self._distances(q)

        best_per_member: dict = {}
        for owner, d in zip(owners.tolist(), dists.tolist()):
            if d < best_per_member.get(owner, float("inf")):
                best_per_member[owner] = d
        ranked = sorted(best_per_member.items(), key=lambda kv: kv[1])
        best_id, best_d = ranked[0]
        rival_id, rival_d = (ranked[1] if len(ranked) > 1 else (None, float("inf")))

        matched = best_d <= S.tolerance and (rival_d - best_d) >= S.identify_margin
        if not matched:
            log.info("identify: no match (best member=%s dist=%.3f rival=%s dist=%s)",
                     best_id, best_d, rival_id, f"{rival_d:.3f}" if np.isfinite(rival_d) else "inf")
            return Match(None, None, float(best_d), rival_id, float(rival_d))
        match = Match(best_id, self._names.get(best_id), float(best_d), rival_id, float(rival_d))
        log.info("identify: member=%s dist=%.3f conf=%.2f rival=%s margin=%.3f",
                 best_id, match.distance, match.confidence, rival_id, match.margin)
        return match

    def decide(self, member_id: str, q: np.ndarray) -> Decision:
        """Legacy 1:1 verification with a rival check (kept for compatibility; scanning now identifies 1:N).

        Accept only if the claimed member's closest encoding is within TOLERANCE AND beats every
        other enrolled member by RIVAL_MARGIN. Lookalikes/siblings/twins therefore get rejected
        (sent to staff) instead of being silently accepted as someone else.
        """
        E, owners = self._snap
        mine_mask = owners == member_id
        if E.shape[0] == 0 or not mine_mask.any():
            return Decision(False, -1.0, -1.0, 0.0)
        dists = self._distances(q)
        own_d = float(dists[mine_mask].min())
        other_mask = ~mine_mask
        rival, rival_id, margin = -1.0, None, 0.0
        if other_mask.any():
            other = dists[other_mask]
            j = int(np.argmin(other))
            other_d = float(other[j])
            rival, rival_id, margin = 1.0 - other_d, str(owners[other_mask][j]), other_d - own_d
        ok = own_d <= S.tolerance and margin >= S.rival_margin
        return Decision(ok, 1.0 - own_d, rival, margin, rival_id)

    # ------------------------------------------------------------- attendance
    def last_attendance_ts(self, member_id: str) -> Optional[float]:
        with self.lock:
            r = self.db.execute("SELECT MAX(ts) FROM attendance WHERE member_id=?", (member_id,)).fetchone()
        return r[0] if r and r[0] else None

    def mark_attendance(self, member_id: str, score: float, margin: float, session: str) -> str:
        ts = time.time()
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO attendance(member_id, ts, score, margin, session) VALUES(?,?,?,?,?)",
                (member_id, ts, score, margin, session),
            )
        return iso(ts)

    def recent_attendance(self, limit: int = 100) -> List[dict]:
        with self.lock:
            rows = self.db.execute(
                """SELECT a.member_id, m.name, a.ts, a.score, a.margin FROM attendance a
                   LEFT JOIN members m ON m.member_id = a.member_id ORDER BY a.ts DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {"member_id": r[0], "name": r[1], "time": iso(r[2]), "score": round(r[3] or 0, 3), "margin": round(r[4] or 0, 3)}
            for r in rows
        ]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")
