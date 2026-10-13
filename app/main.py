"""FastAPI app: admin enrolment API, scan-start API, and the real-time scan WebSocket.

Run a SINGLE worker (sessions and the gallery snapshot live in process memory):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import (Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings as S
from .pipeline import FaceEngine, ObjectDetector, load_upload_image
from .quality import quality_problem
from .session import TERMINAL, ScanSession
from .store import Gallery, iso
from routers import pages                 # HTML UI + static assets, served from this same app
from dotenv import load_dotenv


load_dotenv()

log = logging.getLogger("attendance")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")

ctx = SimpleNamespace()                 # engine, gallery, objects, sem, guard
sessions: dict[str, ScanSession] = {}


class Guard:
    """Tiny in-memory abuse protection: lock a member after repeated failed verifications, limit scan starts per IP."""

    def __init__(self) -> None:
        self.fails: dict[str, deque] = defaultdict(deque)
        self.starts: dict[str, deque] = defaultdict(deque)

    @staticmethod
    def _prune(q: deque, window: float) -> None:
        now = time.time()
        while q and now - q[0] > window:
            q.popleft()

    def locked(self, member_id: str) -> bool:
        q = self.fails[member_id]
        self._prune(q, S.failure_window_s)
        return len(q) >= S.max_failures

    def record_failure(self, member_id: str) -> None:
        self.fails[member_id].append(time.time())

    def start_allowed(self, ip: str) -> bool:
        q = self.starts[ip]
        self._prune(q, 60)
        if len(q) >= S.starts_per_min:
            return False
        q.append(time.time())
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(os.getenv("ADMIN_KEY"))
    if S.admin_key == "change-me":
        log.warning("ADMIN_KEY is still the default. Set a strong value before exposing this server.")
    ctx.engine = FaceEngine()
    ctx.engine.analyze(np.zeros((480, 640, 3), np.uint8))          # warm-up (first ONNX call is slow)
    ctx.gallery = Gallery(os.path.join(S.data_dir, "gallery.db"))
    ctx.objects = ObjectDetector() if S.object_detection else None
    ctx.sem = asyncio.Semaphore(S.max_concurrent_inference)         # bound CPU use
    ctx.guard = Guard()
    log.info("ready: model=%s members=%d", S.model_pack, len(ctx.gallery.list_members()))
    yield


app = FastAPI(title="Gym face attendance", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=list(S.cors_origins), allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------- helpers
def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    if not x_admin_key or not secrets.compare_digest(x_admin_key, S.admin_key):
        raise HTTPException(401, "Invalid admin key")


def check_id(member_id: str) -> str:
    if not ID_RE.match(member_id):
        raise HTTPException(400, "Member ID may contain letters, digits, - and _ (max 32 chars)")
    return member_id


def decode_image(data: bytes, max_bytes: int = 2_000_000):
    if not data or len(data) > max_bytes:
        return None
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > 960:
        img = cv2.resize(img, (960, int(h * 960 / w)))
    return img


def prune_sessions() -> None:
    for t in [t for t, s in sessions.items() if s.expired or s.state in TERMINAL]:
        sessions.pop(t, None)


@app.get("/api/health")
def health():
    return {"ok": True, "model": S.model_pack}


# ----------------------------------------------------------------------- admin
class MemberReq(BaseModel):
    member_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=80)


@app.post("/api/admin/members", dependencies=[Depends(require_admin)])
def create_member(req: MemberReq):
    check_id(req.member_id)
    created = ctx.gallery.add_member(req.member_id, req.name.strip())
    return {"created": created, "samples": ctx.gallery.count(req.member_id), "needed": S.min_enroll_samples}


@app.get("/api/admin/members", dependencies=[Depends(require_admin)])
def list_members():
    return ctx.gallery.list_members()


@app.delete("/api/admin/members/{member_id}", dependencies=[Depends(require_admin)])
def delete_member(member_id: str):
    ctx.gallery.delete_member(check_id(member_id))
    return {"deleted": True}


@app.post("/api/admin/members/{member_id}/reset", dependencies=[Depends(require_admin)])
def reset_member(member_id: str):
    ctx.gallery.reset_embeddings(check_id(member_id))
    return {"samples": 0}


@app.get("/api/admin/attendance", dependencies=[Depends(require_admin)])
def attendance(limit: int = 100):
    return ctx.gallery.recent_attendance(min(max(limit, 1), 1000))


@app.post("/api/admin/members/{member_id}/enroll", dependencies=[Depends(require_admin)])
async def enroll(member_id: str, image: UploadFile = File(...)):
    """Add ONE enrolment photo. Staff calls this ~7 times with slightly different poses."""
    check_id(member_id)
    if not ctx.gallery.get_member(member_id):
        raise HTTPException(404, "Unknown member. Create the member first.")
    img = load_upload_image(await image.read())          # RGB via face_recognition.load_image_file
    if img is None:
        raise HTTPException(400, "Invalid image")

    def reply(accepted: bool, reason: str | None = None, **extra):
        n = ctx.gallery.count(member_id)
        return {"accepted": accepted, "reason": reason, "samples": n, "ready": n >= S.min_enroll_samples, **extra}

    async with ctx.sem:
        a = await asyncio.to_thread(ctx.engine.analyze, img)
        problem = quality_problem(a, None, strict=False)
        if problem:
            return reply(False, problem)
        emb = await asyncio.to_thread(ctx.engine.embed, img, a, S.enroll_jitters)

    # safety checks in distance space: TOLERANCE separates same-person from different-person pairs
    own = ctx.gallery.own_similarity(member_id, emb)
    if own is not None and own < 1.0 - S.tolerance:
        return reply(False, "This face doesn't look like the earlier photos of this member. "
                            "Use 'Start over' if the wrong person was enrolled.")
    sim, other_id = ctx.gallery.best_other(member_id, emb)
    if sim >= 1.0 - S.tolerance:
        # Same person registered under two IDs (buddy punching), or a lookalike the system cannot separate.
        log.info("enroll rejected: member=%s matches other member=%s (dist=%.3f)",
                 member_id, other_id, 1.0 - sim)
        return reply(False, "This face closely matches another member already enrolled. Ask a supervisor.",
                     duplicate_of=other_id)
    ctx.gallery.add_embedding(member_id, emb)
    log.info("enrolled member=%s sample=%d", member_id, ctx.gallery.count(member_id))
    return reply(True)


# ------------------------------------------------------------------------ scan
class StartReq(BaseModel):
    # optional on purpose: scanning identifies the person 1:N from their face alone. When an ID IS
    # supplied it is only used for the start-time checks below (backward compatibility), never for
    # recognition.
    member_id: str | None = Field(default=None, max_length=32)


@app.post("/api/scan/start")
def scan_start(req: StartReq, request: Request):
    ip = request.client.host if request.client else "?"
    if not ctx.guard.start_allowed(ip):
        raise HTTPException(429, "Too many attempts. Please wait a minute.")
    member = None
    member_id = req.member_id.strip() if req.member_id else ""
    if member_id:
        check_id(member_id)
        member = ctx.gallery.get_member(member_id)
        if not member or ctx.gallery.count(member_id) < S.min_enroll_samples:
            raise HTTPException(404, "Member not found or face not enrolled. Please see the front desk.")
        if ctx.guard.locked(member_id):
            raise HTTPException(429, "Too many failed attempts. Please see the front desk.")

        last = ctx.gallery.last_attendance_ts(member_id)
        if last and time.time() - last < S.cooldown_min * 60:
            return {"already_marked": True, "at": iso(last), "name": member["name"]}
    elif ctx.gallery.enrolled_count(S.min_enroll_samples) == 0:
        raise HTTPException(404, "No registered users yet. Please see the front desk.")

    prune_sessions()
    steps = [secrets.choice(["left", "right"]) for _ in range(S.challenge_steps)]   # unpredictable => replay-resistant
    sess = ScanSession(member_id, member["name"] if member else "", steps)
    sessions[sess.token] = sess
    return {"token": sess.token, "expires_in": S.session_timeout_s, "name": member["name"] if member else ""}


@app.websocket("/ws/scan/{token}")
async def ws_scan(ws: WebSocket, token: str):
    origin = ws.headers.get("origin")

    print("Origin:", repr(origin))
    print("Allowed origins:", repr(S.cors_origins))
    print("Token exists:", token in sessions)

    # 1. Validate Origin before accepting the connection.
    if origin and origin not in S.cors_origins:
        print(f"WebSocket rejected: Origin not allowed: {origin}")
        await ws.close(code=1008)
        return

    # 2. Validate the session token.
    sess = sessions.get(token)
    if sess:
        print("Session expired:", sess.expired)
        print("Session already attached:", sess.attached)

    if sess is None or sess.expired or sess.attached:
        print(
            "WebSocket rejected: invalid session, expired session, "
            "or session already attached"
        )
        await ws.close(code=1008)
        return

    # 3. Accept exactly once, after validation.
    sess.attached = True
    await ws.accept()

    try:
        while sess.state not in TERMINAL:
            data = await ws.receive_bytes()

            frame = decode_image(data, max_bytes=400_000)

            if frame is None:
                await ws.send_json({
                    "state": sess.state.value,
                    "ok": False,
                    "message": "Bad frame.",
                    "progress": 0,
                })
                continue

            async with ctx.sem:
                out = await asyncio.to_thread(sess.process, frame, ctx)

            await ws.send_json(out)

        if sess.verification_failed:
            ctx.guard.record_failure(sess.member_id)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session")

    except Exception as exc:
        print(f"WebSocket processing error: {exc}")
        raise

    finally:
        sessions.pop(token, None)

        try:
            await ws.close()
        except Exception:
            pass

# -------------------------------------------------------------------- frontend
# Same-origin HTML UI (ported from React, see routers/pages.py). Registered LAST: the page
# router ends in a catch-all SPA fallback and Starlette matches routes in registration order.
pages.install(app)
