"""Serves the web UI (HTML pages + static assets) from the same FastAPI app, same origin.

The React app that used to live in ../frontend is the source of truth; it was ported 1:1 to
plain HTML/CSS/vanilla-JS ES modules with no build step. This module only *serves* pages and
assets: it adds no API behaviour and changes no existing route, WebSocket or recognition logic.

Adding a page:
 1. add a template under templates/ extending base.html (put reusable chunks in templates/pages/)
 2. add a route here that returns templates.TemplateResponse(request, "my-page.html")
 3. put the page's behaviour in static/js/pages/ and load it from the template
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

ROOT = Path(__file__).resolve().parent.parent      # backend/
STATIC_DIR = ROOT / "static"
TEMPLATES_DIR = ROOT / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR), autoescape=True)   # autoescape: dynamic text is always escaped


def _build_id() -> str:
    """Content hash of the static tree -> every asset URL carries ?v=<build>, so an update is
    picked up immediately even with a long cache lifetime."""
    h = hashlib.sha256()
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


BUILD_ID = _build_id()
templates.env.globals["build"] = BUILD_ID


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Same-origin hosting must not weaken anything. Applies to every HTTP response
    (WebSocket upgrades bypass BaseHTTPMiddleware, so frame traffic is untouched)."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            # Content-hash query (?v=) busts the cache on every deploy, so a day of caching is safe.
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(self)")
        # Everything self-hosted; ws:/wss: only needed because some browsers do not treat
        # 'self' as covering an upgraded WebSocket to the same host.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self'; font-src 'self'; "
            "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'self'; "
            "form-action 'self'; frame-ancestors 'none'",
        )
        return response


router = APIRouter()


def _index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@router.get("/", include_in_schema=False)
def index(request: Request):
    return _index(request)


@router.get("/{path:path}", include_in_schema=False)
def spa_fallback(request: Request, path: str):
    """The React app had no client-side router: Vite served index.html for unknown paths, so
    the browser always showed the app at its default tab. Reproduce that here. API/WS paths
    keep their existing 404 behaviour."""
    if path.startswith(("api/", "ws/")):
        raise HTTPException(404, "Not Found")
    return _index(request)


def install(app) -> None:
    """Mount static files and register the page routes + page-serving middleware.

    MUST be called after every API route is registered: Starlette matches routes in
    registration order, and the SPA fallback below is a catch-all.
    """
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )
    # GZip only ever sees HTTP responses; WebSocket frames are sent on the raw protocol
    # after the 101 upgrade, so scan traffic is never compressed or delayed.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(router)
