# Gym face-attendance system

Real-time face check-in. One FastAPI app serves **everything** — the UI (plain HTML/CSS/vanilla
JS, no Node, no npm, no build step), the REST API and the scan WebSocket — from a single origin.
FastAPI + `face_recognition` (dlib) backend, CPU-only. Designed for ~1,000 members with **1:N
identification**: at scan time the person provides NO ID — the system recognises who they are from
their face alone (`face_recognition.face_distance` against every stored 128-d encoding,
`TOLERANCE = 0.5`).

## How a check-in works

```
POST /api/scan/start (member ID optional; the face alone decides who is checked in)
     ─► server creates a session with a RANDOM head-turn sequence
Browser streams JPEG frames over WebSocket (next frame sent only after the previous reply)

POSITIONING  4 consecutive good frames: one face, centred, right size, frontal, sharp, well lit,
             not moving, nothing in front of it. The face is identified 1:N; the identified member
             is adopted by the session.
CHALLENGE    OFF by default (fast, still-face check-in). Set CHALLENGE_STEPS=2 to require random
             server-checked head turns — a liveness check that flat photos/screens cannot pass.
VERIFYING    4 more good frames. Must identify the SAME member as before (the person who started
             the scan is the person who finished it).
MARKED       attendance row written for the identified member, cooldown applies (default 4 h).
             Confidence is returned; raw distances are never sent to the browser.
```

Accuracy measures built in:

| Risk | Defence |
|---|---|
| Lookalikes / siblings | Best member must beat the runner-up by `IDENTIFY_MARGIN`; otherwise the face is returned as "Unknown" |
| Photo or phone screen shown | OFF by default for speed — re-enable with `CHALLENGE_STEPS=2` (random head turns a flat photo cannot fake) and/or `OBJECT_DETECTION=1` |
| Recorded video replay | Challenge direction is random per session; session token is single-use |
| Someone swaps in mid-scan | Identity must match both before and after the challenge |
| Blurry / dark / off-angle frames | Hard quality gates with a clear message for each ("Move closer", "Hold still", ...) |
| Bad enrolment (the #1 cause of errors) | Staff-assisted, 7 guided poses, per-photo quality gate, consistency check, duplicate-ID check |
| Guessing attacks | Lockout after repeated failures, per-IP rate limit, generic failure message |

---

# Using the app, step by step

## Step 1 — Install the backend

Python 3.10 or 3.11 recommended (3.12 works; see note). One-time setup:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- The pip package names matter: `face_recognition` (underscores, one word) and
  `opencv-python-headless` (never plain `opencv`). They are already correct in `requirements.txt`.
- `dlib` compiles from source on install: install `cmake` and a C++ toolchain first
  (`apt install cmake build-essential` on Debian/Ubuntu, Xcode tools on macOS,
  Visual Studio Build Tools + CMake on Windows).
- On Python >= 3.12 keep `setuptools<81` installed (pinned in `requirements.txt`) —
  `face_recognition_models` still needs `pkg_resources`.
- No model downloads at first start: the dlib shape predictor ships inside the
  `face_recognition_models` package.

## Step 2 — Configure and start

1. Create your config file:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set at least `ADMIN_KEY` to a long random string (it protects every
   admin/enrolment endpoint). `uvicorn` loads `.env` automatically — no extra export needed.

   Also make sure `CORS_ORIGINS` contains **every origin the UI is served from** — the
   `/ws/scan/{token}` endpoint checks the browser's `Origin` header against that list and closes
   the socket with 4403 otherwise. The default already covers `http://localhost:8000` and
   `http://127.0.0.1:8000`; add your LAN or HTTPS origin for a real kiosk.

3. Start the server from `backend/` (keep the virtualenv activated):

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   Keep **ONE worker** — sessions and the gallery cache live in process memory.

4. Verify it is up:

   ```bash
   curl http://localhost:8000/api/health      # -> {"ok":true,"model":"buffalo_l"}
   ```

## Step 3 — Open the app

Open **http://localhost:8000** in the browser. That's it: the same server renders the pages, so
there is no Node.js, npm or build step anywhere.

Browsers only allow the webcam on `https://` or `localhost` — `http://localhost:8000` works for
development, but a real gym kiosk must be served over HTTPS. Put the app behind a TLS-terminating
reverse proxy, e.g. with Caddy (it auto-proxies WebSockets too):

```
# Caddyfile — kiosk.gym.example gets a free Let's Encrypt certificate
kiosk.gym.example {
    reverse_proxy 127.0.0.1:8000
}
```

or nginx:

```nginx
server {
    listen 443 ssl;
    server_name kiosk.gym.example;
    ssl_certificate     /etc/letsencrypt/live/kiosk.gym.example/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/kiosk.gym.example/privkey.pem;

    location / { proxy_pass http://127.0.0.1:8000; proxy_http_version 1.1;
                 proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";
                 proxy_set_header Host $host; }
}
```

Then add `https://kiosk.gym.example` to `CORS_ORIGINS` and restart. (If you serve HTTPS on
another port/host, keep uvicorn itself on plain HTTP behind the proxy.)

## Step 4 — Enrol members (staff task)

Enrolment happens once per member; after it they can check in with their face alone.

1. Open the **Enrol member (staff)** tab.
2. Enter the **staff key** (the `ADMIN_KEY` from `.env`), a **Member ID** (letters, digits, `-`, `_`,
   max 32 chars) and the member's **Full name**.
3. Click **Start enrolment**. The camera view appears with a pose prompt.
4. Click **Capture** for each of the 7 guided poses ("Look straight", "Turn slightly left/right",
   chin up/down, relaxed, slight smile). Each photo is checked for face, pose, blur and lighting;
   a rejected photo shows the exact reason — just capture it again.
5. The counter shows how many photos were accepted. The minimum is 5 (`MIN_ENROLL_SAMPLES`);
   more slightly different poses = more robust recognition.
6. If a photo "closely matches another member already enrolled", the system refuses it (possible
   duplicate identity / buddy punching) — ask a supervisor.
7. Click **Finish** when the counter says ready. Wrong person enrolled? Click **Start over** to
   delete this member's photos and redo step 4.

## Step 5 — Check in (member task)

1. Open the **Check in** tab. No login, no Member ID needed — the face is the ID.
2. Click **Start scan** and look at the camera. With the defaults a still face is enough and the
   whole scan takes a few seconds:
   - positioning: hold still until the progress dots fill (you'll be told to move closer, centre
     your face, improve lighting, etc. if needed);
   - keep looking straight until the second set of dots fills — done, the welcome screen appears.
   (If the operator enabled `CHALLENGE_STEPS=2`, there is a head-turn step in between: slowly turn
   your head the way the screen asks.)
3. On success you'll see **"Welcome, {name}!"** with the time and a confidence percentage.
4. On "Unknown" or a failed liveness check the scan ends with a generic message; the person can
   retry or go to the front desk. Repeated failures temporarily lock the member, and there is a
   per-IP rate limit.
5. A member cannot check in twice within the cooldown window (default 4 h) — the scanner says
   they are already checked in.

Tip: typing a Member ID before **Start scan** is still allowed for kiosk-style setups, but it is
only used for start-time checks (name greeting, cooldown, lockout) — recognition itself always
works 1:N from the face.

## Step 6 — Admin: inspect members and attendance

All of these need the `X-Admin-Key: <ADMIN_KEY>` header (or use curl):

```bash
curl -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8000/api/admin/members          # list + sample counts
curl -H "X-Admin-Key: $ADMIN_KEY" "http://localhost:8000/api/admin/attendance?limit=50"
curl -X DELETE -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8000/api/admin/members/m123
curl -X POST    -H "X-Admin-Key: $ADMIN_KEY" http://localhost:8000/api/admin/members/m123/reset
```

- `DELETE /api/admin/members/{id}` removes the member AND their encodings (GDPR/DPDP erasure).
- `POST .../reset` keeps the member but wipes their face encodings so they can be re-enrolled.
- Every attendance row stores score + margin for audit; review low-margin rows periodically.

## Step 7 — Tune on your camera (first day, once)

1. Set `DEBUG_METRICS=1` in `.env` and restart the backend. Run a few check-ins; the scanner UI
   prints raw yaw/pitch/blur/brightness values per frame.
2. Adjust the gates in `.env` to match **your** camera and lighting: `PITCH_MIN`/`PITCH_MAX`,
   `MIN_SHARPNESS`, `MIN_BRIGHTNESS`/`MAX_BRIGHTNESS`, `MIN_CONTRAST`. The defaults are educated
   guesses.
3. Collect real photos of members (5–10 per person, including lookalikes/siblings — they are the
   cases that matter), then calibrate the match threshold:

   ```bash
   python -m scripts.calibrate dataset/       # dataset/<member_id>/*.jpg
   ```

   Set `TOLERANCE` from the printed FAR/FRR table (subtract a small safety buffer).
4. Set `DEBUG_METRICS=0` again for production.

## Step 8 — Verify your install (no server needed)

```bash
cd backend
python -m tests.test_logic                        # model-free: pose geometry + session state machine
python -m scripts.test_identify dataset/          # real models: enrol -> identify -> unknown -> no-face
```

`test_identify` uses a throw-away database, never your real `data/gallery.db`. Its dataset layout:
folders of enrolled members (folder name = member id), plus optional `_unknown/` (faces never
enrolled, must return "Unknown") and `_noface/` (blank images, must return "no face detected").

---

## The web UI: how it is served and how to add a page

The React app was ported 1:1 to plain HTML + CSS + vanilla-JS ES modules (no framework, no build).
The old React sources are kept untouched in `legacy_frontend/` for comparison and rollback; they
are not used at runtime and can be excluded from any deployment.

```
backend/templates/base.html         shared shell (head, CSS link, page script)
backend/templates/index.html        the app page: header + two tab panels (same URLs as before)
backend/templates/pages/scan.html   Check-in tab markup (inert <template>, cloned in by JS)
backend/templates/pages/enroll.html Enrol tab markup
backend/static/css/app.css          the CSS from the React app, verbatim
backend/static/js/api.js            fetch wrapper (X-Admin-Key, JSON errors) - ported from React
backend/static/js/camera.js         getUserMedia + JPEG capture - ported from useCamera.js
backend/static/js/ws.js             WebSocket client (same endpoint/protocol) - same as Scanner.jsx
backend/static/js/ui.js             DOM helpers (textContent-only, React-like compose())
backend/static/js/app.js            tab host - ported from App.jsx (remount on tab switch)
backend/static/js/pages/scan.js     Check-in page - ported from Scanner.jsx
backend/static/js/pages/enroll.js   Enrol page - ported from Enroll.jsx
backend/routers/pages.py            static mount, security headers, GZip, page routes
```

To add a page:

1. Write `backend/templates/mypage.html` extending `base.html`.
2. Add a route in `backend/routers/pages.py` returning
   `templates.TemplateResponse(request, "mypage.html")`. Register it **before** `install()` is
   called from `app/main.py` (the last route is a catch-all SPA fallback that returns
   `index.html` for any unknown path — the same behaviour Vite had).
3. Put its behaviour in `backend/static/js/pages/mypage.js` and load it from the template.
   Asset URLs automatically carry `?v=<content-hash>` (`{{ build }}`) so browsers pick up edits.

Security, kept or added: pages are rendered with Jinja autoescape; all dynamic text in the page
JS is written with `textContent` (never `innerHTML`); the admin key lives in `sessionStorage`,
exactly like the React app (see the note below); responses carry `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: same-origin`, `Permissions-Policy: camera=(self)` and a
self-only `Content-Security-Policy`; static files get a 1-day `Cache-Control`; pages are gzipped
(`GZipMiddleware`), while WebSocket frames are never compressed.

---

## What is verified, and what is not

Verified: pose geometry and the whole session state machine (happy path, wrong-direction turns,
photo that never turns, identity mismatch, mid-scan swap, broken stability window) pass
`tests/test_logic.py`; the real-model pipeline (`scripts/test_identify.py`) correctly identifies
enrolled faces, returns "Unknown" for un-enrolled ones and a clean result for faceless images; all
Python compiles; the page JS parses cleanly; the app serves `/`, the static assets, the security
headers and the SPA fallback, and the scan WebSocket accepts connections from the serving origin.

**Not verified here**: no real webcam was involved. Expect to tune the quality thresholds on your
hardware. The recognition threshold is uncalibrated until you run Step 7.

## Known limits (be honest with yourself about these)

- **Identical twins** cannot be separated by any face system. With the rival margin they come back
  "Unknown" rather than mis-accepted, so route them to staff verification.
- **A printed photo or a phone screen of an enrolled member will be accepted** while the head-turn
  challenge is disabled (the default). If that risk matters in your gym, set `CHALLENGE_STEPS=2`
  (a few seconds slower, stops photos/screens/replays) and/or `OBJECT_DETECTION=1`. Neither stops a
  3D mask or a deepfake injected through a virtual camera; for that, add a passive anti-spoofing
  model (e.g. MiniFASNet / Silent-Face-Anti-Spoofing via ONNX). For a browser client, a staffed or
  locked-down kiosk device is the practical defence against virtual cameras.
- Object detection (`OBJECT_DETECTION=1`) uses a generic COCO model: coarse, ~60-100 ms per frame on
  CPU. Occlusion by masks/sunglasses is mainly caught by the fact that the match fails.
- Rough CPU expectation: a few frames per second per scan. Measure on your machine; lower
  `DETECT_SCALE` further or raise `MIN_FRAME_MATCH_RATIO` if it feels slow.
  `MAX_CONCURRENT_INFERENCE` caps parallel work.
- Single process by design (sessions and the gallery cache are in memory). For multiple workers,
  move sessions to Redis.
- Old 512-d InsightFace embeddings are NOT migrated: they are skipped at startup with a warning —
  re-enrol members after upgrading.

## Privacy and security (India: DPDP Act 2023)

- Only encodings are stored, never photos. Still, they are biometric data: get explicit written
  consent, state the purpose (attendance only), set a retention period, and provide a way to delete
  a member (`DELETE /api/admin/members/{id}`). Encrypt the disk holding `data/gallery.db`.
- Set a strong `ADMIN_KEY`; serve over HTTPS/WSS; keep `CORS_ORIGINS` minimal; admin endpoints
  should not be exposed to the public internet (VPN or IP allow-list).
- The staff key is kept in `sessionStorage` in the browser (same as the React app did) and sent as
  the `X-Admin-Key` header. Optional follow-up (needs your go-ahead): switch to an `HttpOnly`,
  `Secure`, `SameSite` cookie so JavaScript can never read the key.
- Every attendance row stores the score and margin for audit. Review rows with a low margin
  periodically.

## Layout

```
backend/app/config.py     every threshold (env-overridable), incl. TOLERANCE / IDENTIFY_MARGIN
backend/app/quality.py    pose geometry + quality gates (pure, tested)
backend/app/session.py    scan state machine + liveness challenge (tested)
backend/app/pipeline.py   face_recognition (dlib HOG) detection + 128-d encodings, optional YOLO
backend/app/store.py      SQLite + in-memory 1:N matcher (face_distance, TOLERANCE, rival margin)
backend/app/main.py       REST + WebSocket API, abuse guard; installs the page router last
backend/routers/pages.py  static mount, security headers, GZip, page routes (SPA fallback)
backend/templates/        base.html, index.html, pages/scan.html, pages/enroll.html
backend/static/           css/ + js/ (api, camera, ws, ui, app + pages/)
backend/scripts/calibrate.py        FAR/FRR threshold calibration on your data
backend/scripts/test_identify.py    enrol/identify/unknown/no-face end-to-end check
legacy_frontend/          the old React app (comparison/rollback only; no Node needed to run)
```
