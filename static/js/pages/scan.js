// Scanner page, ported 1:1 from React src/Scanner.jsx (+ CameraStage.jsx).
// Same phases, same messages, same ack-based frame flow: send the next frame only after the
// server answered the previous one, +60ms between frames, retry every 100ms while opening -
// so the scan runs as fast as the server's CPU allows and never builds a queue.
import { api, fmtTime } from "../api.js";
import { createCamera } from "../camera.js";
import { openScanSocket } from "../ws.js";
import { clear, compose, el } from "../ui.js";

const TERMINAL = new Set(["MARKED", "FAILED"]);

export function mountScanner(root) {
  const panel = root.querySelector(".panel");
  const stage = panel.querySelector(".stage");
  const video = panel.querySelector("[data-video]");
  const loading = panel.querySelector("[data-cam-loading]");
  const camError = panel.querySelector("[data-cam-error]");
  const scanControls = panel.querySelector("[data-scan-controls]");
  const startBtn = panel.querySelector("[data-start-scan]");
  const errorEl = panel.querySelector("[data-error]");
  const statusEl = panel.querySelector("[data-status]");
  const statusMessage = panel.querySelector("[data-status-message]");
  const stepsEl = panel.querySelector("[data-steps]");
  const connectingEl = panel.querySelector("[data-connecting]");
  const doneEl = panel.querySelector("[data-done]");
  const metricsEl = panel.querySelector("[data-metrics]");
  const ringBase = panel.querySelector("[data-ring-base]");
  const ringProgress = panel.querySelector("[data-ring-progress]");

  let phase = "idle";          // idle | scanning | done
  let status = null;
  let error = "";
  let finished = false;
  let ws = null;
  let disposed = false;
  const RETRY_DELAY_MS = 1650;  // User-UX: after a terminal result (success or fail) we hide the banner
                              // and start the next scan on its own, so the member doesn't have to
                              // hunt for a "Next member" / "Try again" button. The delay lets the
                              // last result sink in first.

  const cam = createCamera(video, { loading, error: camError });
  camError.remove();

  let hiddenTerminal = false;
  let autoRetryTimer = null;

  function render() {
    const nodes = [stage];
    if (phase !== "scanning" && phase !== "done") {
      if (scanControls) {
        scanControls.style.display = "";
        nodes.push(scanControls);
      }
    } else {
      if (scanControls) scanControls.style.display = "none";
    }
    if (error && !hiddenTerminal) {
      errorEl.textContent = error;
      errorEl.style.display = "";
      nodes.push(errorEl);
    } else {
      errorEl.style.display = "none";
      errorEl.textContent = "";
    }
    if (phase === "scanning" && status && !hiddenTerminal) {
      showStatus(status);
      statusEl.style.display = "";
      nodes.push(statusEl);
    } else {
      statusEl.style.display = "none";
    }
    if (phase === "scanning" && !status && !hiddenTerminal) {
      connectingEl.style.display = "";
      nodes.push(connectingEl);
    } else {
      connectingEl.style.display = "none";
    }
    if (phase === "done" && status && !hiddenTerminal) {
      showDone(status);
      doneEl.style.display = "";
      nodes.push(doneEl);
    } else {
      doneEl.style.display = "none";
    }
    if (status && status.metrics && !hiddenTerminal) {
      metricsEl.textContent = JSON.stringify(status.metrics);
      metricsEl.style.display = "";
      nodes.push(metricsEl);
    } else {
      metricsEl.style.display = "none";
    }
    compose(panel, nodes);
    setTone();
  }

  function showStatus(s) {
    statusEl.className = `msg ${s.ok ? "good" : "warn"}`;
    statusMessage.textContent = s.message ?? "";
    const ch = s.challenge;
    const showSteps = s.state === "CHALLENGE" && ch;
    if (showSteps) {
      clear(stepsEl);
      for (let i = 0; i < ch.total; i++) {
        stepsEl.appendChild(el("span", i < ch.step - 1 ? "dot done" : i === ch.step - 1 ? "dot cur" : "dot"));
      }
      stepsEl.appendChild(el("span", "hint", `Step ${ch.step} of ${ch.total}`));
      if (stepsEl.parentElement !== statusEl) statusEl.appendChild(stepsEl);
    } else if (stepsEl.parentElement === statusEl) {
      stepsEl.remove();
    }
  }

  function showDone(s) {
    doneEl.className = `msg big ${s.state === "MARKED" ? "ok" : "bad"}`;
    clear(doneEl);
    if (s.state === "MARKED") {
      doneEl.appendChild(el("strong", "", s.member ? `Welcome, ${s.member.name}!` : s.message));
      if (s.time) doneEl.appendChild(el("div", "", `Attendance marked at ${fmtTime(s.time)}`));
    } else {
      doneEl.appendChild(el("strong", "", s.message));
    }
  }

  function setTone() {
    const tone = status?.state === "MARKED" ? "ok" : status?.state === "FAILED" ? "bad" : status ? (status.ok ? "good" : "warn") : "idle";
    const progress = status?.progress ?? 0;
    ringBase.setAttribute("class", `ring ring-base tone-${tone}`);
    ringProgress.setAttribute("class", `ring ring-progress tone-${tone}`);
    ringProgress.setAttribute("stroke-dasharray", `${Math.round(progress * 100)} 100`);
  }

  async function start(e) {
    e?.preventDefault();
    error = "";
    status = null;
    render();
    if (!cam.isReady()) {
      error = "Camera is not ready yet.";
      return render();
    }
    try {
      const s = await api("/api/scan/start", { method: "POST", body: {} });
      if (disposed) return;
      if (s.already_marked) {
        status = { state: "MARKED", message: `Hi ${s.name}, you're already checked in (${fmtTime(s.at)}).` };
        phase = "done";
        return render();
      }
      openSocket(s.token);
    } catch (err) {
      if (disposed) return;
      error = err.message;
      render();
    }
  }

  function handleClose() {
    if (disposed || finished) return;
    error = "Connection lost. Please try again.";
    phase = "idle";
    render();
  }

  function openSocket(token) {
    finished = false;
    phase = "scanning";
    render();                       // show "Connecting…" first, then dial (never the other way round)
    ws = openScanSocket(token, { onOpen: sendNext, onMessage: handleMessage, onClose: handleClose });
  }

  async function sendNext() {
    if (disposed || !ws || ws.readyState !== WebSocket.OPEN) return;
    const blob = await cam.capture();
    if (disposed) return;
    if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
    else setTimeout(sendNext, 100);
  }

  function handleMessage(ev) {
    if (disposed) return;
    const m = JSON.parse(ev.data);
    status = m;
    if (TERMINAL.has(m.state)) {
      finished = true;
      phase = "done";
      hideTerminalAfterwards(m.state, error);
    } else {
      render();
      setTimeout(sendNext, 60);
    }
  }

  function hideTerminalAfterwards(resultState, resultError) {
    // Show the result banner first; give the operator / member a moment to see it, then
    // hide it and start the next scan automatically.
    render();
    if (!autoRetryTimer) {
      autoRetryTimer = setTimeout(() => {
        autoRetryTimer = null;
        if (disposed) return;
        hiddenTerminal = false;
        error = "";
        status = null;
        phase = "idle";
        render();
        start();                 // restart the scan (with the new token) on its own
      }, RETRY_DELAY_MS);
    }
    scheduleToneReset();
  }

  // Ring color: show the terminal color while the banner is visible, then snap back to idle
  // when it auto-hides.
  let toneResetTimer = null;
  function scheduleToneReset() {
    if (toneResetTimer) clearTimeout(toneResetTimer);
    toneResetTimer = setTimeout(() => {
      toneResetTimer = null;
      if (disposed) return;
      setTone();
    }, RETRY_DELAY_MS + 300);
  }

  setTone();                   // initial idle tone
  cam.start();                  // open the webcam (same as the Enroll page does)

  if (startBtn) startBtn.addEventListener("click", start);
  render();

  return function unmount() {
    disposed = true;
    if (autoRetryTimer) { clearTimeout(autoRetryTimer); autoRetryTimer = null; }
    if (toneResetTimer) { clearTimeout(toneResetTimer); toneResetTimer = null; }
    try { ws?.close(); } catch {}
    cam.stop();
  };
}
