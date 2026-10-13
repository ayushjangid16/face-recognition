// Enrol page, ported 1:1 from React src/Enroll.jsx (+ CameraStage.jsx).
// Varied poses make the stored embeddings robust to real-life angles. Staff clicks Capture for each.
import { api } from "../api.js";
import { createCamera } from "../camera.js";
import { clear, compose, el } from "../ui.js";

const PROMPTS = [
  "Look straight at the camera",
  "Turn slightly to the left",
  "Turn slightly to the right",
  "Raise your chin slightly",
  "Lower your chin slightly",
  "Look straight, relaxed expression",
  "Look straight, slight smile",
];

export function mountEnroller(root) {
  const panel = root.querySelector(".panel");
  const stage = panel.querySelector(".stage");
  const video = panel.querySelector("[data-video]");
  const loading = panel.querySelector("[data-cam-loading]");
  const camError = panel.querySelector("[data-cam-error]");
  const form = panel.querySelector("[data-form]");
  const adminKeyInput = form.querySelector("[data-admin-key]");
  const memberIdInput = form.querySelector("[data-member-id]");
  const nameInput = form.querySelector("[data-name]");
  const beginBtn = form.querySelector("[data-begin]");
  const captureCol = panel.querySelector("[data-capture-col]");
  const progressMsg = captureCol.querySelector("[data-progress-msg]");
  const promptEl = captureCol.querySelector("[data-prompt]");
  const captureBtn = captureCol.querySelector("[data-capture]");
  const startOverBtn = captureCol.querySelector("[data-start-over]");
  const finishBtn = captureCol.querySelector("[data-finish]");
  const noteEl = panel.querySelector("[data-note]");
  const ringBase = panel.querySelector("[data-ring-base]");
  const ringProgress = panel.querySelector("[data-ring-progress]");

  let started = false;
  let samples = 0;
  let needed = 5;
  let ready = false;
  let step = 0;
  let busy = false;
  let note = { kind: "", text: "" };

  adminKeyInput.value = "ADMIN"

  const cam = createCamera(video, { loading, error: camError });
  camError.remove();            // React renders the error overlay only when there is an error
  cam.start();

  function render() {
    const nodes = [stage];
    if (!started) {
      updateBeginDisabled();          // inputs were cleared by finish(): recompute the gate
      nodes.push(form);
    } else {
      clear(progressMsg);
      progressMsg.appendChild(el("strong", "", nameInput.value || memberIdInput.value));
      progressMsg.appendChild(document.createTextNode(`: ${samples} photo(s) captured (minimum ${needed}).`));
      promptEl.textContent = PROMPTS[step];
      captureBtn.textContent = busy ? "Checking…" : "Capture";
      captureBtn.disabled = busy;
      finishBtn.disabled = !ready;
      nodes.push(captureCol);
    }
    if (note.text) {
      noteEl.className = `msg ${note.kind}`;
      noteEl.textContent = note.text;
      nodes.push(noteEl);
    }
    compose(panel, nodes);
    setTone();
  }

  function setTone() {
    const tone = note.kind === "bad" ? "warn" : note.kind === "good" ? "good" : "idle";
    ringBase.setAttribute("class", `ring ring-base tone-${tone}`);
    ringProgress.setAttribute("class", `ring ring-progress tone-${tone}`);
    ringProgress.setAttribute("stroke-dasharray", "0 100");   // the enrol page never shows progress
  }

  function updateBeginDisabled() {
    beginBtn.disabled = !adminKeyInput.value || !memberIdInput.value.trim() || !nameInput.value.trim();
  }

  async function begin(e) {
    e.preventDefault();
    note = { kind: "", text: "" };
    try {
      sessionStorage.setItem("adminKey", adminKeyInput.value);
      const r = await api("/api/admin/members", {
        method: "POST",
        body: { member_id: memberIdInput.value.trim(), name: nameInput.value },
        adminKey: adminKeyInput.value,
      });
      samples = r.samples;
      needed = r.needed;
      ready = r.samples >= r.needed;
      started = true;
      if (!r.created) note = { kind: "warn", text: `Member already exists with ${r.samples} sample(s). Adding more.` };
    } catch (err) {
      note = { kind: "bad", text: err.message };
    }
    render();
  }

  async function capture() {
    if (!cam.isReady()) return;
    busy = true;
    note = { kind: "", text: "" };
    render();
    try {
      const blob = await cam.capture(0.92);
      const upload = new FormData();
      upload.append("image", blob, "face.jpg");
      const r = await api(`/api/admin/members/${memberIdInput.value.trim()}/enroll`, {
        method: "POST",
        form: upload,
        adminKey: adminKeyInput.value,
      });
      samples = r.samples;
      ready = r.ready;
      if (r.accepted) {
        step = Math.min(step + 1, PROMPTS.length - 1);
        note = { kind: "good", text: "Captured." };
      } else {
        note = { kind: "bad", text: r.reason };
      }
    } catch (err) {
      note = { kind: "bad", text: err.message };
    } finally {
      busy = false;
      render();
    }
  }

  async function startOver() {
    if (!confirm("Delete all captured photos for this member and start again?")) return;
    try {
      await api(`/api/admin/members/${memberIdInput.value.trim()}/reset`, {
        method: "POST",
        adminKey: adminKeyInput.value,
      });
      samples = 0;
      ready = false;
      step = 0;
      note = { kind: "", text: "" };
    } catch (err) {
      note = { kind: "bad", text: err.message };
    }
    render();
  }

  function finish() {
    started = false;
    memberIdInput.value = "";
    nameInput.value = "";
    samples = 0;
    ready = false;
    step = 0;
    note = { kind: "good", text: "Member enrolled. They can now check in." };
    render();
  }

  for (const input of [adminKeyInput, memberIdInput, nameInput]) input.addEventListener("input", updateBeginDisabled);

  form.addEventListener("submit", begin);
  captureBtn.addEventListener("click", capture);
  startOverBtn.addEventListener("click", startOver);
  finishBtn.addEventListener("click", finish);

  updateBeginDisabled();
  render();

  return function unmount() {
    cam.stop();
  };
}
