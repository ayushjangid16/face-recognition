// Tab host, ported from React src/App.jsx. The React app had no router: "Check in" and
// "Enrol member (staff)" are two tabs on the one URL, and switching tabs remounts the panel
// (React key=), which stops the camera and closes the WebSocket. We reproduce that by cloning
// the inert <template> for the tab and tearing it down again on switch.
import { mountScanner } from "./pages/scan.js";
import { mountEnroller } from "./pages/enroll.js";

const tabs = {
  scan: { mount: mountScanner },
  enroll: { mount: mountEnroller },
};

const buttons = Array.from(document.querySelectorAll("nav [data-tab]"));
const containers = {
  scan: document.getElementById("panel-scan"),
  enroll: document.getElementById("panel-enroll"),
};

let active = null;
let cleanup = null;

function detachVideos(element) {
  if (!element) return;
  const videos = element.querySelectorAll("video");
  for (const v of videos) {
    try { v.pause(); } catch {}
    v.srcObject = null;
  }
}

function switchTab(tab) {
  if (!tabs[tab] || tab === active) return;                 // clicking the active tab does nothing
  if (cleanup) {
    try { cleanup(); } catch (e) { console.error("Cleanup error:", e); }
    cleanup = null;
  }
  for (const c of Object.values(containers)) {
    detachVideos(c);
    c.replaceChildren();
  }

  active = tab;
  for (const btn of buttons) btn.classList.toggle("active", btn.dataset.tab === tab);

  const template = document.getElementById(`tpl-${tab}`);
  if (template) {
    containers[tab].appendChild(template.content.cloneNode(true));
    try {
      cleanup = tabs[tab].mount(containers[tab]) || null;
    } catch (e) {
      console.error(`Mount error for tab ${tab}:`, e);
    }
  }
}

for (const btn of buttons) btn.addEventListener("click", () => switchTab(btn.dataset.tab));

switchTab("scan");                                          // default tab, like the React app
