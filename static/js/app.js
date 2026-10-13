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

function switchTab(tab) {
  if (!tabs[tab] || tab === active) return;                 // clicking the active tab does nothing
  if (active) { cleanup?.(); containers[active].replaceChildren(); }

  const template = document.getElementById(`tpl-${tab}`);
  containers[tab].appendChild(template.content.cloneNode(true));
  cleanup = tabs[tab].mount(containers[tab]) || null;
  active = tab;

  for (const btn of buttons) btn.classList.toggle("active", btn.dataset.tab === tab);
}

for (const btn of buttons) btn.addEventListener("click", () => switchTab(btn.dataset.tab));

switchTab("scan");                                          // default tab, like the React app
