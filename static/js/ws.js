// WebSocket client for the live scan, on the exact endpoint and protocol the React app used:
//   ws(s)://{location.host}/ws/scan/{token}   (wss when the page is served over https)
// Auth is the same as before: none beyond the one-time token in the URL; the server checks the
// Origin header against CORS_ORIGINS. There is no heartbeat and no auto-reconnect - the React
// app had neither: a drop surfaces as "Connection lost. Please try again." and the user restarts.
export function openScanSocket(token, { onOpen, onMessage, onClose } = {}) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/scan/${token}`);
  if (onOpen) ws.onopen = onOpen;
  if (onMessage) ws.onmessage = onMessage;
  if (onClose) ws.onclose = onClose;
  return ws;
}
