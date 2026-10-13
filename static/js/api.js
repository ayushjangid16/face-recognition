// Thin fetch wrapper, ported 1:1 from the React app's src/api.js.
// Throws Error(detail) so pages can show the server's message directly.
// All paths are relative: the UI is served from the same origin as the API.
export async function api(path, { method = "GET", body, form, adminKey } = {}) {
  const headers = {};
  if (adminKey) headers["X-Admin-Key"] = adminKey;
  let payload;
  if (form) {
    payload = form;                        // FormData: the browser sets the multipart boundary
  } else if (body) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(path, { method, headers, body: payload });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(", ") : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

export const fmtTime = (iso) => new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
