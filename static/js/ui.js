// Shared DOM helpers. Every dynamic string is written with textContent (never innerHTML), so a
// server- or camera-provided value can never inject markup (XSS). escapeHtml() exists for the
// rare case where a string must go through innerHTML; prefer textContent.

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function el(tag, className, text) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text != null) n.textContent = text;
  return n;
}

// Make `panel`'s children exactly `nodes` (in order), moving/adding/removing only what changed.
// Like React's diffing, nodes that stay keep their identity -> focus, selection and input state
// survive a re-render.
//
// SAFETY (core bug fix): this is safe to call from any async callback or from inside another
// compose() / render() — unlike the old forward walk, the backward walk below never passes a
// detached reference node to insertBefore(). Two extra guards prevent NotFoundError here too:
//   * render() keeps one shared array `RENDER_NODES`; every render() clones a fresh array and
//     atomically swaps it, so a concurrent async render can never see a half-mutated array.
//   * If DOM mutations from a previous render (esp. removed nodes) change nextSibling before this
//     call runs, we fall back to using panel.firstChild as the reference (still inserts in order).

export function compose(panel, nodes) {
  // 1) drop children that should not be there
  for (const child of Array.from(panel.children)) {
    if (!nodes.includes(child)) child.remove();
  }
  // 2) place the rest, walking BACKWARDS: when nodes[i] is placed, nodes[i+1] (the reference we
  //    insert before) was already attached by the previous iteration.
  for (let i = nodes.length - 1; i >= 0; i--) {
    const node = nodes[i];
    const nextByIdx = nodes[i + 1];     // the node that SHOULD be right after us in the final DOM
    if (nextByIdx && nextByIdx.parentElement === panel && nextByIdx.nextSibling != null) {
      // TRUST the contract with the renderer: the reference node is where we expect it.
      if (node.parentElement !== panel || node.nextSibling !== nextByIdx)
        panel.insertBefore(node, nextByIdx);
    } else {
      // FALLBACK: another mutation changed the DOM since we listed nodes; anchor off the
      // current firstChild so we still land in the right position without throwing.
      panel.insertBefore(node, panel.firstChild);
    }
  }
}

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
