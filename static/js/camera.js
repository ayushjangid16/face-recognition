// Webcam handling, ported 1:1 from the React app's src/useCamera.js:
// same getUserMedia constraints, same capture (hidden canvas, max 640px wide, JPEG), same
// error messages, and the same start/stop lifecycle (stop on page-unmount).
//
// NOTE: getUserMedia is only allowed in a secure context -> serve over HTTPS or localhost.

let sharedStream = null;
let startingPromise = null;

function isLive(stream) {
  if (!stream) return false;
  const tracks = stream.getVideoTracks();
  return tracks.length > 0 && tracks[0].readyState === "live";
}

async function getStream() {
  if (isLive(sharedStream)) {
    return sharedStream;
  }
  if (startingPromise) {
    return startingPromise;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("insecure");
  }
  if (sharedStream) {
    try {
      sharedStream.getTracks().forEach((t) => t.stop());
    } catch {}
    sharedStream = null;
  }
  startingPromise = (async () => {
    try {
      sharedStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      return sharedStream;
    } finally {
      startingPromise = null;
    }
  })();
  return startingPromise;
}

export function createCamera(video, { loading, error: errorEl } = {}) {
  let cancelled = false;
  let ready = false;
  let canvas = null;

  function stop() {
    cancelled = true;
    ready = false;
    if (video) {
      try { video.pause(); } catch {}
      video.srcObject = null;
    }
  }

  function showError(message) {
    loading?.remove();
    if (!errorEl) return;
    errorEl.textContent = message;
    errorEl.style.display = "";
    video.parentElement?.appendChild(errorEl);   // React renders the overlay only on error
  }

  async function start() {
    try {
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;
      const stream = await getStream();
      if (cancelled) return stop();
      video.srcObject = stream;

      const onPlaying = () => {
        ready = true;
        loading?.remove();
        errorEl?.remove();
      };

      if (video.readyState >= 2) {
        onPlaying();
      } else {
        video.onloadeddata = onPlaying;
        video.onplaying = onPlaying;
      }

      await video.play().catch(() => {});
      ready = true;
      loading?.remove();
      errorEl?.remove();
    } catch (e) {
      if (cancelled) return;
      if (e.message === "insecure") showError("Camera needs HTTPS (or localhost).");
      else if (e.name === "NotAllowedError") showError("Camera permission was denied.");
      else showError("Could not open the camera: " + (e.message || e.name));
    }
  }

  // JPEG Blob (max 640px wide, unmirrored: the server expects raw frames).
  function capture(quality = 0.85) {
    return new Promise((resolve) => {
      if (!video || !video.videoWidth) return resolve(null);
      if (!canvas) canvas = document.createElement("canvas");
      const scale = Math.min(1, 640 / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(resolve, "image/jpeg", quality);
    });
  }

  return { start, stop, capture, isReady: () => ready };
}

export function releaseCamera() {
  if (sharedStream) {
    sharedStream.getTracks().forEach((t) => t.stop());
    sharedStream = null;
  }
}
