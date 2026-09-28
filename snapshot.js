/**
 * Snapshot: save what's on screen (video + overlays) with the measurement table as a PNG,
 * and the same data as JSON.
 * @module snapshot
 */

const PANEL_WIDTH = 520;

/**
 * Compose the PNG: video and overlay canvases as displayed (mirrored if shown mirrored),
 * with the measurement table to the right.
 * @param {Object} args
 * @param {HTMLVideoElement} args.video
 * @param {HTMLCanvasElement[]} args.overlays - canvases drawn over the video, in order
 * @param {boolean} args.mirrored
 * @param {string[][]} args.lines - text rows: [label, value] or [heading]
 * @returns {Promise<Blob>}
 */
export function composeImage({ video, overlays, mirrored, lines }) {
  // video: a <video>, or a canvas already holding the frame
  const vw = video.videoWidth || video.width, vh = video.videoHeight || video.height;
  const canvas = document.createElement("canvas");
  canvas.width = vw + PANEL_WIDTH;
  const headings = lines.filter((l) => l[1] === undefined).length;
  canvas.height = Math.max(vh, 44 + lines.length * 30 + headings * 10 + 20);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  if (mirrored) {
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, vw, vh);
  for (const overlay of overlays) ctx.drawImage(overlay, 0, 0, vw, vh);
  ctx.restore();

  let y = 44;
  for (const [label, value] of lines) {
    if (value === undefined) {
      ctx.fillStyle = "#00ffc8";
      ctx.font = "600 22px system-ui, sans-serif";
      ctx.fillText(label, vw + 24, (y += 10));
    } else {
      ctx.fillStyle = "#ddd";
      ctx.font = "18px system-ui, sans-serif";
      ctx.fillText(label, vw + 24, y);
      ctx.fillStyle = "#fff";
      ctx.font = "600 18px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(value, vw + PANEL_WIDTH - 24, y);
      ctx.textAlign = "left";
    }
    y += 30;
  }
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

/** The raw video frame (no overlays, not mirrored) as PNG, for tuning detection;
 * video may also be a canvas holding a frame */
export function rawFrame(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || video.width;
  canvas.height = video.videoHeight || video.height;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}

// fflate (MIT): small zip writer, loaded on first use
const FFLATE_URL = "https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js";

/**
 * Bundle files into one zip (images stored as-is, since they are already compressed)
 * @param {Object<string, Blob>} files - file name -> contents
 * @returns {Promise<Blob>}
 */
export async function zipFiles(files) {
  const { zipSync } = await import(FFLATE_URL);
  const entries = {};
  for (const [name, blob] of Object.entries(files)) {
    const level = /\.(png|jpe?g)$/i.test(name) ? 0 : 6;
    entries[name] = [new Uint8Array(await blob.arrayBuffer()), { level }];
  }
  return new Blob([zipSync(entries)], { type: "application/zip" });
}

/** Trigger a browser download of a Blob */
export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** File-name stem: headsize-gnm-YYYYMMDD-HHMMSS[-label] */
export function snapshotName(label) {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  const slug = (label || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `headsize-gnm-${stamp}${slug ? `-${slug}` : ""}`;
}
