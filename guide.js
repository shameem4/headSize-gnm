/**
 * Capture guidance: head-sized guide oval, alignment checks, and a progress ring
 * drawn along the live head outline.
 * @module guide
 */

// MediaPipe face-contour landmarks, in order around the face starting at the forehead
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const FOREHEAD = 10;
const CHIN = 152;
const CHEEK_RIGHT = 234;
const CHEEK_LEFT = 454;
const NOSE_TIP = 1;

// Working distance range (cm), measured from the iris
const MIN_DISTANCE_CM = 50;
// Beyond ~1 m a webcam face gets too small for MediaPipe's short-range face detector
const MAX_DISTANCE_CM = 100;
// The guide oval is drawn where an average head (crown to chin) would appear at the
// current distance, with a little margin
const HEAD_HEIGHT_MM = 230;
const OVAL_MARGIN = 1.1;
const OVAL_ASPECT = 0.72; // width / height
// Head centre may be off the oval centre by this fraction of the oval height
const MAX_OFFSET = 0.2;
// Nose-to-cheek distance ratio beyond which the head counts as turned (~20 degrees)
const MAX_TURN_LOG_RATIO = 0.35;

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{focalNorm: number, mirrored: boolean}} options - focal length as a fraction
 *   of video width, and whether the canvas is shown mirrored
 */
export function createGuide(canvas, { focalNorm, mirrored }) {
  const ctx = canvas.getContext("2d");
  let w = 1;
  let h = 1;
  let ovalRy = null; // smoothed, in display pixels

  const toDisplay = (lm, i) => [lm[i].x * w, lm[i].y * h];

  function resize(width, height) {
    const dpr = window.devicePixelRatio || 1;
    w = width;
    h = height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Oval half-height (display px) for a head at distanceCm, smoothed between frames */
  function updateOval(distanceCm) {
    const d = Math.min(MAX_DISTANCE_CM, Math.max(MIN_DISTANCE_CM, distanceCm)) * 10;
    const target = (OVAL_MARGIN * focalNorm * w * HEAD_HEIGHT_MM) / d / 2;
    ovalRy = ovalRy === null ? target : ovalRy + 0.2 * (target - ovalRy);
  }

  /**
   * Is the head at a usable distance, centred in the oval, and facing the camera?
   * @param {Array} lm - landmarks
   * @param {number} distanceCm - camera distance from the iris
   * @returns {{ok: boolean, reason: null|"center"|"closer"|"back"|"turn", faceHeight: number}}
   */
  function assess(lm, distanceCm) {
    updateOval(distanceCm);
    const [, topY] = toDisplay(lm, FOREHEAD);
    const [, chinY] = toDisplay(lm, CHIN);
    const faceHeight = (chinY - topY) / h;

    const head = headContour(lm);
    const xs = head.map((p) => p[0]), ys = head.map((p) => p[1]);
    const offX = ((Math.min(...xs) + Math.max(...xs)) / 2 - w / 2) / (2 * ovalRy);
    const offY = ((Math.min(...ys) + Math.max(...ys)) / 2 - h / 2) / (2 * ovalRy);

    const nose = toDisplay(lm, NOSE_TIP);
    const dist = (p) => Math.hypot(p[0] - nose[0], p[1] - nose[1]);
    const turn = Math.abs(Math.log(dist(toDisplay(lm, CHEEK_RIGHT)) / dist(toDisplay(lm, CHEEK_LEFT))));

    let reason = null;
    if (distanceCm < MIN_DISTANCE_CM) reason = "back";
    else if (distanceCm > MAX_DISTANCE_CM) reason = "closer";
    else if (Math.abs(offX) > MAX_OFFSET || Math.abs(offY) > MAX_OFFSET) reason = "center";
    else if (turn > MAX_TURN_LOG_RATIO) reason = "turn";
    return { ok: !reason, reason, faceHeight };
  }

  /** Head outline: the face contour pushed out to take in the top and sides of the head */
  function headContour(lm) {
    const pts = FACE_OVAL.map((i) => toDisplay(lm, i));
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    return pts.map(([x, y]) => [cx + 1.12 * (x - cx), cy + (y < cy ? 1.35 : 1.06) * (y - cy)]);
  }

  function drawOval(color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.setLineDash([12, 10]);
    ctx.beginPath();
    const ry = ovalRy ?? (OVAL_MARGIN * focalNorm * w * HEAD_HEIGHT_MM) / (MIN_DISTANCE_CM * 10) / 2;
    ctx.ellipse(w / 2, h / 2, OVAL_ASPECT * ry, ry, 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  /** Full outline faintly, and the first `fraction` of it highlighted: from the top of the
   * head, clockwise as seen on screen */
  function drawProgress(lm, fraction) {
    let pts = headContour(lm);
    // Landmark order runs one way round the face; flip it if that isn't clockwise on screen
    // (the canvas may be mirrored). pts[0] is the top of the head.
    const screenX = (p) => (mirrored ? w - p[0] : p[0]);
    if (screenX(pts[1]) < screenX(pts[0])) pts = [pts[0], ...pts.slice(1).reverse()];
    pts.push(pts[0]);
    ctx.save();
    ctx.lineJoin = ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();

    const lengths = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
    let remaining = fraction * lengths.reduce((a, b) => a + b, 0);
    ctx.strokeStyle = "#00ffc8";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    for (let i = 0; i < lengths.length && remaining > 0; i++) {
      const t = Math.min(1, remaining / lengths[i]);
      ctx.lineTo(pts[i][0] + t * (pts[i + 1][0] - pts[i][0]), pts[i][1] + t * (pts[i + 1][1] - pts[i][1]));
      remaining -= lengths[i];
    }
    ctx.stroke();
    ctx.restore();
  }

  /**
   * @param {"align"|"capture"|"done"} phase
   * @param {Array|null} lm - landmarks, if a face is visible
   * @param {boolean} aligned - head currently in position
   * @param {number} fraction - capture progress 0..1
   */
  function draw(phase, lm, aligned, fraction) {
    ctx.clearRect(0, 0, w, h);
    if (phase === "align") drawOval(aligned ? "#00ffc8" : "rgba(255, 255, 255, 0.85)");
    else if (phase === "capture") lm ? drawProgress(lm, fraction) : drawOval("rgba(255, 255, 255, 0.85)");
  }

  /** Outline a quadrilateral given in normalized (0..1) video coordinates */
  function drawQuad(points, color = "#00ffc8") {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(p.x * w, p.y * h) : ctx.moveTo(p.x * w, p.y * h)));
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  /** Dots joined by polylines, each dot numbered; points normalized (0..1), lines as
   * [start, end) index ranges */
  function drawLines(points, lines, color = "#00ffc8") {
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = 2;
    for (const [a, b] of lines) {
      ctx.beginPath();
      for (let i = a; i < b; i++) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, points[i].x * w, points[i].y * h);
      ctx.stroke();
    }
    ctx.font = "10px system-ui, sans-serif";
    points.forEach((p, i) => {
      const x = p.x * w, y = p.y * h;
      ctx.fillRect(x - 2, y - 2, 4, 4);
      // The canvas is shown mirrored: flip the text back so it reads normally
      ctx.save();
      ctx.translate(x, y);
      if (mirrored) ctx.scale(-1, 1);
      ctx.fillText(String(i), 3, -3);
      ctx.restore();
    });
    ctx.restore();
  }

  return { resize, assess, draw, drawQuad, drawLines };
}
