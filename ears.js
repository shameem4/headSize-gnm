/**
 * Ear measurement from side views: Ear_Landmarker (BlazeEar + 55-point model,
 * github.com/shameem4/Ear_Landmarker; research use, trained on non-commercial data)
 * finds the ear, and each point is projected onto the fitted GNM head's ear plane to
 * get millimetres, corrected for the viewing angle.
 * @module ears
 *
 * Point numbering (iBUG ears): helix 0-13, lobe 14-19, inner helix (antihelix) 20-34,
 * tragus 35-38, canal 39, antitragus 40-42, concha 43-46, inferior crus 47-49,
 * superior crus 50-54.
 */

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";
// Helix and lobe down to its lowest point: the lobe arc's last points (17-19) run
// forward to where the lobe joins the cheek, past what a ruler measures to
const OUTLINE = [...Array(17).keys()];
const CONCHA = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]; // tragus to concha back wall
const TRAGUS_LOW = 38;
const ANTITRAGUS = 40;
// Search a square around the face (this x face height): the ear finder shrinks its
// input to 128 px, and a whole 1080p frame would leave an ear ~15 px tall
const HEAD_CROP = 2.2;
// Parts of the crop outside the video are filled with the grey Ear_Landmarker was
// trained with (data/dataset.py pads with 128). The pipeline pads its own ear ROI the
// same way, so the crop needs no extra margin.
const PAD_GREY = "rgb(128, 128, 128)";
// The crop only moves when the face centre drifts more than this fraction of its side:
// a steady crop keeps the landmark tracker's coordinates consistent between frames
const CROP_DEADBAND = 0.1;

export const EAR_LINES = [
  [0, 20],   // helix and lobe
  [20, 35],  // antihelix
  [35, 47],  // tragus, canal, antitragus, concha
  [47, 50],  // inferior crus
  [50, 55],  // superior crus (a separate ridge: 49 and 50 aren't joined)
];

let loading = null;

/** Load ONNX Runtime and both ear models */
export function loadEarPipeline() {
  loading ??= new Promise((resolve, reject) => {
    const el = document.createElement("script");
    el.src = ORT_URL;
    el.onload = resolve;
    el.onerror = () => reject(new Error(`Could not load ${ORT_URL}`));
    document.head.append(el);
  })
    .then(() => import("./ear/earlandmarker_inference.js"))
    .then(async ({ EarLandmarkerPipeline }) => {
      // Temporal smoothing on (the library's One Euro tracker): detectEars keeps the crop
      // steady so the tracker sees consistent coordinates
      const pipeline = new EarLandmarkerPipeline({ confidenceThreshold: 0.5 });
      await pipeline.load("ear/BlazeEar_web.onnx", "ear/EarLandmarker_web.onnx");
      return pipeline;
    });
  return loading;
}

/**
 * Find ears in a head-sized crop around the face (a fresh canvas per call, so overlapping
 * calls can't share pixels). The crop is held steady: its size is fixed per ear session
 * and it only moves when the face drifts past CROP_DEADBAND, and then the pipeline's
 * smoothing restarts, so the tracker never smooths across a jump.
 * @param {{side: number, cx: number, cy: number}|null} steady - crop state for this ear
 *   session (updated in place); null on the first call
 * @returns {Promise<{ears: Array<{bbox, confidence, landmarks}>, crop: {canvas, x, y}, steady}>}
 *   ears in video pixels; crop: the searched square and its position in the frame
 */
export async function detectEars(pipeline, video, faceLandmarks, steady = null) {
  const W = video.videoWidth, H = video.videoHeight;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of faceLandmarks) {
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  const side = steady?.side ?? Math.round(HEAD_CROP * (y1 - y0) * H);
  // Not clamped to the frame: parts outside the video are padded grey like the margin
  const wantX = Math.round(((x0 + x1) / 2) * W - side / 2);
  const wantY = Math.round(((y0 + y1) / 2) * H - side / 2);
  if (!steady || Math.hypot(wantX - steady.cx, wantY - steady.cy) > CROP_DEADBAND * side) {
    steady = { side, cx: wantX, cy: wantY };
    pipeline.resetSmoothing?.();
  }
  const { cx, cy } = steady;
  const size = side;
  const crop = document.createElement("canvas");
  crop.width = crop.height = size;
  const ctx = crop.getContext("2d");
  ctx.fillStyle = PAD_GREY;
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(video, -cx, -cy);
  const results = await pipeline.detect(crop);
  return {
    ears: results.map((r) => ({
      confidence: r.confidence,
      bbox: { xmin: r.bbox.xmin + cx, xmax: r.bbox.xmax + cx, ymin: r.bbox.ymin + cy, ymax: r.bbox.ymax + cy },
      landmarks: r.landmarks.map((p) => ({ x: p.x + cx, y: p.y + cy })),
    })),
    crop: { canvas: crop, x: cx, y: cy },
    steady,
  };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/**
 * Camera-frame ray through video pixel (u, v), in the observation frame (x right,
 * y up, camera looking down -z; see observe() in main.js)
 */
function pixelRay(u, v, W, H, focalPx) {
  return [(u - W / 2) / focalPx, -(v - H / 2) / focalPx, -1];
}

/** Project a camera-frame point to video pixels */
export function projectPoint(p, W, H, focalPx) {
  return { x: W / 2 + (focalPx * p[0]) / -p[2], y: H / 2 - (focalPx * p[1]) / -p[2] };
}

/**
 * Measure one ear: intersect each landmark's ray with the ear plane, then take
 * distances in 3D.
 * @param {{x: number, y: number}[]} landmarks - 55 points, video pixels
 * @param {{centroid: number[], normal: number[]}} plane - camera frame, metres
 * @returns {Object} mm, plus viewCos (how squarely the camera sees the ear plane)
 */
export function measureEar(landmarks, plane, W, H, focalPx) {
  const offset = dot(plane.normal, plane.centroid);
  const P = landmarks.map((p) => {
    const d = pixelRay(p.x, p.y, W, H, focalPx);
    const t = offset / dot(plane.normal, d);
    return [d[0] * t * 1000, d[1] * t * 1000, d[2] * t * 1000];
  });
  const viewCos = Math.abs(dot(plane.normal, plane.centroid)) / norm(plane.centroid);

  // The ear's long axis in its plane (principal axis of the outline): length along it,
  // widths across it
  const n = plane.normal;
  const e = norm(cross(n, [0, 1, 0])) > 0.1 ? cross(n, [0, 1, 0]) : cross(n, [1, 0, 0]);
  const u = e.map((x) => x / norm(e)), v = cross(n, u);
  const mean = [0, 1, 2].map((k) => OUTLINE.reduce((s, i) => s + P[i][k], 0) / OUTLINE.length);
  let sxx = 0, syy = 0, sxy = 0;
  for (const i of OUTLINE) {
    const d = sub(P[i], mean), x = dot(d, u), y = dot(d, v);
    sxx += x * x; syy += y * y; sxy += x * y;
  }
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const a = [0, 1, 2].map((k) => Math.cos(th) * u[k] + Math.sin(th) * v[k]);
  const b = cross(n, a);
  const extent = (idx, axis) => {
    const s = idx.map((i) => dot(P[i], axis));
    return Math.max(...s) - Math.min(...s);
  };
  return {
    ear_length_measured: extent(OUTLINE, a),
    ear_width: extent(OUTLINE, b),
    concha_height: extent(CONCHA, a),
    concha_width: extent(CONCHA, b),
    tragus_gap: norm(sub(P[TRAGUS_LOW], P[ANTITRAGUS])),
    viewCos,
  };
}

/** Solve a small dense linear system A x = b (Gaussian elimination, partial pivoting) */
function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/**
 * Scale check with the printed marker held beside the ear: its square measured on the
 * ear plane (should equal its printed size), and its own distance and angle (from its
 * four corners, independent of the head model) against the ear plane on the same ray
 * @param {{x: number, y: number}[]} corners - marker square corners, video pixels, in order
 * @param {{centroid: number[], normal: number[]}} plane - ear plane, camera frame, metres
 * @param {number} sideMm - printed size of the square
 * @returns {{onPlaneMm: number, ownDistMm: number, planeDistMm: number, angleDeg: number}}
 */
export function markerCheck(corners, plane, W, H, focalPx, sideMm) {
  // Size on the ear plane: mean side of the corners projected onto it
  const offset = dot(plane.normal, plane.centroid);
  const onPlane = corners.map((p) => {
    const d = pixelRay(p.x, p.y, W, H, focalPx);
    const t = offset / dot(plane.normal, d);
    return d.map((v) => v * t * 1000);
  });
  const onPlaneMm = [0, 1, 2, 3].reduce((s, i) => s + norm(sub(onPlane[i], onPlane[(i + 1) % 4])), 0) / 4;

  // Marker pose from the homography square -> normalized image (camera looking down +z,
  // y down), converted to the observation frame (y up, looking down -z) at the end
  const square = [[0, 0], [sideMm, 0], [sideMm, sideMm], [0, sideMm]];
  const A = [], b = [];
  corners.forEach((p, i) => {
    const [X, Y] = square[i];
    const x = (p.x - W / 2) / focalPx, y = (p.y - H / 2) / focalPx;
    A.push([X, Y, 1, 0, 0, 0, -x * X, -x * Y]); b.push(x);
    A.push([0, 0, 0, X, Y, 1, -y * X, -y * Y]); b.push(y);
  });
  const h = solve(A, b);
  const h1 = [h[0], h[3], h[6]], h2 = [h[1], h[4], h[7]], h3 = [h[2], h[5], 1];
  const lambda = 2 / (norm(h1) + norm(h2));
  const r1 = h1.map((v) => v * lambda), r2 = h2.map((v) => v * lambda), t = h3.map((v) => v * lambda);
  const centre = [0, 1, 2].map((k) => ((r1[k] + r2[k]) * sideMm) / 2 + t[k]);
  const toObs = (v) => [v[0], -v[1], -v[2]];
  const centreObs = toObs(centre);
  const normalObs = toObs(cross(r1, r2)).map((v, _, a) => v / norm(a));

  // Ear plane distance along the ray through the marker's centre
  const tPlane = (offset * 1000) / dot(plane.normal, centreObs.map((v) => v / norm(centreObs)));
  return {
    onPlaneMm,
    ownDistMm: norm(centreObs),
    planeDistMm: tPlane,
    angleDeg: (Math.acos(Math.min(1, Math.abs(dot(normalObs, plane.normal)))) * 180) / Math.PI,
  };
}
