/**
 * Bank-card calibration: measure the user's own iris size from a card of known width
 * held flat against the forehead.
 * @module calibrate
 *
 * The frozen frame is shown zoomed in on the card with two vertical lines the user drags
 * onto the card's left and right edges. For privacy the view is pixelated except narrow
 * strips around the two lines, so the card's face is never shown sharp; the frozen frame
 * is kept in memory only and dropped when marking ends. The card's width in pixels gives mm per pixel at
 * the forehead; the forehead is a little in front of the eyes, so that is scaled to eye
 * depth using the tracked landmarks' relative depth. Result: iris diameter in mm.
 */

// ISO/IEC 7810 ID-1 (bank card) width
const CARD_WIDTH_MM = 85.6;
// Plausible human iris diameters; outside this the edges were probably misplaced
export const IRIS_MIN_MM = 10;
export const IRIS_MAX_MM = 13.5;
// Crop shown for marking: this many card widths across
const CROP_CARD_WIDTHS = 2.2;
// Sharp strip around each line, as a fraction of the expected card width (each side)
const SHARP_STRIP = 0.05;
// Pixelation block size (display px) for everything outside the strips
const PIXEL_BLOCK = 14;
// Auto-detected edges must be this many times stronger than a typical column
const MIN_EDGE_CONFIDENCE = 4;

/**
 * Iris diameter (mm) from the width of an object of known size (card or printed marker)
 * held flat on the forehead.
 * mm/px at the object (forehead) = objectMm / cardPx. The iris is further away by
 * delta, so mm/px there is larger by D / (D - delta), where delta / D depends only on
 * relative depth: (zEyes - zForehead) * depthScale / focalNorm.
 * @param {Object} p
 * @param {number} p.cardPx - object width, video pixels
 * @param {number} [p.objectMm] - object's real width (default: bank card)
 * @param {number} p.irisPx - iris diameter, video pixels
 * @param {number} p.zEyes - MediaPipe z of the pupils (mean)
 * @param {number} p.zForehead - MediaPipe z where the card rests
 * @param {number} p.focalNorm - focal length / video width
 * @param {number} p.depthScale - correction for MediaPipe's too-deep z
 */
export function irisFromCard({ cardPx, irisPx, zEyes, zForehead, focalNorm, depthScale, objectMm = CARD_WIDTH_MM }) {
  const deltaOverD = ((zEyes - zForehead) * depthScale) / focalNorm;
  const mmPerPxAtEyes = (objectMm / cardPx) / (1 - deltaOverD);
  return irisPx * mmPerPxAtEyes;
}

/**
 * Find the card's left and right edges near the expected position (no model needed: the
 * card's location and approximate width are known). In a horizontal band inside the
 * card, each column's horizontal gradient is averaged down the band; a card edge gives a
 * strong same-signed average, skin texture cancels. The best pair of columns about the
 * expected width apart wins, refined to sub-pixel by a parabola through each peak.
 * @param {HTMLCanvasElement} frame - frozen video frame
 * @param {{centerX: number, centerY: number, expected: number}} guess - video px
 * @returns {{left: number, right: number, confidence: number}|null}
 */
function findCardEdges(frame, { centerX, centerY, expected }) {
  const x0 = Math.max(1, Math.round(centerX - 0.9 * expected));
  const x1 = Math.min(frame.width - 2, Math.round(centerX + 0.9 * expected));
  const y0 = Math.max(0, Math.round(centerY - 0.2 * expected));
  const y1 = Math.min(frame.height - 1, Math.round(centerY + 0.2 * expected));
  const W = x1 - x0 + 3, H = y1 - y0 + 1;
  if (W < 10 || H < 4) return null;
  const px = frame.getContext("2d").getImageData(x0 - 1, y0, W, H).data;
  const gray = (x, y) => {
    const i = (y * W + x) * 4;
    return 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  };
  // Column profile of |mean horizontal gradient| (index 0 = video column x0)
  const profile = new Float32Array(x1 - x0 + 1);
  for (let c = 0; c < profile.length; c++) {
    let sum = 0;
    for (let y = 0; y < H; y++) sum += gray(c + 2, y) - gray(c, y);
    profile[c] = Math.abs(sum / H);
  }
  const sorted = Array.from(profile).sort((a, b) => a - b);
  const noise = sorted[sorted.length >> 1] + 1e-6;

  // Best pair about the expected width apart
  let best = null;
  const minSep = 0.8 * expected, maxSep = 1.2 * expected;
  for (let a = 0; a < profile.length; a++) {
    for (let b = Math.ceil(a + minSep); b < profile.length && b <= a + maxSep; b++) {
      const score = profile[a] + profile[b];
      if (!best || score > best.score) best = { a, b, score };
    }
  }
  if (!best) return null;
  const refine = (c) => {
    if (c <= 0 || c >= profile.length - 1) return c;
    const [l, m, r] = [profile[c - 1], profile[c], profile[c + 1]];
    const d = l - 2 * m + r;
    return d ? c + (0.5 * (l - r)) / d : c;
  };
  return {
    left: x0 + refine(best.a),
    right: x0 + refine(best.b),
    // Weaker edge relative to the typical column: ~1 means no clear edges
    confidence: Math.min(profile[best.a], profile[best.b]) / noise,
  };
}

/**
 * Marking UI over the video
 * @param {Object} els - { root, canvas, text, confirm, retake, cancel }
 * @param {{mirrored: boolean, onConfirm: (irisMm:number)=>void, onRetake: ()=>void, onCancel: ()=>void}} options
 */
export function createCardCalibration(els, { mirrored, onConfirm, onRetake, onCancel }) {
  const ctx = els.canvas.getContext("2d");
  let w = 1, h = 1;
  let frame = null;   // canvas holding the frozen video frame
  let info = null;    // { irisPx, zEyes, zForehead, focalNorm, depthScale }
  let crop = null;    // { x, y, cw, ch } in video px
  let lines = [0, 0]; // card edges, video px x
  let expectedPx = 0; // expected card width, video px
  let dragging = -1;
  let found = null; // auto-detected edges, if confident
  const mosaic = document.createElement("canvas");

  function resize(width, height) {
    const dpr = window.devicePixelRatio || 1;
    w = width;
    h = height;
    els.root.style.width = `${width}px`;
    els.root.style.height = `${height}px`;
    els.canvas.width = Math.round(width * dpr);
    els.canvas.height = Math.round(height * dpr);
    els.canvas.style.width = `${width}px`;
    els.canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (frame) draw();
  }

  // Video px <-> canvas px (the crop fills the canvas, mirrored like the live view)
  const toCanvasX = (vx) => {
    const u = (vx - crop.x) / crop.cw;
    return (mirrored ? 1 - u : u) * w;
  };
  const toVideoX = (cx) => {
    const u = cx / w;
    return crop.x + (mirrored ? 1 - u : u) * crop.cw;
  };

  function currentIris() {
    const cardPx = Math.abs(lines[1] - lines[0]);
    return cardPx > 0 ? irisFromCard({ cardPx, ...info }) : NaN;
  }

  /** Draw the crop (mirrored like the live view) into ctx, optionally only inside a clip */
  function drawCrop(target, clip) {
    target.save();
    if (clip) clip(target);
    if (mirrored) {
      target.translate(w, 0);
      target.scale(-1, 1);
    }
    target.drawImage(frame, crop.x, crop.y, crop.cw, crop.ch, 0, 0, w, h);
    target.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, w, h);
    // Pixelated everywhere: draw small, then scale up without smoothing
    mosaic.width = Math.max(1, Math.round(w / PIXEL_BLOCK));
    mosaic.height = Math.max(1, Math.round(h / PIXEL_BLOCK));
    const m = mosaic.getContext("2d");
    m.save();
    if (mirrored) {
      m.translate(mosaic.width, 0);
      m.scale(-1, 1);
    }
    m.drawImage(frame, crop.x, crop.y, crop.cw, crop.ch, 0, 0, mosaic.width, mosaic.height);
    m.restore();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mosaic, 0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    // Sharp only in narrow strips around the lines, so the card's edges can be placed
    const half = (SHARP_STRIP * expectedPx * w) / crop.cw;
    drawCrop(ctx, (c) => {
      c.beginPath();
      for (const vx of lines) c.rect(toCanvasX(vx) - half, 0, 2 * half, h);
      c.clip();
    });

    for (const vx of lines) {
      const x = toCanvasX(vx);
      ctx.strokeStyle = "#00ffc8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillStyle = "#00ffc8";
      ctx.beginPath();
      ctx.arc(x, h / 2, 10, 0, 2 * Math.PI);
      ctx.fill();
    }
    const iris = currentIris();
    const how = found
      ? "Card edges found automatically: check the lines sit on its left and right edges (drag to adjust)."
      : "Couldn't find the card's edges automatically: drag the lines onto its left and right edges.";
    els.text.textContent = `${how} Card ${Math.abs(lines[1] - lines[0]).toFixed(0)} px → your iris ${iris.toFixed(2)} mm`;
  }

  function pointerX(e) {
    const r = els.canvas.getBoundingClientRect();
    return e.clientX - r.left;
  }

  els.canvas.addEventListener("pointerdown", (e) => {
    const x = pointerX(e);
    const d = lines.map((vx) => Math.abs(toCanvasX(vx) - x));
    dragging = d[0] <= d[1] ? 0 : 1;
    lines[dragging] = toVideoX(x);
    els.canvas.setPointerCapture(e.pointerId);
    draw();
  });
  els.canvas.addEventListener("pointermove", (e) => {
    if (dragging < 0) return;
    lines[dragging] = toVideoX(pointerX(e));
    draw();
  });
  els.canvas.addEventListener("pointerup", () => (dragging = -1));

  els.confirm.addEventListener("click", () => {
    const iris = currentIris();
    if (!(iris >= IRIS_MIN_MM && iris <= IRIS_MAX_MM)) {
      els.text.textContent =
        `That gives an iris of ${iris.toFixed(2)} mm, outside the human range ` +
        `(${IRIS_MIN_MM}–${IRIS_MAX_MM} mm). Check the lines are on the card's edges, or retake.`;
      return;
    }
    stop();
    onConfirm(iris);
  });
  els.retake.addEventListener("click", () => {
    stop();
    onRetake();
  });
  els.cancel.addEventListener("click", () => {
    stop();
    onCancel();
  });

  /**
   * Show the frozen frame for marking
   * @param {HTMLCanvasElement} frozen - video frame at native resolution
   * @param {Object} data - { irisPx, zEyes, zForehead, focalNorm, depthScale,
   *   centerX, centerY (card centre guess, video px) }
   */
  function start(frozen, data) {
    frame = frozen;
    info = data;
    // Expected card width, to place the lines and size the crop
    const mmPerPxAtEyes = 11.7 / data.irisPx;
    const deltaOverD = ((data.zEyes - data.zForehead) * data.depthScale) / data.focalNorm;
    const expected = CARD_WIDTH_MM / (mmPerPxAtEyes * (1 - deltaOverD));
    expectedPx = expected;
    const cw = Math.min(frozen.width, CROP_CARD_WIDTHS * expected);
    const ch = Math.min(frozen.height, (cw * h) / w);
    crop = {
      x: Math.max(0, Math.min(frozen.width - cw, data.centerX - cw / 2)),
      y: Math.max(0, Math.min(frozen.height - ch, data.centerY - ch / 2)),
      cw,
      ch,
    };
    lines = [data.centerX - expected / 2, data.centerX + expected / 2];
    found = findCardEdges(frozen, { centerX: data.centerX, centerY: data.centerY, expected });
    if (found && found.confidence >= MIN_EDGE_CONFIDENCE) lines = [found.left, found.right];
    else found = null;
    els.root.hidden = false;
    draw();
  }

  function stop() {
    els.root.hidden = true;
    frame = null;
  }

  return { resize, start, stop, get active() { return !els.root.hidden; } };
}
