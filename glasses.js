/**
 * Glasses detection (heuristic): the bridge of a pair of glasses draws a horizontal
 * line across the smooth skin between the eyes.
 * @module glasses
 *
 * A small patch between the eyes (brow to nose bridge, rotated with the eye line) is
 * sampled from the frame. For each row, the vertical intensity gradient is averaged
 * across the patch: a line spanning the patch gives a large, same-signed average,
 * while skin texture and the vertical edges of the nose cancel out ("bridge").
 * The same test on columns catches the inner lens rims of thin frames ("rim").
 *
 * Thresholds: on six generated faces, thick and thin-metal frames scored >= 0.63 and
 * faces without glasses <= 0.28. On a real webcam user, bare-face bridge scores reached
 * 0.59 (shadow under a heavy brow ridge) while their glasses scored 1.25, so the bridge
 * threshold is 0.8. Thin metal frames are caught by the rim score instead.
 * Rimless glasses scored like bare skin and are not detected.
 */

export const BRIDGE_THRESHOLD = 0.8;
export const RIM_THRESHOLD = 0.45;

/** Decide from recent scores: median of either cue above the threshold */
export function looksLikeGlasses(scores) {
  if (!scores.length) return false;
  const median = (values) => values.slice().sort((a, b) => a - b)[values.length >> 1];
  return (
    median(scores.map((s) => s.bridge)) >= BRIDGE_THRESHOLD ||
    median(scores.map((s) => s.rim)) >= RIM_THRESHOLD
  );
}

const PATCH_W = 48;
const PATCH_H = 32;

/** MediaPipe landmarks: inner eye corners, between the brows, nose bridge */
const INNER_RIGHT = 133;
const INNER_LEFT = 362;
const BROW_MID = 9;
const NOSE_BRIDGE = 6;

export function createGlassesDetector() {
  const canvas = document.createElement("canvas");
  canvas.width = PATCH_W;
  canvas.height = PATCH_H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  /**
   * Line scores for one frame (higher = more likely glasses)
   * @param {CanvasImageSource} source - video frame or image
   * @param {Array<{x:number,y:number}>} landmarks - MediaPipe normalized landmarks
   * @param {number} width - source width in pixels
   * @param {number} height - source height in pixels
   * @returns {{bridge: number, rim: number}}
   */
  function score(source, landmarks, width, height) {
    const P = (i) => [landmarks[i].x * width, landmarks[i].y * height];
    const [ax, ay] = P(INNER_RIGHT), [bx, by] = P(INNER_LEFT);
    const [tx, ty] = P(BROW_MID), [nx, ny] = P(NOSE_BRIDGE);

    // Patch frame: ex along the eye line, ey perpendicular (down the face)
    const span = Math.hypot(bx - ax, by - ay);
    if (!span) return { bridge: 0, rim: 0 };
    const ex = [(bx - ax) / span, (by - ay) / span];
    const ey = [-ex[1], ex[0]];
    const cx = (tx + nx) / 2, cy = (ty + ny) / 2;
    const patchW = 0.7 * span;
    const patchH = Math.max(1, Math.abs((nx - tx) * ey[0] + (ny - ty) * ey[1]));
    const sx = patchW / PATCH_W, sy = patchH / PATCH_H;

    // canvas = S^-1 R^T (image - C) + patch centre
    const a = ex[0] / sx, c = ex[1] / sx, b = ey[0] / sy, d = ey[1] / sy;
    ctx.setTransform(a, b, c, d, PATCH_W / 2 - (a * cx + c * cy), PATCH_H / 2 - (b * cx + d * cy));
    ctx.drawImage(source, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    const px = ctx.getImageData(0, 0, PATCH_W, PATCH_H).data;
    const g = new Float32Array(PATCH_W * PATCH_H);
    let sum = 0;
    for (let i = 0; i < g.length; i++) {
      g[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
      sum += g[i];
    }
    const mean = sum / g.length;
    let variance = 0;
    for (const v of g) variance += (v - mean) ** 2;
    const std = Math.sqrt(variance / g.length);

    // Strongest row of same-signed vertical gradient, relative to patch contrast
    let best = 0;
    for (let r = 1; r < PATCH_H - 1; r++) {
      let rowSum = 0;
      for (let col = 0; col < PATCH_W; col++) {
        rowSum += g[(r + 1) * PATCH_W + col] - g[(r - 1) * PATCH_W + col];
      }
      best = Math.max(best, Math.abs(rowSum / PATCH_W));
    }
    // Same test for vertical lines (inner lens rims at the patch sides)
    let bestCol = 0;
    for (let col = 1; col < PATCH_W - 1; col++) {
      let colSum = 0;
      for (let r = 0; r < PATCH_H; r++) colSum += g[r * PATCH_W + col + 1] - g[r * PATCH_W + col - 1];
      bestCol = Math.max(bestCol, Math.abs(colSum / PATCH_H));
    }
    return { bridge: best / (std + 8), rim: bestCol / (std + 8) };
  }

  return { score, canvas };
}
