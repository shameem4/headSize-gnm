/**
 * Measurement pipeline: landmarks -> iris scale -> 3D points -> mm measurements
 * @module measure
 *
 * 1. Fit a circle to each iris (video pixels) and smooth the average diameter.
 * 2. The iris is assumed to be irisDiameterMm wide, which gives the mm-per-pixel
 *    scale at the eyes and the eye-to-camera distance (pinhole model).
 * 3. Each landmark is back-projected to 3D millimetres using MediaPipe's
 *    relative depth (z, scaled by depthScale), so points in front of or behind
 *    the eyes are scaled correctly and measurements don't shrink when the head
 *    turns or tilts.
 * 4. Lengths are 3D distances; heights and angles are taken in the face's
 *    frontal plane.
 */

/** @typedef {{x: number, y: number}} Point */
/** @typedef {[number, number, number]} Vec3 */

// ============================================================================
// GEOMETRY
// ============================================================================

const EPSILON = 1e-3;

function circleFromTwoPoints(p1, p2) {
  return {
    center: { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 },
    radius: Math.hypot(p1.x - p2.x, p1.y - p2.y) / 2,
  };
}

function circleFromThreePoints(p1, p2, p3) {
  const d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
  if (Math.abs(d) < EPSILON) return null;
  const s1 = p1.x ** 2 + p1.y ** 2;
  const s2 = p2.x ** 2 + p2.y ** 2;
  const s3 = p3.x ** 2 + p3.y ** 2;
  const center = {
    x: (s1 * (p2.y - p3.y) + s2 * (p3.y - p1.y) + s3 * (p1.y - p2.y)) / d,
    y: (s1 * (p3.x - p2.x) + s2 * (p1.x - p3.x) + s3 * (p2.x - p1.x)) / d,
  };
  return { center, radius: Math.hypot(center.x - p1.x, center.y - p1.y) };
}

function isPointInsideCircle(point, circle) {
  return Math.hypot(point.x - circle.center.x, point.y - circle.center.y) <= circle.radius + EPSILON;
}

/**
 * Minimum enclosing circle (Welzl-style incremental algorithm)
 * @param {Point[]} points
 * @returns {{center: Point, radius: number}|null}
 */
function minEnclosingCircle(points) {
  let circle = null;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (circle && isPointInsideCircle(p, circle)) continue;
    circle = { center: { ...p }, radius: 0 };
    for (let j = 0; j < i; j++) {
      const q = points[j];
      if (isPointInsideCircle(q, circle)) continue;
      circle = circleFromTwoPoints(p, q);
      for (let k = 0; k < j; k++) {
        const r = points[k];
        if (isPointInsideCircle(r, circle)) continue;
        circle = circleFromThreePoints(p, q, r) || circle;
      }
    }
  }
  return circle;
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const normalize3 = (v) => {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
};

/** Unsigned angle at vertex o between a and b, in degrees (2D) */
function angleDeg(o, a, b) {
  const ux = a.x - o.x, uy = a.y - o.y, vx = b.x - o.x, vy = b.y - o.y;
  const mag = Math.hypot(ux, uy) * Math.hypot(vx, vy);
  if (!mag) return null;
  return (Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / mag))) * 180) / Math.PI;
}

// ============================================================================
// MEASURER
// ============================================================================

/**
 * Screen-space summary of a landmark row, for drawing brackets
 * @param {Point[]} points
 */
function rowDrawData(points) {
  let left = points[0];
  let right = points[0];
  for (const p of points) {
    if (p.x < left.x) left = p;
    if (p.x > right.x) right = p;
  }
  const midY = points.reduce((s, p) => s + p.y, 0) / points.length;
  return { left, right, midY };
}

/**
 * Create a stateful measurer (holds the smoothed iris diameter)
 * @param {Object} camera - CAMERA_CONFIG
 * @param {Object} lm - HEAD_CONFIG landmark indices
 */
export function createMeasurer(camera, lm) {
  let smoothedIrisPx = null;

  /**
   * Measure one frame
   * @param {Array|null} landmarks - MediaPipe normalized landmarks (already mirrored for display)
   * @param {{width: number, height: number}} videoSize - Video resolution in pixels
   * @param {{width: number, height: number}} displaySize - On-screen canvas size (for drawing points)
   * @returns {Object} measurements (all null when no face)
   */
  function update(landmarks, videoSize, displaySize) {
    const empty = { distanceCm: null, ipd: null, faceWidth: null, eyes: { left: null, right: null }, nose: null };
    if (!landmarks) {
      smoothedIrisPx = null;
      return empty;
    }

    const { width: vw, height: vh } = videoSize;
    const screen = (i) => ({ x: landmarks[i].x * displaySize.width, y: landmarks[i].y * displaySize.height });

    // 1. Iris diameter in video pixels, averaged over both eyes and smoothed
    const irisPx = (indices) =>
      minEnclosingCircle(indices.map((i) => ({ x: landmarks[i].x * vw, y: landmarks[i].y * vh })))?.radius * 2;
    const rawIrisPx = (irisPx(lm.iris.left) + irisPx(lm.iris.right)) / 2;
    if (!(rawIrisPx > 0)) return empty;
    smoothedIrisPx =
      smoothedIrisPx === null
        ? rawIrisPx
        : smoothedIrisPx + (rawIrisPx - smoothedIrisPx) * camera.irisSmoothing;

    // 2. Scale and distance at the eyes
    const mmPerPx = camera.irisDiameterMm / smoothedIrisPx;
    const focalPx = camera.focalLengthNorm * vw;
    const eyeDistMm = focalPx * mmPerPx;

    // 3. Back-project landmarks to 3D mm (camera frame). MediaPipe z is depth
    //    in the same units as x (fraction of image width), relative to the head.
    const zEyes = (landmarks[lm.pupil.left].z + landmarks[lm.pupil.right].z) / 2;
    const cache = new Map();
    const p3 = (i) => {
      let p = cache.get(i);
      if (!p) {
        const l = landmarks[i];
        const depth = eyeDistMm + camera.depthScale * (l.z - zEyes) * vw * mmPerPx;
        p = [((l.x - 0.5) * vw * depth) / focalPx, ((l.y - 0.5) * vh * depth) / focalPx, depth];
        cache.set(i, p);
      }
      return p;
    };
    const len = (a, b) => dist3(p3(a), p3(b));

    // Face frame: x across the face, y up the face (for frontal heights/angles)
    const xAxis = normalize3(sub(p3(lm.faceWidth[1]), p3(lm.faceWidth[0])));
    const up = sub(p3(lm.faceUp[1]), p3(lm.faceUp[0]));
    const yAxis = normalize3(sub(up, xAxis.map((c) => c * dot(up, xAxis))));
    const frontal = (i) => ({ x: dot(p3(i), xAxis), y: dot(p3(i), yAxis) });
    const frontalAngle = ([o, a, b]) => angleDeg(frontal(o), frontal(a), frontal(b));
    const rowCenterY = (row) => row.reduce((s, i) => s + frontal(i).y, 0) / row.length;

    // 4. Measurements
    const ipdNear = len(lm.pupil.left, lm.pupil.right);
    // Eyes converge on the camera, pulling each pupil inward by ~r·(PD/2)/D
    const ipdFar = ipdNear / (1 - camera.eyeRotationRadiusMm / eyeDistMm);

    const eye = (corners) => ({ valueMm: len(corners[0], corners[1]), points: corners.map(screen) });
    const angleLines = ([o, a, b]) => ({ origin: screen(o), lineAEnd: screen(a), lineBEnd: screen(b) });
    const { bridgeRow, padRow } = lm;

    return {
      point3D: p3,                    // [x, y, z] mm in the camera frame for any landmark index
      irisPx: smoothedIrisPx,         // smoothed iris diameter in video pixels
      distanceCm: eyeDistMm / 10,
      ipd: {
        near: ipdNear,
        far: ipdFar,
        left: screen(lm.pupil.left),
        right: screen(lm.pupil.right),
      },
      faceWidth: {
        valueMm: len(lm.faceWidth[0], lm.faceWidth[1]),
        left: screen(lm.faceWidth[0]),
        right: screen(lm.faceWidth[1]),
      },
      eyes: { left: eye(lm.eyeCorners.left), right: eye(lm.eyeCorners.right) },
      nose: {
        bridgeWidthMm: len(bridgeRow[0], bridgeRow[bridgeRow.length - 1]),
        padSpanMm: len(padRow[0], padRow[padRow.length - 1]),
        padHeightMm: Math.abs(rowCenterY(padRow) - rowCenterY(bridgeRow)),
        padAngleDeg: frontalAngle(lm.padAngle),
        padAngleLines: angleLines(lm.padAngle),
        flareAngleDeg: frontalAngle(lm.flareAngle),
        flareAngleLines: angleLines(lm.flareAngle),
        rows: {
          bridge: rowDrawData(bridgeRow.map(screen)),
          pad: rowDrawData(padRow.map(screen)),
        },
      },
    };
  }

  return { update };
}
