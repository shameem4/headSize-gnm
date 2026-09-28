/**
 * Iris scale and 3D landmarks: landmarks -> iris size -> mm scale, distance, 3D points
 * @module measure
 *
 * 1. Fit a circle to each iris (video pixels) and smooth the average diameter.
 * 2. The iris is assumed to be irisDiameterMm wide (or your calibrated size), which
 *    gives the mm-per-pixel scale at the eyes and the eye-to-camera distance (pinhole).
 * 3. Each landmark is back-projected to 3D millimetres using MediaPipe's relative
 *    depth, scaled by depthScale.
 * 4. Pupil-to-pupil distance (IPD), corrected for the eyes converging on the camera.
 */

/** @typedef {{x: number, y: number}} Point */

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

// ============================================================================
// MEASURER
// ============================================================================

/**
 * Create a stateful measurer (holds the smoothed iris diameter)
 * @param {Object} camera - CAMERA_CONFIG (irisDiameterMm, focalLengthNorm, depthScale,
 *   irisSmoothing, eyeRotationRadiusMm)
 * @param {Object} lm - HEAD_CONFIG landmark indices (iris, pupil)
 */
export function createMeasurer(camera, lm) {
  let smoothedIrisPx = null;
  const empty = { point3D: null, irisPx: null, distanceCm: null, ipd: null };

  /**
   * Measure one frame
   * @param {Array|null} landmarks - MediaPipe normalized landmarks (unmirrored)
   * @param {{width: number, height: number}} videoSize - Video resolution in pixels
   */
  function update(landmarks, videoSize) {
    if (!landmarks) {
      smoothedIrisPx = null;
      return empty;
    }
    const { width: vw, height: vh } = videoSize;

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

    // 4. IPD. Eyes converge on the camera, pulling each pupil inward by ~r·(PD/2)/D
    const [a, b] = [p3(lm.pupil.left), p3(lm.pupil.right)];
    const ipdNear = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const ipdFar = ipdNear / (1 - camera.eyeRotationRadiusMm / eyeDistMm);

    return {
      point3D: p3,                    // [x, y, z] mm in the camera frame for any landmark index
      irisPx: smoothedIrisPx,         // smoothed iris diameter in video pixels
      distanceCm: eyeDistMm / 10,
      ipd: { near: ipdNear, far: ipdFar },
    };
  }

  return { update };
}
