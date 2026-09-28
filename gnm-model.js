/**
 * GNM Head in the browser: load, fit to face landmarks, measure.
 * @module gnm-model
 *
 * Port of experiments (gnm_measure.py, experiment.py fit()). Model units are
 * metres; axes x = subject's left, y up, z out of the face. Observations use the
 * same axes, in "observation units" (metres as measured with the iris scale).
 */

// ============================================================================
// LINEAR ALGEBRA
// ============================================================================

/** Eigen-decomposition of a symmetric n×n matrix (cyclic Jacobi). */
function jacobiEigen(A, n) {
  const a = A.slice();
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = Array.from({ length: n }, (_, i) => a[i * n + i]);
  return { values, vectors: v };
}

/**
 * Rotation R (row-major 3×3) minimising |R (P - mean P) - (Q - mean Q)|, via Horn's
 * quaternion method. P, Q: Float64Array(n*3).
 */
function kabsch(P, Q) {
  const n = P.length / 3;
  const mp = [0, 0, 0], mq = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { mp[k] += P[i * 3 + k] / n; mq[k] += Q[i * 3 + k] / n; }
  const S = new Float64Array(9);                     // S[r][c] = sum a_r b_c
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      const ar = P[i * 3 + r] - mp[r];
      for (let c = 0; c < 3; c++) S[r * 3 + c] += ar * (Q[i * 3 + c] - mq[c]);
    }
  }
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
  const N = new Float64Array([
    xx + yy + zz, yz - zy, zx - xz, xy - yx,
    yz - zy, xx - yy - zz, xy + yx, zx + xz,
    zx - xz, xy + yx, -xx + yy - zz, yz + zy,
    xy - yx, zx + xz, yz + zy, -xx - yy + zz,
  ]);
  const { values, vectors } = jacobiEigen(N, 4);
  const best = values.indexOf(Math.max(...values));
  const [w, x, y, z] = [0, 1, 2, 3].map((i) => vectors[i * 4 + best]);
  return {
    R: [
      w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
    ],
    meanP: mp,
    meanQ: mq,
  };
}

/**
 * Rigid transform {R, t} with R P + t ~ Q (least squares, no scale)
 * @param {Float64Array} P - n*3
 * @param {Float64Array} Q - n*3
 */
export function rigidAlign(P, Q) {
  const { R, meanP, meanQ } = kabsch(P, Q);
  const t = [0, 1, 2].map((r) => meanQ[r] - (R[r * 3] * meanP[0] + R[r * 3 + 1] * meanP[1] + R[r * 3 + 2] * meanP[2]));
  return { R, t };
}

/**
 * Least-squares plane through some of the points of V (n*3)
 * @param {Float64Array} V
 * @param {number[]} indices
 * @returns {{centroid: number[], normal: number[]}}
 */
export function planeFit(V, indices) {
  const c = [0, 0, 0];
  for (const i of indices) for (let k = 0; k < 3; k++) c[k] += V[i * 3 + k] / indices.length;
  const C = new Float64Array(9);
  for (const i of indices) {
    const d = [0, 1, 2].map((k) => V[i * 3 + k] - c[k]);
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) C[r * 3 + k] += d[r] * d[k];
  }
  const { values, vectors } = jacobiEigen(C, 3);
  const least = values.indexOf(Math.min(...values));
  return { centroid: c, normal: [0, 1, 2].map((r) => vectors[r * 3 + least]) };
}

/** Apply {R, t} to n*3 points */
export function applyRigid({ R, t }, P) {
  const out = new Float64Array(P.length);
  for (let i = 0; i < P.length; i += 3) {
    for (let r = 0; r < 3; r++) {
      out[i + r] = R[r * 3] * P[i] + R[r * 3 + 1] * P[i + 1] + R[r * 3 + 2] * P[i + 2] + t[r];
    }
  }
  return out;
}

/** Solve symmetric positive-definite A x = b (A: n×n row-major Float64Array). */
function choleskySolve(A, b, n) {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * n + j];
      for (let k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = i === j ? Math.sqrt(sum) : sum / L[j * n + j];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let k = 0; k < i; k++) sum -= L[i * n + k] * y[k];
    y[i] = sum / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = y[i];
    for (let k = i + 1; k < n; k++) sum -= L[k * n + i] * x[k];
    x[i] = sum / L[i * n + i];
  }
  return x;
}

/** Inverse of {R, t} (R row-major) */
function invertRigid({ R, t }) {
  const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
  return { R: Rt, t: [0, 1, 2].map((r) => -(Rt[r * 3] * t[0] + Rt[r * 3 + 1] * t[1] + Rt[r * 3 + 2] * t[2])) };
}

// Pose refinement: Gauss-Newton steps, and the image-plane error (in units of focal
// length, ~5 px on a 1080p webcam) beyond which a landmark is down-weighted (Huber):
// landmarks on the hidden side of a turned face are guesses
const POSE_ITERATIONS = 6;
const POSE_HUBER = 0.004;

/**
 * Refine a model-to-camera pose {R, t} so the model points project onto the observed
 * landmark rays (perspective-n-point). Camera looks down -z, y up.
 * @param {{R: number[], t: number[]}} pose - initial model-to-camera pose
 * @param {Float64Array} Q - model points (n*3)
 * @param {Float64Array} obs - observed points (n*3); only their directions are used
 */
function refinePose({ R, t }, Q, obs) {
  const n = Q.length / 3;
  R = R.slice();
  t = t.slice();
  for (let iter = 0; iter < POSE_ITERATIONS; iter++) {
    const H = new Float64Array(36);
    const g = new Float64Array(6);
    for (let i = 0; i < n; i++) {
      const q = [Q[i * 3], Q[i * 3 + 1], Q[i * 3 + 2]];
      // rotated point (before translation) and camera-frame point
      const a = [0, 1, 2].map((r) => R[r * 3] * q[0] + R[r * 3 + 1] * q[1] + R[r * 3 + 2] * q[2]);
      const [X, Y, Z] = [a[0] + t[0], a[1] + t[1], a[2] + t[2]];
      const D = -Z;
      if (D <= 0) continue;
      const od = -obs[i * 3 + 2];
      const res = [X / D - obs[i * 3] / od, Y / D - obs[i * 3 + 1] / od];
      // d(u, v)/d(X, Y, Z); d(X, Y, Z)/d(rotation w) = -[a]x, d/d(translation) = I
      const dP = [[1 / D, 0, X / (D * D)], [0, 1 / D, Y / (D * D)]];
      const dW = [[0, a[2], -a[1]], [-a[2], 0, a[0]], [a[1], -a[0], 0]];
      const size = Math.hypot(res[0], res[1]);
      const w = size > POSE_HUBER ? POSE_HUBER / size : 1;
      for (let k = 0; k < 2; k++) {
        const J = new Array(6);
        for (let j = 0; j < 3; j++) {
          J[j] = dP[k][0] * dW[0][j] + dP[k][1] * dW[1][j] + dP[k][2] * dW[2][j];
          J[j + 3] = dP[k][j];
        }
        for (let r = 0; r < 6; r++) {
          g[r] += w * J[r] * res[k];
          for (let c = 0; c < 6; c++) H[r * 6 + c] += w * J[r] * J[c];
        }
      }
    }
    for (let r = 0; r < 6; r++) H[r * 7] *= 1 + 1e-6;
    const step = choleskySolve(H, g.map((v) => -v), 6);
    // R <- exp(w) R (Rodrigues), t <- t + dt
    const [wx, wy, wz] = step;
    const angle = Math.hypot(wx, wy, wz);
    if (angle > 1e-12) {
      const [x, y, z] = [wx / angle, wy / angle, wz / angle];
      const s = Math.sin(angle), cs = Math.cos(angle), C = 1 - cs;
      const E = [
        cs + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, cs + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, cs + z * z * C,
      ];
      R = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => E[r * 3] * R[c] + E[r * 3 + 1] * R[3 + c] + E[r * 3 + 2] * R[6 + c]));
    }
    for (let r = 0; r < 3; r++) t[r] += step[r + 3];
  }
  return { R, t };
}

// ============================================================================
// MODEL
// ============================================================================

const DTYPES = { float32: Float32Array, int8: Int8Array, uint8: Uint8Array, uint16: Uint16Array };

// Template height (metres) below which the mesh isn't drawn: the neck is narrowest
// (11.4 cm) at 0.18-0.20 and widens into the shoulders from about 0.16
const NECK_CUT_Y = 0.16;

// Per-person landmark error after averaging frames (metres); depth is noisier
const SIGMA_XY = 0.001;
const SIGMA_Z = 0.003;

/**
 * Load the trimmed GNM Head exported by experiments/export_web.py
 * @param {string|ArrayBuffer} source - URL or already-fetched buffer
 */
export async function loadGNM(source) {
  const buffer = typeof source === "string" ? await (await fetch(source)).arrayBuffer() : source;
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "GNMF") throw new Error("Not a GNMF file");
  const jsonLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, jsonLength)));
  const dataStart = 8 + jsonLength;
  const s = {};
  for (const { name, dtype, shape, offset } of header.sections) {
    const count = shape.reduce((a, b) => a * b, 1);
    s[name] = new DTYPES[dtype](buffer, dataStart + offset, count);
  }
  return new GNMModel(header, s);
}

class GNMModel {
  constructor({ K, numVertices }, s) {
    this.K = K;
    this.n = numVertices;
    this.s = s;
    // For display: head and neck, without the shoulders
    const low = (v) => s.template[v * 3 + 1] < NECK_CUT_Y;
    this.headTriangles = s.triangles.filter((_, i, tri) => {
      const t = i - (i % 3);
      return !(low(tri[t]) || low(tri[t + 1]) || low(tri[t + 2]));
    });

    // Rigid (skull-fixed) correspondence: MediaPipe landmark -> vertex, reference positions
    const rigid = [];
    for (let i = 0; i < s.corrRigid.length; i++) if (s.corrRigid[i]) rigid.push(i);
    this.rigidLandmarks = rigid.map((i) => s.corrLandmark[i]);
    this.rigidRef = Float64Array.from(rigid.flatMap((i) => [s.corrRef[i * 3], s.corrRef[i * 3 + 1], s.corrRef[i * 3 + 2]]));

    // Basis at the rigid points (full precision), as M[row * K + k], rows = 3 * rigid count
    const rows = rigid.length * 3;
    this.M = new Float64Array(rows * K);
    for (let k = 0; k < K; k++) {
      for (let r = 0; r < rows; r++) this.M[r * K + k] = s.rigidBasis[k * rows + r];
    }
    this.weights = Float64Array.from({ length: rows }, (_, i) => (i % 3 === 2 ? 1 / SIGMA_Z ** 2 : 1 / SIGMA_XY ** 2));

    // MediaPipe landmark -> vertex (all correspondences), for face measurements
    this.mpVertex = new Map();
    for (let i = 0; i < s.corrLandmark.length; i++) this.mpVertex.set(s.corrLandmark[i], s.corrVertex[i]);

    // Measurement edges: skin without ears
    const flags = s.flags;
    const edgeSet = new Set();
    const tri = s.triangles;
    for (let t = 0; t < tri.length; t += 3) {
      const [a, b, c] = [tri[t], tri[t + 1], tri[t + 2]];
      if (!(flags[a] & 1 && flags[b] & 1 && flags[c] & 1)) continue;
      for (const [u, v] of [[a, b], [b, c], [c, a]]) edgeSet.add(u < v ? u * this.n + v : v * this.n + u);
    }
    this.edges = Uint32Array.from([...edgeSet].flatMap((e) => [Math.floor(e / this.n), e % this.n]));
    const pick = (bit) => [...flags.keys()].filter((i) => flags[i] & bit);
    this.earL = pick(2);
    this.earR = pick(4);
    this.midBrow = pick(8);
    this.templeL = pick(16);
    this.templeR = pick(32);
  }

  /** Mesh vertices (Float64Array n*3, metres) for identity c (length K) */
  mesh(c) {
    const { n, K, s } = this;
    const V = Float64Array.from(s.template);
    for (let k = 0; k < K; k++) {
      const w = c[k] * s.basisScale[k];
      if (!w) continue;
      const B = s.basis.subarray(k * n * 3, (k + 1) * n * 3);
      for (let i = 0; i < n * 3; i++) V[i] += w * B[i];
    }
    return V;
  }

  /** Eyeball centres [[x,y,z] left, right] for identity c */
  eyeJoints(c) {
    const J = Array.from(this.s.eyeJoints);
    for (let k = 0; k < this.K; k++) for (let i = 0; i < 6; i++) J[i] += c[k] * this.s.eyeJointBasis[k * 6 + i];
    return [J.slice(0, 3), J.slice(3, 6)];
  }

  /** Rigid model points in observation units: s * (ref + M c) */
  rigidTarget(c, scale) {
    const { K, M, rigidRef } = this;
    const out = new Float64Array(rigidRef.length);
    for (let r = 0; r < out.length; r++) {
      let sum = rigidRef[r];
      for (let k = 0; k < K; k++) sum += M[r * K + k] * c[k];
      out[r] = scale * sum;
    }
    return out;
  }

  /**
   * Fit identity and scale to averaged rigid landmark observations.
   * Same as experiment.py fit(): Kabsch rotation, then a joint linear MAP solve for
   * identity, scale (model side) and translation.
   * @param {Float64Array} obs - rigid landmarks (3 per point), observation units
   * @param {number} scaleSd - 0 = trust iris (scale 1), Infinity = GNM decides, else prior SD
   * @returns {{c: Float64Array, scale: number}} c = identity of the real head,
   *   scale = observation units per real unit
   */
  fit(obs, scaleSd, iters = 6) {
    const { K, M, weights: W, rigidRef: ref } = this;
    const rows = ref.length;
    const P = K + 4;                                   // [c' (K), s, t (3)]
    const ps = scaleSd === 0 ? 1e12 : Number.isFinite(scaleSd) ? 1 / scaleSd ** 2 : 0;
    let c = new Float64Array(K);
    let scale = 1;
    for (let it = 0; it < iters; it++) {
      const { R } = kabsch(obs, this.rigidTarget(c, scale));
      const u = new Float64Array(rows);
      for (let i = 0; i < rows / 3; i++) {
        for (let r = 0; r < 3; r++) {
          u[i * 3 + r] = R[r * 3] * obs[i * 3] + R[r * 3 + 1] * obs[i * 3 + 1] + R[r * 3 + 2] * obs[i * 3 + 2];
        }
      }
      // A row = [M row, ref, -e_axis]; accumulate A^T W A and A^T W u
      const AtA = new Float64Array(P * P);
      const Atu = new Float64Array(P);
      const row = new Float64Array(P);
      for (let r = 0; r < rows; r++) {
        for (let k = 0; k < K; k++) row[k] = M[r * K + k];
        row[K] = ref[r];
        row[K + 1] = row[K + 2] = row[K + 3] = 0;
        row[K + 1 + (r % 3)] = -1;
        const w = W[r];
        for (let i = 0; i < P; i++) {
          const wi = w * row[i];
          if (!wi) continue;
          Atu[i] += wi * u[r];
          for (let j = i; j < P; j++) AtA[i * P + j] += wi * row[j];
        }
      }
      for (let i = 0; i < P; i++) for (let j = 0; j < i; j++) AtA[i * P + j] = AtA[j * P + i];
      for (let k = 0; k < K; k++) AtA[k * P + k] += 1;   // identity prior
      AtA[K * P + K] += ps;                               // scale prior around 1
      Atu[K] += ps;
      const theta = choleskySolve(AtA, Atu, P);
      scale = theta[K];
      c = theta.subarray(0, K).map((v) => v / scale);
    }
    return { c, scale };
  }

  /**
   * Rigid pose mapping observation-frame points to the fitted model:
   * model ~ R obs + t. Fits the model to where the landmarks appear in the image
   * (their rays from the camera), not to their estimated depth, which is the least
   * reliable part and worst when the head is turned.
   */
  pose(obs, c, scale) {
    const target = this.rigidTarget(c, scale);
    // Start: 3D alignment, with the observations rescaled about the camera (keeps their
    // image positions) to the fitted head's size, since the per-frame iris distance drifts
    const { R, meanP, meanQ } = kabsch(obs, target);
    let num = 0, den = 0;
    for (let i = 0; i < obs.length; i += 3) {
      const p = [0, 1, 2].map((k) => obs[i + k] - meanP[k]);
      for (let r = 0; r < 3; r++) {
        num += (target[i + r] - meanQ[r]) * (R[r * 3] * p[0] + R[r * 3 + 1] * p[1] + R[r * 3 + 2] * p[2]);
        den += p[r] * p[r];
      }
    }
    const start = rigidAlign(obs.map((v) => (v * num) / den), target);
    return invertRigid(refinePose(invertRigid(start), target, obs));
  }

  // ==========================================================================
  // MEASUREMENTS (mirror gnm_measure.py)
  // ==========================================================================

  /** Points where measurement edges cross V[axis] == value: array of [x, y, z] */
  cut(V, axis, value) {
    const out = [];
    const E = this.edges;
    for (let e = 0; e < E.length; e += 2) {
      const a = E[e] * 3, b = E[e + 1] * 3;
      const da = V[a + axis] - value, db = V[b + axis] - value;
      if (da * db >= 0) continue;
      const t = da / (da - db);
      out.push([V[a] + t * (V[b] - V[a]), V[a + 1] + t * (V[b + 1] - V[a + 1]), V[a + 2] + t * (V[b + 2] - V[a + 2])]);
    }
    return out;
  }

  /**
   * Full measurement set (mm)
   * @param {Float64Array} V - mesh from mesh(c)
   * @param {number[][]} eyes - eyeJoints(c)
   * @returns {{values: Object<string, number>}}
   */
  measure(V, eyes) {
    const P = (i) => [V[i * 3], V[i * 3 + 1], V[i * 3 + 2]];
    const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const mp = (i) => P(this.mpVertex.get(i));
    const values = {};

    // Face
    values.ipd_far = dist(eyes[0], eyes[1]);
    values.face_width = dist(mp(127), mp(356));
    values.eye_width = (dist(mp(362), mp(263)) + dist(mp(33), mp(133))) / 2;
    values.bridge_width = dist(mp(190), mp(414));
    values.pad_width = dist(mp(114), mp(343));
    // Nose bridge (experiments/nose_check.py): sellion (168) in front of the inner eye
    // corners (133, 362), and its height relative to the eyeball centres
    const sellion = mp(168), canthi = [0, 1, 2].map((k) => (mp(133)[k] + mp(362)[k]) / 2);
    values.bridge_projection = sellion[2] - canthi[2];
    values.bridge_height = sellion[1] - (eyes[0][1] + eyes[1][1]) / 2;

    // Head circumference and length: tape band above the brows
    const y0 = mean(this.midBrow.map((i) => V[i * 3 + 1]));
    let circ = 0, length = 0;
    for (let h = 0; h <= 0.0605; h += 0.003) {
      const pts = this.cut(V, 1, y0 + h);
      if (pts.length < 3) continue;
      circ = Math.max(circ, perimeter(convexHull(pts.map((p) => [p[0], p[2]]))));
      const zs = pts.map((p) => p[2]);
      length = Math.max(length, Math.max(...zs) - Math.min(...zs));
    }
    values.circumference = circ;
    values.length = length;

    // Breadth: widest point above the ears
    let breadth = 0;
    for (let h = -0.02; h <= 0.0805; h += 0.004) {
      const xs = this.cut(V, 1, y0 + h).map((p) => p[0]);
      if (xs.length) breadth = Math.max(breadth, Math.max(...xs) - Math.min(...xs));
    }
    values.breadth = breadth;

    // Tragion and the ear-to-ear arc over the head
    const tl = this.tragion(V, this.earL), tr = this.tragion(V, this.earR);
    values.bitragion = dist(tl, tr);
    const zc = (tl[2] + tr[2]) / 2;
    const ylow = Math.min(tl[1], tr[1]);
    const arcPts = this.cut(V, 2, zc).filter((p) => p[1] > ylow).map((p) => [p[0], p[1]]);
    const arcHull = convexHull([...arcPts, [tl[0], tl[1]], [tr[0], tr[1]]]);
    values.ear_to_ear_arc = perimeter(arcHull) - Math.hypot(tl[0] - tr[0], tl[1] - tr[1]);

    // Glasses: temple width, eye (corneal apex) to top of ear
    values.temple_width = V[argmax(this.templeL, (i) => V[i * 3]) * 3] - V[argmax(this.templeR, (i) => -V[i * 3]) * 3];
    const [apexL, apexR] = this.s.eyeApex;
    const earTopL = P(argmax(this.earL, (i) => V[i * 3 + 1]));
    const earTopR = P(argmax(this.earR, (i) => V[i * 3 + 1]));
    values.eye_to_ear = (dist(P(apexL), earTopL) + dist(P(apexR), earTopR)) / 2;

    // Earbuds (coarse, from face shape): ear length
    const span = (idx) => {
      const ys = idx.map((i) => V[i * 3 + 1]);
      return Math.max(...ys) - Math.min(...ys);
    };
    values.ear_length = (span(this.earL) + span(this.earR)) / 2;

    for (const k of Object.keys(values)) values[k] *= 1000;
    return { values };
  }

  /** Front of the ear at mid-ear height (coarse tragion) */
  tragion(V, ear) {
    const midY = mean(ear.map((i) => V[i * 3 + 1]));
    const band = ear.filter((i) => Math.abs(V[i * 3 + 1] - midY) < 0.006);
    const i = argmax(band, (j) => V[j * 3 + 2]);
    return [V[i * 3], V[i * 3 + 1], V[i * 3 + 2]];
  }
}

// ============================================================================
// 2D HELPERS
// ============================================================================

const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

function argmax(items, key) {
  let best = items[0], bestValue = -Infinity;
  for (const item of items) {
    const v = key(item);
    if (v > bestValue) { bestValue = v; best = item; }
  }
  return best;
}

/** Convex hull (Andrew's monotone chain), counter-clockwise, no repeated end point */
function convexHull(points) {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function perimeter(hull) {
  let sum = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    sum += Math.hypot(a[0] - b[0], a[1] - b[1]);
  }
  return sum;
}
