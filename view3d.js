/**
 * Three.js view of the fitted GNM head over the video
 * @module view3d
 *
 * The scene is the camera frame: Three's axes (x right, y up, looking down -z) equal the
 * observation axes (subject's left, up, out of the face) when the subject faces the
 * camera, so a perspective camera with the webcam's field of view lines everything up.
 */

import * as THREE from "three";

// Contour spacing: horizontal rings (model units, metres) and meridians (degrees)
const RING_STEP = 0.012;
const MERIDIAN_STEP_DEG = 15;

/**
 * Line segments where the mesh crosses a set of planes: horizontal planes every
 * RING_STEP, and vertical planes through the head's vertical axis every MERIDIAN_STEP_DEG
 * @returns {Float32Array} segment end points (6 numbers per segment)
 */
function contours(V, triangles) {
  const n = V.length / 3;
  let cx = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += V[i * 3] / n; cz += V[i * 3 + 2] / n; }
  // Each plane: signed distance of vertex i, as a function
  const planes = [];
  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) { minY = Math.min(minY, V[i * 3 + 1]); maxY = Math.max(maxY, V[i * 3 + 1]); }
  for (let y = Math.ceil(minY / RING_STEP) * RING_STEP; y < maxY; y += RING_STEP) {
    planes.push((i) => V[i * 3 + 1] - y);
  }
  for (let deg = 0; deg < 180; deg += MERIDIAN_STEP_DEG) {
    const a = (deg * Math.PI) / 180, nx = Math.cos(a), nz = Math.sin(a);
    planes.push((i) => nx * (V[i * 3] - cx) + nz * (V[i * 3 + 2] - cz));
  }
  const out = [];
  const lerp = (a, b, da, db) => {
    const t = da / (da - db);
    return [0, 1, 2].map((k) => V[a * 3 + k] + t * (V[b * 3 + k] - V[a * 3 + k]));
  };
  for (const dist of planes) {
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) d[i] = dist(i);
    for (let t = 0; t < triangles.length; t += 3) {
      const ids = [triangles[t], triangles[t + 1], triangles[t + 2]];
      const hits = [];
      for (let e = 0; e < 3; e++) {
        const a = ids[e], b = ids[(e + 1) % 3];
        if ((d[a] < 0) !== (d[b] < 0)) hits.push(lerp(a, b, d[a], d[b]));
      }
      if (hits.length === 2) out.push(...hits[0], ...hits[1]);
    }
  }
  return Float32Array.from(out);
}

export function createView(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.01, 10);

  // Head placed by a rigid transform each frame
  const head = new THREE.Group();
  head.matrixAutoUpdate = false;
  scene.add(head);

  // Coarse wireframe of the front surface only: contour lines (horizontal rings and
  // vertical meridians) cut from the mesh; an invisible depth pass hides the far side
  const geometry = new THREE.BufferGeometry();
  const wireGeometry = new THREE.BufferGeometry();
  const mesh = new THREE.Group();
  const occluder = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide }));
  const wireframe = new THREE.LineSegments(
    wireGeometry,
    new THREE.LineBasicMaterial({ color: 0x88ccff, transparent: true, opacity: 0.7 })
  );
  // Draw the lines slightly in front of the depth pass so they aren't hidden by it
  occluder.material.polygonOffset = true;
  occluder.material.polygonOffsetFactor = 1;
  occluder.material.polygonOffsetUnits = 1;
  wireframe.renderOrder = 1;
  mesh.add(occluder, wireframe);
  head.add(mesh);
  head.visible = false;

  return {
    /** Match the webcam: vertical FOV from focal length, and the canvas display size */
    setCamera(videoWidth, videoHeight, focalPx, displayWidth, displayHeight) {
      camera.fov = (2 * Math.atan(videoHeight / 2 / focalPx) * 180) / Math.PI;
      camera.aspect = videoWidth / videoHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(displayWidth, displayHeight);
    },

    /**
     * Replace the head shape (model coordinates, observation units)
     * @param {Float64Array} vertices - n*3
     * @param {Uint16Array} triangles
     */
    setHead(vertices, triangles) {
      geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(vertices), 3));
      geometry.setIndex(new THREE.BufferAttribute(triangles, 1));
      geometry.computeBoundingSphere();
      wireGeometry.setAttribute("position", new THREE.BufferAttribute(contours(vertices, triangles), 3));
      wireGeometry.computeBoundingSphere();
    },

    /** Model -> camera frame: obs = R^T (model - t) */
    setPose(R, t) {
      const Rt = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]];
      const tx = -(Rt[0] * t[0] + Rt[1] * t[1] + Rt[2] * t[2]);
      const ty = -(Rt[3] * t[0] + Rt[4] * t[1] + Rt[5] * t[2]);
      const tz = -(Rt[6] * t[0] + Rt[7] * t[1] + Rt[8] * t[2]);
      head.matrix.set(Rt[0], Rt[1], Rt[2], tx, Rt[3], Rt[4], Rt[5], ty, Rt[6], Rt[7], Rt[8], tz, 0, 0, 0, 1);
      head.matrixWorldNeedsUpdate = true;
      head.visible = true;
    },

    hide() {
      head.visible = false;
    },

    setMeshVisible(visible) {
      mesh.visible = visible;
    },

    render() {
      renderer.render(scene, camera);
    },
  };
}
