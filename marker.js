/**
 * Printed marker calibration: an ArUco marker (ARUCO_MIP_36h12, id 0) of known printed size,
 * found automatically with js-aruco2 (MIT, https://github.com/damianofalcioni/js-aruco2).
 * @module marker
 */

const ARUCO_BASE = "https://cdn.jsdelivr.net/npm/js-aruco2@2.0.0/src";
export const MARKER_ID = 0;
// Side of the marker's black square when printed at 100%
export const MARKER_MM = 45;
// The library's SVG has a 1-cell white margin: black square = 8 of its 10 cells
export const SVG_TO_BLACK = 8 / 10;
// Search window around the forehead, in expected marker widths
const SEARCH_WIDTHS = 3;

let loading = null;

/** Load js-aruco2 (classic scripts defining window.CV and window.AR) */
export function loadAruco() {
  const script = (src) =>
    new Promise((resolve, reject) => {
      const el = document.createElement("script");
      el.src = src;
      el.onload = resolve;
      el.onerror = () => reject(new Error(`Could not load ${src}`));
      document.head.append(el);
    });
  loading ??= script(`${ARUCO_BASE}/cv.js`).then(() => script(`${ARUCO_BASE}/aruco.js`)).then(() => window.AR);
  return loading;
}

/** SVG markup for the printable marker */
export function markerSvg(AR) {
  return new AR.Dictionary("ARUCO_MIP_36h12").generateSVG(MARKER_ID);
}

/**
 * Finds the marker near the forehead in video frames
 * @param {Object} AR - js-aruco2 namespace
 */
export function createMarkerFinder(AR) {
  const detector = new AR.Detector({ dictionaryName: "ARUCO_MIP_36h12" });
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  /**
   * @param {HTMLVideoElement} video
   * @param {{x: number, y: number}} center - search centre (video px), e.g. mid-forehead
   * @param {number} expectedPx - expected marker side (video px)
   * @returns {{corners: {x:number,y:number}[], sizePx: number}|null} corners in video px;
   *   sizePx = the larger of the mean horizontal and mean vertical side lengths (tilt
   *   about one axis only shortens the other pair)
   */
  function find(video, center, expectedPx) {
    const side = Math.round(Math.min(video.videoWidth, video.videoHeight, SEARCH_WIDTHS * expectedPx));
    const x0 = Math.round(Math.max(0, Math.min(video.videoWidth - side, center.x - side / 2)));
    const y0 = Math.round(Math.max(0, Math.min(video.videoHeight - side, center.y - side / 2)));
    if (canvas.width !== side) canvas.width = canvas.height = side;
    ctx.drawImage(video, x0, y0, side, side, 0, 0, side, side);
    const markers = detector.detect(ctx.getImageData(0, 0, side, side));
    const marker = markers.find((m) => m.id === MARKER_ID);
    if (!marker) return null;
    const c = marker.corners.map((p) => ({ x: p.x + x0, y: p.y + y0 }));
    const len = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const pairA = (len(c[0], c[1]) + len(c[2], c[3])) / 2;
    const pairB = (len(c[1], c[2]) + len(c[3], c[0])) / 2;
    return { corners: c, sizePx: Math.max(pairA, pairB) };
  }

  return { find };
}
