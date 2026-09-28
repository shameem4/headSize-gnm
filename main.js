/**
 * GNM head demo: MediaPipe tracks the face, GNM Head is fitted to it, and all
 * measurements are taken on the fitted head (including parts the camera can't see).
 * @module main
 */

import { CAMERA_CONFIG, HEAD_CONFIG, UI_CONFIG } from "./config.js";
import { createMeasurer } from "./measure.js";
import { CameraManager, ModelManager } from "./camera.js";
import { loadGNM, rigidAlign, applyRigid, planeFit } from "./gnm-model.js";
import { loadEarPipeline, detectEars, measureEar, markerCheck, projectPoint, EAR_LINES } from "./ears.js";
import { createView } from "./view3d.js";
import { createGuide } from "./guide.js";
import { createGlassesDetector, looksLikeGlasses } from "./glasses.js";
import { composeImage, download, rawFrame, snapshotName, zipFiles } from "./snapshot.js";
import { createCardCalibration, irisFromCard, IRIS_MIN_MM, IRIS_MAX_MM } from "./calibrate.js";
import { loadAruco, createMarkerFinder, MARKER_MM } from "./marker.js";

// Frames averaged into one fit
const CAPTURE_FRAMES = 90;
// Nose movement per frame (fraction of face height) above which the head isn't still
const MAX_MOTION = 0.015;
// Glasses check: score every Nth frame, decide on the median of the last few scores
const GLASSES_EVERY = 3;
const GLASSES_WINDOW = 15;

// Card calibration: still frames averaged before freezing, and assumed calibration error
const CALIBRATION_FRAMES = 30;
const CALIBRATION_SD = 0.015;
const IRIS_STORAGE_KEY = "headsize.irisMm";
// After marker calibration, frames without the marker before measuring starts (~1 s)
const LOWER_FRAMES = 30;
// Where the card rests: mid-forehead landmark
const FOREHEAD_MID = 151;

// Ear measurement: frames per ear, and the head turn (degrees, from the fitted head's
// pose) where the ear is seen well and the face is still tracked (step 1 test: ear points
// fit well at ~50-60 degrees; face tracking is lost beyond ~70)
const EAR_FRAMES = 45;
const EAR_TURN_MIN = 40;
const EAR_TURN_MAX = 70;
// Frames where the camera sees the ear plane too obliquely are skipped
const EAR_MIN_VIEW_COS = 0.4;
// Iris-scale size error (experiments/results.txt, scenario A)
const IRIS_SCALE_SD = 0.041;

const MESSAGES = {
  marker: "Hold the printed marker flat against your forehead, facing the camera.",
  markerFound: "Marker found: keep still…",
  markerLower: "Calibrated. Now put the marker down and look at the camera.",
  markerLoadFailed: "Could not load the marker detector. Check your connection, then try again.",
  card: "Hold a card flat against your forehead, long side level, by its top edge " +
    "(keep fingers off its left and right sides), then keep still. " +
    "Any bank-card-sized card works (gift, loyalty, hotel key), or the back of a bank card.",
  align: "Fit your head in the frame, 0.5 to 1 m from the camera",
  center: "Center your head in the frame",
  closer: "Move closer (within 1 m)",
  back: "Move back (at least 0.5 m away)",
  turn: "Look straight at the camera",
  still: "Keep still…",
  measuring: "Keep still while measuring…",
  lost: "Face not found: fit your head in the frame, 0.5 to 1 m away",
  earsFirst: "Measure your head first, then your ears",
  earsLoading: "Loading the ear detector…",
  earsLoadFailed: "Could not load the ear detector. Check your connection, then try again.",
  earsTurn: "Turn your head slowly to one side, about 50°, and hold. Keep hair off your ears. " +
    "(Optional scale check: hold the printed marker flat just behind the ear, not on the cheek: " +
    "the face must stay visible.)",
  earsOther: "Now turn to the other side, about 60°, and hold.",
  earsMore: "Turn a little further…",
  earsBack: "Turn back a little (your face must stay in view)",
  earsFound: "Ear found: hold still…",
  earsNotFound: "Can't see your ear: keep hair off it and hold still",
  done: "Done",
};

// Typical error with iris scale (nominal scenario, experiments/results.txt).
// rmseShape: error with a perfect scale (scenario B), used after card calibration together
// with CALIBRATION_SD for the calibration's own uncertainty.
// Size always comes from the iris; GNM supplies the shape. GNM's own size estimate
// (from face proportions) was ~5% off on the first real test, so it isn't used.
const METRICS = [
  { key: "circumference", label: "Circumference", group: "Head", rmse: 22.7, rmseShape: 8.1,
    desc: "Around the head just above the brows, like a tape measure for hat size. Hair not included." },
  { key: "length", label: "Length", group: "Head", rmse: 7.8, rmseShape: 3.4,
    desc: "Front to back of the head at the hat band (brow to the back of the head)." },
  { key: "breadth", label: "Breadth", group: "Head", rmse: 6.6, rmseShape: 2.7,
    desc: "Widest side-to-side distance of the head, above the ears." },
  { key: "temple_width", label: "Temple width", group: "Glasses", rmse: 6.1, rmseShape: 1.4,
    desc: "Width of the head at the temples, where a frame's arms pass: guides the frame's overall width." },
  { key: "eye_to_ear", label: "Eye to ear", group: "Glasses", rmse: 6.1, rmseShape: 4.5,
    desc: "Front of the eye to the top of the ear on the same side: guides the length of the frame's arms." },
  { key: "ipd_far", label: "IPD (far)", group: "Glasses", rmse: 2.4, rmseShape: 0.5,
    desc: "Pupillary distance for distance vision: pupil centre to pupil centre, measured directly and corrected for the eyes turning in to look at the camera." },
  { key: "bridge_width", label: "Bridge width", group: "Glasses", rmse: 1.5, rmseShape: 0.5,
    desc: "Width of the nose at eye level, between the inner eye corners: where a frame's bridge sits." },
  { key: "pad_width", label: "Pad width", group: "Glasses", rmse: 1.3, rmseShape: 0.5,
    desc: "Width of the nose a little lower, on its sides at about lower-eyelid level: where a frame's nose pads rest." },
  // experiments/nose_check.py
  { key: "bridge_projection", label: "Bridge projection", group: "Glasses", rmse: 0.9, rmseShape: 0.9,
    desc: "How far the top of the nose (between the eyes) stands in front of the inner eye corners. Low values mean a low nose bridge: frames tend to slide down or rest on the cheeks." },
  { key: "bridge_height", label: "Bridge height", group: "Glasses", rmse: 0.5, rmseShape: 0.5,
    desc: "Height of the top of the nose relative to the pupils (negative = below them). A lower-set bridge also points to a low-bridge fit." },
  { key: "bitragion", label: "Ear to ear (straight)", group: "Headphones", rmse: 6.2, rmseShape: 1.8,
    desc: "Straight line between the fronts of the ears: how far apart headphone cups sit." },
  { key: "ear_to_ear_arc", label: "Ear to ear (over head)", group: "Headphones", rmse: 13.0, rmseShape: 6.4,
    desc: "From ear to ear over the top of the head: the headband length needed." },
  { key: "ear_length", label: "Ear length (from face)", group: "Earbuds", rmse: 3.9, rmseShape: 3.8,
    desc: "Ear length predicted from face shape alone (weak): top of the ear to the bottom of the lobe. \"Measure ears\" measures it instead." },
  // Measured from side views ("Measure ears"); error = frame spread + scale error
  { key: "ear_length_measured", label: "Ear length (measured)", group: "Earbuds", measured: true,
    desc: "From the side views: top of the ear's rim to the bottom of the lobe, along the ear's long axis." },
  { key: "ear_width", label: "Ear width", group: "Earbuds", measured: true,
    desc: "From the side views: front to back of the ear, across its long axis." },
  { key: "concha_height", label: "Concha height", group: "Earbuds", measured: true,
    desc: "Height of the bowl in front of the ear canal, where an earbud sits." },
  { key: "concha_width", label: "Concha width", group: "Earbuds", measured: true,
    desc: "Width of the bowl in front of the ear canal, across the ear's axis." },
  { key: "tragus_gap", label: "Tragus to antitragus", group: "Earbuds", measured: true,
    desc: "Gap between the small flap in front of the ear canal (tragus) and the bump below it (antitragus): the notch an earbud's stem passes through." },
  { key: "face_width", label: "Face width", group: "Head", rmse: 5.9, rmseShape: 0.7,
    desc: "Width of the face at eye level, from one side of the face outline to the other." },
  { key: "eye_width", label: "Eye width", group: "Head", rmse: 0.9, rmseShape: 0.3,
    desc: "Average width of each eye, corner to corner." },
];

// Nose bridge category from bridge projection: thirds of the GNM Head population
// (experiments/nose_check.py). Relative to that population, not an industry standard.
const BRIDGE_LOW_MM = 11.0;
const BRIDGE_HIGH_MM = 14.6;
/** Metrics to show: the face-based ear length is dropped once ears are measured */
function shownMetrics(values) {
  return values?.ear_length_measured != null ? METRICS.filter((m) => m.key !== "ear_length") : METRICS;
}

function noseBridge(projection) {
  return projection < BRIDGE_LOW_MM ? "low" : projection > BRIDGE_HIGH_MM ? "high" : "medium";
}
// Panel and snapshot order
const GROUPS = ["Glasses", "Earbuds", "Headphones", "Head"];

// DOM
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const canvas = document.getElementById("gnm_canvas");
const guideCanvas = document.getElementById("guide_canvas");
const guideMsg = document.getElementById("guide_msg");
const guideText = document.getElementById("guide_text");
const glassesStatus = document.getElementById("glasses_status");
const statusEl = document.getElementById("status");
const fitStatus = document.getElementById("fit_status");
const progressBar = document.getElementById("progress_bar");
const panelBody = document.getElementById("metrics_panel_body");

// Raw MediaPipe depth (depthScale 1), not the corrected depth the face demo uses: the fit
// compares against a reference cloud built from raw MediaPipe landmarks, and subtracting
// it is what cancels MediaPipe's too-deep faces. Halving depth here breaks that match
// (head length came out ~30% short).
let irisMm = loadIris();
let measurer = makeMeasurer();
// Ask for 1080p: at 1 m a 720p webcam sees the iris only ~12 px wide (the browser falls
// back to what the camera supports). Landmarks stay unmirrored; CSS mirrors the display.
const camera = new CameraManager(video, { ...CAMERA_CONFIG, videoSize: { width: 1920, height: 1080 } });
const view = createView(canvas);
const guide = createGuide(guideCanvas, {
  focalNorm: CAMERA_CONFIG.focalLengthNorm,
  mirrored: UI_CONFIG.mirrorEnabled,
});
const glassesDetector = createGlassesDetector();
const calibration = createCardCalibration(
  {
    root: document.getElementById("calib"),
    canvas: document.getElementById("calib_canvas"),
    text: document.getElementById("calib_text"),
    confirm: document.getElementById("calib_confirm"),
    retake: document.getElementById("calib_retake"),
    cancel: document.getElementById("calib_cancel"),
  },
  {
    mirrored: UI_CONFIG.mirrorEnabled,
    onConfirm: (iris) => {
      // The marking view as confirmed (pixelated except at the card's edges)
      calibrationRecord = captionedCopy(document.getElementById("calib_canvas"), `Card calibration: iris ${iris.toFixed(2)} mm`, iris, "card");
      applyCalibration(iris);
    },
    onRetake: startCalibration,
    onCancel: resetCapture,
  }
);

let models;
let gnm;
let capture;        // { reference, sum, count }
let fitResult = null;
let phase = "align"; // "align" -> "capture" -> "done"
let glassesScores = [];
let lastNose = null;
let frameNumber = 0;
// Latest values, for snapshots
let lastValues = null;
let lastDistanceCm = null;
let lastGlasses = false;
let snapshotRequested = false;
// Images kept from the moment each step happened, for "Save snapshot" (by the time it's
// clicked the live view shows something else)
let fitRecord = null;          // { view, raw, time } right after the head fit
let fitCaptureRequested = false;
let calibrationRecord = null;  // { image, irisMm, kind, time } from this session's calibration
let calibrationFrames = [];
let markerFinder = null;
let markerQuad = null; // last detected marker corners (normalized), for drawing
let lowerFrames = 0;      // frames without the marker since calibration (phase "lower")
let fitVertices = null;   // fitted head, observation units (for the ear planes)
let earPipeline = null;
let ears = null;          // { left: [], right: [], busy, points } during/after "ears"
const earCrop = document.createElement("canvas");

/** Calibrated iris diameter from this browser, or null */
function loadIris() {
  try {
    const v = parseFloat(localStorage.getItem(IRIS_STORAGE_KEY));
    return Number.isFinite(v) ? v : null;
  } catch {
    return null;
  }
}

function saveIris(value) {
  try {
    if (value == null) localStorage.removeItem(IRIS_STORAGE_KEY);
    else localStorage.setItem(IRIS_STORAGE_KEY, String(value));
  } catch {
    // storage unavailable: calibration lasts for this page only
  }
}

function makeMeasurer() {
  return createMeasurer(
    { ...CAMERA_CONFIG, depthScale: 1, irisDiameterMm: irisMm ?? CAMERA_CONFIG.irisDiameterMm },
    HEAD_CONFIG
  );
}

// ============================================================================
// CAPTURE AND FIT
// ============================================================================

function resetCapture() {
  // ipdSum: direct pupil-to-pupil far PD (iris scale, no GNM), summed over captured frames
  capture = { reference: null, sum: null, count: 0, ipdSum: 0 };
  fitResult = null;
  fitRecord = null;
  ears = null;
  phase = "align";
  glassesScores = [];
  calibrationFrames = [];
  if (calibration.active) calibration.stop();
  view.hide();
  renderPanel(null);
  setProgress(0, MESSAGES.align);
}

/** Store a calibrated iris size and re-measure with it */
function applyCalibration(value) {
  irisMm = value;
  saveIris(value);
  measurer = makeMeasurer();
  updateScaleStatus();
  resetCapture();
}

async function startMarkerCalibration() {
  try {
    markerFinder ??= createMarkerFinder(await loadAruco());
  } catch (error) {
    console.error(error);
    showMessage(MESSAGES.markerLoadFailed, { warn: true });
    return;
  }
  phase = "marker";
  calibrationFrames = [];
  markerQuad = null;
  view.hide();
  setProgress(0, MESSAGES.marker);
  showMessage(MESSAGES.marker);
}

/**
 * Find the printed marker on the forehead each frame; after enough detections, the
 * median iris size (card maths with the marker's printed size) is stored
 */
/** Look for the printed marker around the forehead */
function findMarker(landmarks, m) {
  const W = video.videoWidth, H = video.videoHeight;
  const zEyes = (landmarks[468].z + landmarks[473].z) / 2;
  const zForehead = landmarks[FOREHEAD_MID].z;
  const deltaOverD = ((zEyes - zForehead) * CAMERA_CONFIG.depthScale) / CAMERA_CONFIG.focalLengthNorm;
  const expectedPx = (MARKER_MM * m.irisPx) / ((irisMm ?? CAMERA_CONFIG.irisDiameterMm) * (1 - deltaOverD));
  const center = { x: landmarks[FOREHEAD_MID].x * W, y: landmarks[FOREHEAD_MID].y * H };
  return { found: markerFinder.find(video, center, expectedPx), zEyes, zForehead };
}

/** After marker calibration: wait until the marker is put down before measuring again
 * (the hand and marker cover the forehead) */
function lowerStep(landmarks, m) {
  lowerFrames = landmarks && findMarker(landmarks, m).found ? 0 : lowerFrames + 1;
  showMessage(MESSAGES.markerLower);
  if (lowerFrames >= LOWER_FRAMES) {
    phase = "align";
    showMessage(MESSAGES.align);
  }
}

function markerStep(landmarks, m, still) {
  const W = video.videoWidth, H = video.videoHeight;
  const { found, zEyes, zForehead } = findMarker(landmarks, m);
  markerQuad = found ? found.corners.map((p) => ({ x: p.x / W, y: p.y / H })) : null;
  if (!found) return showMessage(MESSAGES.marker);
  if (!still) return showMessage(MESSAGES.still, { warn: true });

  const objectMm = MARKER_MM;   // printed at 100% (the marker page's ruler checks that)
  calibrationFrames.push(
    irisFromCard({
      cardPx: found.sizePx,
      irisPx: m.irisPx,
      zEyes,
      zForehead,
      focalNorm: CAMERA_CONFIG.focalLengthNorm,
      depthScale: CAMERA_CONFIG.depthScale,
      objectMm,
    })
  );
  showMessage(MESSAGES.markerFound);
  setProgress(calibrationFrames.length / CALIBRATION_FRAMES, MESSAGES.markerFound);
  if (calibrationFrames.length < CALIBRATION_FRAMES) return;

  const iris = calibrationFrames.slice().sort((a, b) => a - b)[CALIBRATION_FRAMES >> 1];
  calibrationFrames = [];
  markerQuad = null;
  const markerFrame = grabFrame(false);
  const mctx = markerFrame.getContext("2d");
  mctx.strokeStyle = "#00ffc8";
  mctx.lineWidth = 4;
  mctx.beginPath();
  found.corners.forEach((p, i) => (i ? mctx.lineTo(p.x, p.y) : mctx.moveTo(p.x, p.y)));
  mctx.closePath();
  mctx.stroke();
  if (!(iris >= IRIS_MIN_MM && iris <= IRIS_MAX_MM)) {
    showMessage(
      `That gives an iris of ${iris.toFixed(2)} mm, outside the human range. ` +
        "Check the marker was printed at 100% (\"Actual size\"), then try again.",
      { warn: true }
    );
    return;
  }
  calibrationRecord = captionedCopy(markerFrame, `Printed marker calibration: iris ${iris.toFixed(2)} mm`, iris, "marker");
  applyCalibration(iris);
  phase = "lower";
  lowerFrames = 0;
}

function updateScaleStatus() {
  document.getElementById("scale_status").textContent =
    irisMm == null ? "average iris (11.7 mm)" : `your calibration (iris ${irisMm.toFixed(2)} mm)`;
  document.getElementById("calib_reset").hidden = irisMm == null;
}

function startCalibration() {
  phase = "calibrate";
  calibrationFrames = [];
  view.hide();
  setProgress(0, MESSAGES.card);
  showMessage(MESSAGES.card);
}

/** Collect still frames with the card held up, then freeze one for marking */
function calibrationStep(landmarks, m, still) {
  if (!still) return showMessage(MESSAGES.still, { warn: true });
  showMessage(MESSAGES.card);
  calibrationFrames.push({
    irisPx: m.irisPx,
    zEyes: (landmarks[468].z + landmarks[473].z) / 2,
    zForehead: landmarks[FOREHEAD_MID].z,
    centerX: landmarks[FOREHEAD_MID].x * video.videoWidth,
    centerY: landmarks[FOREHEAD_MID].y * video.videoHeight,
  });
  setProgress(calibrationFrames.length / CALIBRATION_FRAMES, MESSAGES.card);
  if (calibrationFrames.length < CALIBRATION_FRAMES) return;

  const median = (key) => calibrationFrames.map((f) => f[key]).sort((a, b) => a - b)[CALIBRATION_FRAMES >> 1];
  const frozen = document.createElement("canvas");
  frozen.width = video.videoWidth;
  frozen.height = video.videoHeight;
  frozen.getContext("2d").drawImage(video, 0, 0);
  phase = "mark";
  showMessage(null);
  calibration.start(frozen, {
    irisPx: median("irisPx"),
    zEyes: median("zEyes"),
    zForehead: median("zForehead"),
    centerX: median("centerX"),
    centerY: median("centerY"),
    focalNorm: CAMERA_CONFIG.focalLengthNorm,
    depthScale: CAMERA_CONFIG.depthScale,
  });
  calibrationFrames = [];
}

/** Show a guidance message over the video (warn = orange, for things to fix) */
function showMessage(text, { warn = false } = {}) {
  guideMsg.hidden = !text;
  guideText.textContent = text || "";
  guideMsg.classList.toggle("warn", warn);
}

/** Rigid landmarks as model-axis metres (x = subject's left, y up, z towards camera) */
function observe(point3D) {
  const out = new Float64Array(gnm.rigidLandmarks.length * 3);
  gnm.rigidLandmarks.forEach((index, i) => {
    const [x, y, z] = point3D(index);
    out[i * 3] = x / 1000;
    out[i * 3 + 1] = -y / 1000;
    out[i * 3 + 2] = -z / 1000;
  });
  return out;
}

/** Add a frame to the running average (aligned to the first frame) */
function accumulate(obs) {
  if (!capture.reference) {
    capture.reference = obs;
    capture.sum = Float64Array.from(obs);
  } else {
    const aligned = applyRigid(rigidAlign(obs, capture.reference), obs);
    for (let i = 0; i < aligned.length; i++) capture.sum[i] += aligned[i];
  }
  capture.count++;
}

function refit() {
  const mean = capture.sum.map((v) => v / capture.count);
  const { c, scale } = gnm.fit(mean, 0);   // scale fixed by the iris
  const vertices = gnm.mesh(c);
  const { values } = gnm.measure(vertices, gnm.eyeJoints(c));
  // IPD: report the direct pupil-to-pupil measurement. GNM infers eyeball spacing from
  // overall face shape and pulls wide/narrow IPDs toward the average (on a real 68 mm PD
  // it read 62.0 against 65.7 direct); its value is kept for comparison.
  values.ipd_gnm = values.ipd_far;
  values.ipd_far = capture.ipdSum / capture.count;

  // Draw in observation units so the head lines up with the video
  fitVertices = vertices.map((v) => v * scale);
  view.setHead(fitVertices, gnm.headTriangles);
  fitResult = { c, scale };
  fitCaptureRequested = true;   // grabbed after this frame is rendered, with the head
  lastValues = values;
  renderPanel(values);
}

// ============================================================================
// EARS
// ============================================================================

async function startEars() {
  if (!fitResult) return showMessage(MESSAGES.earsFirst, { warn: true });
  showMessage(MESSAGES.earsLoading);
  try {
    earPipeline ??= await loadEarPipeline();
  } catch (error) {
    console.error(error);
    showMessage(MESSAGES.earsLoadFailed, { warn: true });
    return;
  }
  // Printed marker, optional: held beside the ear it checks the scale there
  try {
    markerFinder ??= createMarkerFinder(await loadAruco());
  } catch (error) {
    console.warn("Marker check unavailable", error);
  }
  ears = { left: [], right: [], busy: false, points: null };
  // Sound needs a user gesture to start: this click
  try {
    audio ??= new AudioContext();
    audio.resume();
  } catch (error) {
    console.warn("No audio", error);
  }
  spoken = { key: null, time: 0 };
  phase = "ears";
  setProgress(0, MESSAGES.earsTurn);
  earPrompt("earsTurn");
}

// ----------------------------------------------------------------------------
// Audio guidance for the ear step: the screen can't be seen while turned
// ----------------------------------------------------------------------------

// Short spoken versions of the ear prompts
const SPOKEN = {
  earsTurn: "Turn your head to one side, and hold",
  earsOther: "Now turn to the other side",
  earsMore: "Turn a little further",
  earsBack: "Turn back a little",
  still: "Hold still",
  earsNotFound: "I can't see your ear",
  done: "Done",
};
// A spoken prompt is repeated if it still applies after this long (ms)
const SPEAK_REPEAT_MS = 4000;
let audio = null;           // AudioContext, created on the "Measure ears" click
let spoken = { key: null, time: 0 };

function speak(key) {
  const now = performance.now();
  if (!window.speechSynthesis || (key === spoken.key && now - spoken.time < SPEAK_REPEAT_MS)) return;
  spoken = { key, time: now };
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(SPOKEN[key]));
}

/** Short tone: frequency (Hz), duration (s), start delay (s) */
function beep(freq, duration = 0.05, delay = 0) {
  if (!audio) return;
  const t = audio.currentTime + delay;
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.15, t);
  gain.gain.exponentialRampToValueAtTime(0.001, t + duration);
  osc.connect(gain).connect(audio.destination);
  osc.start(t);
  osc.stop(t + duration);
}

/** Show an ear prompt and say it (key into MESSAGES/SPOKEN; extra text only on screen) */
function earPrompt(key, extra = "", options) {
  showMessage(MESSAGES[key] + extra, options);
  speak(key);
}

/**
 * One frame of ear capture: from the head pose, check the turn and pick the visible ear;
 * find it in the image (async), and measure it on the fitted head's ear plane
 * @param {{R: number[], t: number[]}} pose - camera -> model
 */
function earStep(landmarks, pose, still) {
  const { R, t } = pose;
  // Model -> camera: x_cam = R^T (x_model - t)
  const toCamera = (p) => {
    const d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
    return [0, 1, 2].map((r) => R[r] * d[0] + R[3 + r] * d[1] + R[6 + r] * d[2]);
  };
  const rotate = (v) => [0, 1, 2].map((r) => R[r] * v[0] + R[3 + r] * v[1] + R[6 + r] * v[2]);
  // Face direction (model +z) in the camera frame; 0 = facing the camera
  const facing = rotate([0, 0, 1]);
  const turn = Math.abs((Math.atan2(facing[0], facing[2]) * 180) / Math.PI);

  const planes = ["left", "right"].map((side) => {
    const plane = planeFit(fitVertices, side === "left" ? gnm.earL : gnm.earR);
    let normal = plane.normal;
    // Point the normal away from the head, so the nearer ear is the one facing the camera
    if (normal[0] * (side === "left" ? 1 : -1) < 0) normal = normal.map((v) => -v);
    return { side, centroid: toCamera(plane.centroid), normal: rotate(normal) };
  });
  const plane = planes[0].centroid[2] > planes[1].centroid[2] ? planes[0] : planes[1];
  const other = plane.side === "left" ? "right" : "left";
  const done = (side) => ears[side].length >= EAR_FRAMES;

  if (done(plane.side)) return earPrompt("earsOther");
  const angle = ` (now ${turn.toFixed(0)}°, need ${EAR_TURN_MIN}–${EAR_TURN_MAX}°)`;
  // Outside the range no ear is detected: drop the last ear points so they don't linger
  if (turn < EAR_TURN_MIN || turn > EAR_TURN_MAX) ears.points = null;
  if (turn < EAR_TURN_MIN) {
    const started = ears[plane.side].length > 0;
    return done(other) ? earPrompt("earsOther") : started ? earPrompt("earsMore", angle) : earPrompt("earsTurn");
  }
  if (turn > EAR_TURN_MAX) return earPrompt("earsBack", angle, { warn: true });
  if (!still) return earPrompt("still", "", { warn: true });
  if (ears.busy) return;

  ears.busy = true;
  const W = video.videoWidth, H = video.videoHeight, focalPx = CAMERA_CONFIG.focalLengthNorm * W;
  // Marker beside the ear (optional): search around the ear, wider than one marker
  const earPx = projectPoint(plane.centroid, W, H, focalPx);
  const markerPx = (MARKER_MM * focalPx) / (1000 * Math.hypot(...plane.centroid));
  let marker = null;
  try {
    marker = markerFinder?.find(video, earPx, 1.6 * markerPx) ?? null;
  } catch (error) {
    console.error(error);   // the check is optional: carry on without it
  }
  markerQuad = marker ? marker.corners.map((p) => ({ x: p.x / W, y: p.y / H })) : null;
  detectEars(earPipeline, video, landmarks, earCrop)
    .then((found) => {
      if (phase !== "ears") return;
      // The detection nearest the fitted head's ear
      const expected = projectPoint(plane.centroid, W, H, focalPx);
      const near = (e) => Math.hypot((e.bbox.xmin + e.bbox.xmax) / 2 - expected.x, (e.bbox.ymin + e.bbox.ymax) / 2 - expected.y);
      const ear = found.sort((a, b) => near(a) - near(b))[0];
      if (!ear || near(ear) > ear.bbox.ymax - ear.bbox.ymin) {
        ears.points = null;
        return earPrompt("earsNotFound", "", { warn: true });
      }
      ears.points = ear.landmarks.map((p) => ({ x: p.x / W, y: p.y / H }));
      const planeMm = { centroid: plane.centroid, normal: plane.normal };
      const values = measureEar(ear.landmarks, planeMm, W, H, focalPx);
      if (values.viewCos < EAR_MIN_VIEW_COS) {
        return earPrompt("earsMore", ` (ear seen too edge-on: ${(Math.acos(values.viewCos) * 180 / Math.PI).toFixed(0)}°)`);
      }
      const frame = {
        ...values,
        turn,
        confidence: ear.confidence,
        // Raw inputs, saved in snapshots so measurements can be re-derived and checked
        landmarks: ear.landmarks.map((p) => [+p.x.toFixed(1), +p.y.toFixed(1)]),
        plane: { centroid: plane.centroid, normal: plane.normal },
      };
      if (marker) {
        const sideMm = MARKER_MM;
        frame.marker = { corners: marker.corners.map((p) => [+p.x.toFixed(1), +p.y.toFixed(1)]), ...markerCheck(marker.corners, planeMm, W, H, focalPx, sideMm), sideMm };
      }
      // Crop of the ear as measured (the ear record is saved from these, since by the time
      // anyone clicks "Save snapshot" they are facing the screen again)
      frame.image = earImage(ear.bbox);
      ears[plane.side].push(frame);
      showMessage(MESSAGES.earsFound);
      // Tick per frame (higher when the marker was seen too); a chime when a side is done
      beep(frame.marker ? 1320 : 880);
      if (ears[plane.side].length === EAR_FRAMES) [660, 880, 1320].forEach((f, i) => beep(f, 0.12, 0.15 * i));
      setProgress(Math.min(1, (Math.min(ears.left.length, EAR_FRAMES) + Math.min(ears.right.length, EAR_FRAMES)) / (2 * EAR_FRAMES)), MESSAGES.earsFound);
      if (done("left") && done("right")) finishEars();
    })
    .catch((error) => console.error(error))
    .finally(() => (ears && (ears.busy = false)));
}

/** JPEG crop around an ear box, with its position in the frame */
function earImage(bbox) {
  const side = Math.round(1.6 * Math.max(bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin));
  const x = Math.round((bbox.xmin + bbox.xmax) / 2 - side / 2);
  const y = Math.round((bbox.ymin + bbox.ymax) / 2 - side / 2);
  const c = document.createElement("canvas");
  c.width = c.height = side;
  c.getContext("2d").drawImage(video, x, y, side, side, 0, 0, side, side);
  return { x, y, side, jpeg: c.toDataURL("image/jpeg", 0.9) };
}

const EAR_KEYS = ["ear_length_measured", "ear_width", "concha_height", "concha_width", "tragus_gap"];

/** Median and spread (IQR / 1.35, ~SD) of each measurement over an ear's frames */
function summarizeEar(frames) {
  if (!frames.length) return null;
  const out = { frames: frames.length, turnDeg: null };
  const stats = (xs) => {
    const s = xs.slice().sort((a, b) => a - b);
    const q = (f) => s[Math.min(s.length - 1, Math.floor(f * s.length))];
    return { median: q(0.5), sd: (q(0.75) - q(0.25)) / 1.35 };
  };
  out.turnDeg = stats(frames.map((f) => f.turn)).median;
  // Marker check: ear plane distance / marker's own distance on the same ray (1 = scale right)
  const withMarker = frames.filter((f) => f.marker);
  if (withMarker.length) {
    out.markerCheck = {
      frames: withMarker.length,
      scaleRatio: stats(withMarker.map((f) => f.marker.planeDistMm / f.marker.ownDistMm)),
      ownDistMm: stats(withMarker.map((f) => f.marker.ownDistMm)).median,
      planeDistMm: stats(withMarker.map((f) => f.marker.planeDistMm)).median,
      angleDeg: stats(withMarker.map((f) => f.marker.angleDeg)).median,
      onPlaneMm: stats(withMarker.map((f) => f.marker.onPlaneMm)).median,
      sideMm: withMarker[0].marker.sideMm,
    };
  }
  for (const key of EAR_KEYS) out[key] = stats(frames.map((f) => f[key]));
  return out;
}

function finishEars() {
  const sides = [summarizeEar(ears.left), summarizeEar(ears.right)].filter(Boolean);
  for (const key of EAR_KEYS) {
    lastValues[key] = sides.reduce((s, e) => s + e[key].median, 0) / sides.length;
    // Spread of the average of both ears, plus half the left-right difference
    // (real ears differ a little; a large difference points to a measuring problem)
    const sd = Math.hypot(...sides.map((e) => e[key].sd)) / sides.length;
    const diff = sides.length === 2 ? Math.abs(sides[0][key].median - sides[1][key].median) / 2 : 0;
    lastValues[`${key}_sd`] = Math.hypot(sd, diff);
  }
  ears.points = null;
  phase = "done";
  renderPanel(lastValues);
  setProgress(1, MESSAGES.done);
  showMessage(MESSAGES.done);
  speak("done");
  setTimeout(() => phase === "done" && showMessage(null), 1500);
}

/**
 * Copy of the current video frame at video resolution; with overlays, also the head
 * and guide canvases, mirrored like the live view
 */
function grabFrame(withOverlays) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const g = c.getContext("2d");
  if (withOverlays && UI_CONFIG.mirrorEnabled) {
    g.translate(c.width, 0);
    g.scale(-1, 1);
  }
  g.drawImage(video, 0, 0);
  if (withOverlays) for (const overlay of [canvas, guideCanvas]) g.drawImage(overlay, 0, 0, c.width, c.height);
  return c;
}

/** A copy of an image with a caption bar underneath */
function captionedCopy(source, caption, irisValue, kind) {
  const c = document.createElement("canvas");
  c.width = source.width;
  c.height = source.height + 48;
  const g = c.getContext("2d");
  g.fillStyle = "#111";
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(source, 0, 0);
  g.fillStyle = "#fff";
  g.font = "600 24px system-ui, sans-serif";
  g.fillText(caption, 16, source.height + 32);
  return { image: c, irisMm: irisValue, kind, time: new Date() };
}

/** The frame closest to an ear's median length and width (its most typical frame) */
function typicalEarFrame(frames, summary) {
  const off = (f) =>
    Math.hypot(f.ear_length_measured / summary.ear_length_measured.median - 1, f.ear_width / summary.ear_width.median - 1);
  return frames.reduce((best, f) => (off(f) < off(best) ? f : best));
}

/**
 * Ear record, saved with the snapshot from frames captured while measuring: a PNG with
 * each ear's most typical frame (points drawn) and its measurements, and a JSON with
 * every frame
 * @param {string} name - file-name stem
 * @returns {Promise<Object<string, Blob>>} file name -> contents
 */
async function earRecordFiles(name) {
  const TILE = 480, PAD = 24, ROW = 30;
  const sides = ["left", "right"].filter((side) => ears[side].length);
  const summaries = Object.fromEntries(sides.map((side) => [side, summarizeEar(ears[side])]));
  const canvas = document.createElement("canvas");
  canvas.width = sides.length * (TILE + PAD) + PAD;
  canvas.height = PAD + TILE + PAD + ROW * (EAR_KEYS.length + 4) + PAD;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const labels = { ear_length_measured: "Ear length", ear_width: "Ear width", concha_height: "Concha height", concha_width: "Concha width", tragus_gap: "Tragus to antitragus" };

  for (const [k, side] of sides.entries()) {
    const frame = typicalEarFrame(ears[side], summaries[side]);
    const x0 = PAD + k * (TILE + PAD);
    const img = new Image();
    img.src = frame.image.jpeg;
    await img.decode();
    ctx.drawImage(img, x0, PAD, TILE, TILE);
    const s = TILE / frame.image.side;
    const at = ([u, v]) => [x0 + (u - frame.image.x) * s, PAD + (v - frame.image.y) * s];
    ctx.strokeStyle = ctx.fillStyle = "#00ffc8";
    ctx.lineWidth = 2;
    for (const [a, b] of EAR_LINES) {
      ctx.beginPath();
      for (let i = a; i < b; i++) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, ...at(frame.landmarks[i]));
      ctx.stroke();
    }
    for (const p of frame.landmarks) { const [u, v] = at(p); ctx.fillRect(u - 2, v - 2, 4, 4); }

    let y = PAD + TILE + PAD + 20;
    ctx.fillStyle = "#00ffc8";
    ctx.font = "600 22px system-ui, sans-serif";
    ctx.fillText(`${side === "left" ? "Left" : "Right"} ear (turned ${summaries[side].turnDeg.toFixed(0)}°, ${ears[side].length} frames)`, x0, y);
    for (const key of EAR_KEYS) {
      y += ROW;
      ctx.fillStyle = "#ddd";
      ctx.font = "18px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(labels[key], x0, y);
      ctx.fillStyle = "#fff";
      ctx.font = "600 18px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(`${summaries[side][key].median.toFixed(1)} ± ${summaries[side][key].sd.toFixed(1)} mm`, x0 + TILE, y);
      ctx.textAlign = "left";
    }
    const check = summaries[side].markerCheck;
    ctx.fillStyle = "#ddd";
    ctx.font = "16px system-ui, sans-serif";
    y += ROW;
    ctx.fillText(
      check
        ? `Marker check (${check.frames} frames): ear plane at ${check.planeDistMm.toFixed(0)} mm, marker at ${check.ownDistMm.toFixed(0)} mm`
        : "Marker check: marker not seen",
      x0, y
    );
    if (check) {
      y += ROW;
      ctx.fillText(`Scale ${((check.scaleRatio.median - 1) * 100).toFixed(1)}% (0 = right), marker ${check.angleDeg.toFixed(0)}° off the ear plane`, x0, y);
    }
  }

  const data = {
    time: new Date().toISOString(),
    video: { width: video.videoWidth, height: video.videoHeight },
    calibration: { irisMm: irisMm ?? CAMERA_CONFIG.irisDiameterMm, calibrated: irisMm != null },
    glassesDetected: lastGlasses,
    earsMm: summaries,
    earFrames: { ...Object.fromEntries(sides.map((side) => [side, ears[side]])), focalPx: CAMERA_CONFIG.focalLengthNorm * video.videoWidth },
  };
  const png = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  return { [`${name}-ears.png`]: png, [`${name}-ears.json`]: new Blob([JSON.stringify(data)], { type: "application/json" }) };
}

// ============================================================================
// SNAPSHOT
// ============================================================================

/** Save the current view + measurement table (PNG) and the data (JSON) */
function saveSnapshot() {
  const label = document.getElementById("snapshot_label").value.trim();
  const name = snapshotName(label);
  const median = (xs) => (xs.length ? xs.slice().sort((a, b) => a - b)[xs.length >> 1] : null);
  const glasses = {
    detected: lastGlasses,
    medianBridgeScore: median(glassesScores.map((g) => g.bridge)),
    medianRimScore: median(glassesScores.map((g) => g.rim)),
  };
  const fmt = (v) => (v == null ? "--" : v.toFixed(1));

  const lines = [
    [label || "headSize GNM"],
    ["Time", new Date().toLocaleString()],
    ["Distance", lastDistanceCm == null ? "--" : `${lastDistanceCm.toFixed(0)} cm`],
    ["Size from", irisMm == null ? "average iris 11.7 mm" : `calibrated, iris ${irisMm.toFixed(2)} mm`],
    ["Glasses detected", glasses.detected ? "yes" : "no"],
    ["Capture", phase === "done" ? "complete" : `${capture.count}/${CAPTURE_FRAMES} frames`],
    ["Image", fitRecord ? `at head fit, ${fitRecord.time.toLocaleTimeString()}` : "live view"],
    ["IPD direct (no GNM)", capture.count ? `${(capture.ipdSum / capture.count).toFixed(1)} mm` : "--"],
  ];
  for (const group of GROUPS) {
    lines.push([group]);
    for (const m of shownMetrics(lastValues).filter((x) => x.group === group)) {
      lines.push([m.label, lastValues ? `${fmt(lastValues[m.key])} ± ${fmt(errorFor(m, lastValues[m.key]))} mm` : "--"]);
    }
  }

  const data = {
    label,
    time: new Date().toISOString(),
    video: { width: video.videoWidth, height: video.videoHeight },
    distanceCm: lastDistanceCm,
    glasses,
    capture: { complete: phase === "done", frames: capture.count },
    scale: fitResult?.scale ?? null,
    measurementsMm: lastValues,
    noseBridge: lastValues ? noseBridge(lastValues.bridge_projection) : null,
    // Per-ear medians from side views, with frame counts (see "Measure ears")
    earsMm: ears ? { left: summarizeEar(ears.left), right: summarizeEar(ears.right) } : null,
    // Pupil-to-pupil distance measured directly from the landmarks (iris scale, far PD),
    // averaged over the captured frames: separates iris-scale error from GNM's fit
    directIpdFarMm: capture.count ? capture.ipdSum / capture.count : null,
    calibration: {
      irisMm: irisMm ?? CAMERA_CONFIG.irisDiameterMm,
      calibrated: irisMm != null,
      // This session's calibration event, if any (its image is saved alongside)
      event: calibrationRecord ? { kind: calibrationRecord.kind, irisMm: calibrationRecord.irisMm, time: calibrationRecord.time.toISOString() } : null,
    },
    imageFrom: fitRecord ? { event: "head fit", time: fitRecord.time.toISOString() } : { event: "live view" },
    typicalErrorMm: Object.fromEntries(
      METRICS.map((m) => [m.key, lastValues ? errorFor(m, lastValues[m.key]) : m.rmse ?? null])
    ),
  };

  Promise.all([
    fitRecord
      ? composeImage({ video: fitRecord.view, overlays: [], mirrored: false, lines })
      : composeImage({ video, overlays: [canvas, guideCanvas], mirrored: UI_CONFIG.mirrorEnabled, lines }),
    rawFrame(fitRecord ? fitRecord.raw : video),
    calibrationRecord && new Promise((resolve) => calibrationRecord.image.toBlob(resolve, "image/png")),
  ]).then(async ([png, frame, calibrationPng]) => {
    const files = {
      [`${name}.png`]: png,
      [`${name}-frame.png`]: frame,
      [`${name}.json`]: new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    };
    if (calibrationPng) files[`${name}-calibration.png`] = calibrationPng;
    // Per-frame ear data and crops go in a separate ear record
    if (ears && (ears.left.length || ears.right.length)) Object.assign(files, await earRecordFiles(name));
    // One download: everything in a zip (falls back to separate files if zip can't load)
    try {
      download(await zipFiles(files), `${name}.zip`);
    } catch (error) {
      console.warn("Zip unavailable, saving separate files", error);
      for (const [file, blob] of Object.entries(files)) download(blob, file);
    }
  });
}

// ============================================================================
// PANEL
// ============================================================================

function setProgress(fraction, text) {
  progressBar.style.width = `${Math.round(fraction * 100)}%`;
  fitStatus.textContent = text;
}

/** US fitted hat size from head circumference: diameter in inches, nearest 1/8 */
function usHatSize(circumferenceMm) {
  const eighths = Math.round((circumferenceMm / 25.4 / Math.PI) * 8);
  const whole = Math.floor(eighths / 8);
  const frac = eighths % 8;
  const g = frac === 0 ? 1 : frac % 4 === 0 ? 4 : frac % 2 === 0 ? 2 : 1;
  return frac ? `${whole} ${frac / g}/${8 / g}` : `${whole}`;
}

/** Typical error: iris-scale simulation error, or shape error + calibration error once calibrated */
function errorFor(metric, value) {
  if (value == null) return null;
  const scaleSd = irisMm == null ? IRIS_SCALE_SD : CALIBRATION_SD;
  if (metric.measured) return Math.hypot(lastValues?.[`${metric.key}_sd`] ?? 0, scaleSd * value);
  return irisMm == null ? metric.rmse : Math.hypot(metric.rmseShape, CALIBRATION_SD * value);
}

function renderPanel(values) {
  const row = (m) => {
    const v = values?.[m.key];
    const value = v == null ? "--" : `${v.toFixed(1)} mm`;
    const error = v == null ? "" : `±${errorFor(m, v).toFixed(1)}`;
    return `<div class="metric-row"><span class="label" title="${m.desc}">${m.label}&nbsp;<span class="info">ⓘ</span></span><span class="value">${value}</span><span class="error">${error}</span></div>`;
  };
  panelBody.innerHTML = GROUPS.map((group) => {
    // US hat size goes right after the hat measurements (circumference, length, breadth)
    let rows = shownMetrics(values).filter((m) => m.group === group).map((m) => {
      let html = row(m);
      if (m.key === "breadth" && values) {
        html += `<div class="metric-row"><span class="label" title="US hat size from the circumference.">US hat size&nbsp;<span class="info">ⓘ</span></span><span class="value">≈ ${usHatSize(values.circumference)}</span></div>`;
      }
      return html;
    }).join("");
    if (group === "Glasses" && values) {
      rows += `<div class="metric-row"><span class="label" title="From the bridge projection: low, medium or high compared with the GNM Head population (thirds). A low bridge suits low-bridge-fit frames or adjustable nose pads.">Nose bridge&nbsp;<span class="info">ⓘ</span></span><span class="value">${noseBridge(values.bridge_projection)}</span></div>`;
    }
    return `<div class="metric-card"><h2>${group}</h2>${rows}</div>`;
  }).join("");
}

// ============================================================================
// DISPLAY
// ============================================================================

function resizeDisplay() {
  const rect = liveView.getBoundingClientRect();
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const scale = Math.min(rect.width / vw, rect.height / vh);
  const w = Math.round(vw * scale);
  const h = Math.round(vh * scale);
  video.style.width = `${w}px`;
  video.style.height = `${h}px`;
  view.setCamera(vw, vh, CAMERA_CONFIG.focalLengthNorm * vw, w, h);
  guide.resize(w, h);
  calibration.resize(w, h);
  guideMsg.style.left = `${w / 2}px`;
  guideMsg.style.top = `${h - 12}px`;
}

function setupControls() {
  document.getElementById("mesh_toggle").addEventListener("change", (e) => view.setMeshVisible(e.target.checked));
  document.getElementById("remeasure").addEventListener("click", resetCapture);
  document.getElementById("snapshot").addEventListener("click", () => (snapshotRequested = true));
  document.getElementById("calibrate_card").addEventListener("click", startCalibration);
  document.getElementById("calibrate_marker").addEventListener("click", startMarkerCalibration);
  document.getElementById("measure_ears").addEventListener("click", startEars);
  document.getElementById("calib_reset").addEventListener("click", () => {
    irisMm = null;
    calibrationRecord = null;
    saveIris(null);
    measurer = makeMeasurer();
    updateScaleStatus();
    resetCapture();
  });
  updateScaleStatus();
  window.addEventListener("resize", resizeDisplay);

  video.classList.toggle("mirrored", UI_CONFIG.mirrorEnabled);
  canvas.classList.toggle("mirrored", UI_CONFIG.mirrorEnabled);
  guideCanvas.classList.toggle("mirrored", UI_CONFIG.mirrorEnabled);
}

// ============================================================================
// LOOP
// ============================================================================

function renderFrame() {
  const { faceResults } = models.processFrame(video);
  const landmarks = faceResults?.faceLandmarks?.[0] || null;
  const videoSize = { width: video.videoWidth, height: video.videoHeight };
  const m = measurer.update(landmarks, videoSize, videoSize);
  frameNumber++;

  let aligned = false;
  if (!landmarks || !m.point3D) {
    lastNose = null;
    if (phase === "done") view.hide();
    else if (phase === "calibrate") showMessage(MESSAGES.card);
    else if (phase === "marker") showMessage(MESSAGES.marker);
    else if (phase === "lower") lowerStep(null, null);
    else if (phase === "ears") earPrompt("earsBack", "", { warn: true });
    else if (phase !== "mark") showMessage(phase === "capture" ? MESSAGES.lost : MESSAGES.align, { warn: phase === "capture" });
  } else {
    // Glasses: sampled every few frames, decided on the median of recent scores
    // (Not scored while a card or marker is held up: its edge above the brows looks
    // like a glasses bridge)
    const holding = phase === "calibrate" || phase === "marker" || phase === "mark" || phase === "ears" || phase === "lower";
    if (!holding && frameNumber % GLASSES_EVERY === 0) {
      glassesScores.push(glassesDetector.score(video, landmarks, videoSize.width, videoSize.height));
      if (glassesScores.length > GLASSES_WINDOW) glassesScores.shift();
    }
    lastGlasses = glassesScores.length >= 5 && looksLikeGlasses(glassesScores);
    glassesStatus.hidden = !lastGlasses;
    lastDistanceCm = m.distanceCm;

    const check = guide.assess(landmarks, m.distanceCm);
    aligned = check.ok;
    const nose = [landmarks[1].x * videoSize.width, landmarks[1].y * videoSize.height];
    const motion = lastNose ? Math.hypot(nose[0] - lastNose[0], nose[1] - lastNose[1]) / (check.faceHeight * videoSize.height) : 0;
    lastNose = nose;
    const still = motion < MAX_MOTION;

    if (phase === "calibrate") {
      calibrationStep(landmarks, m, still);
    } else if (phase === "marker") {
      markerStep(landmarks, m, still);
    } else if (phase === "lower") {
      lowerStep(landmarks, m);
    } else if (phase === "mark") {
      // marking a frozen frame: nothing to do live
    } else if (phase === "ears") {
      const pose = gnm.pose(observe(m.point3D), fitResult.c, fitResult.scale);
      view.setPose(pose.R, pose.t);
      earStep(landmarks, pose, still);
    } else if (phase !== "done") {
      if (!check.ok) showMessage(MESSAGES[check.reason], { warn: phase === "capture" });
      else if (phase === "align") phase = "capture";

      if (phase === "capture" && check.ok) {
        if (!still) {
          showMessage(MESSAGES.still, { warn: true });
        } else {
          showMessage(MESSAGES.measuring);
          accumulate(observe(m.point3D));
          capture.ipdSum += m.ipd.far;
          setProgress(capture.count / CAPTURE_FRAMES, MESSAGES.measuring);
          if (capture.count >= CAPTURE_FRAMES) {
            refit();
            phase = "done";
            setProgress(1, MESSAGES.done);
            showMessage(MESSAGES.done);
            setTimeout(() => phase === "done" && showMessage(null), 1500);
          }
        }
      }
    }

    if (phase === "done" && fitResult) {
      const { R, t } = gnm.pose(observe(m.point3D), fitResult.c, fitResult.scale);
      view.setPose(R, t);
    }
  }

  guide.draw(phase, landmarks, aligned, capture.count / CAPTURE_FRAMES);
  if ((phase === "marker" || phase === "ears") && markerQuad) guide.drawQuad(markerQuad);
  if (phase === "ears" && ears?.points) guide.drawLines(ears.points, EAR_LINES);
  view.render();
  // Read the WebGL canvas right after rendering (it isn't kept afterwards)
  if (fitCaptureRequested) {
    fitCaptureRequested = false;
    fitRecord = { view: grabFrame(true), raw: grabFrame(false), time: new Date() };
  }
  if (snapshotRequested) {
    snapshotRequested = false;
    saveSnapshot();
  }
  window.requestAnimationFrame(renderFrame);
}

function cameraErrorMessage(error) {
  switch (error?.name) {
    case "NotAllowedError":
      return "Camera access was blocked. Allow camera access for this site, then reload.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No usable camera was found.";
    case "NotReadableError":
      return "The camera is in use by another app. Close it, then reload.";
    default:
      return `Could not start the camera (${error?.name || error}).`;
  }
}

(async () => {
  setupControls();
  renderPanel(null);
  try {
    [models, gnm] = await Promise.all([
      ModelManager.initialize(CAMERA_CONFIG),
      loadGNM(new URL("gnm_head_fit.bin", import.meta.url).href),
    ]);
  } catch (error) {
    console.error(error);
    statusEl.textContent = "Could not load the models. Check your connection, then reload.";
    return;
  }
  resetCapture();

  statusEl.textContent = "Starting camera…";
  try {
    await camera.initialize();
  } catch (error) {
    console.error(error);
    statusEl.textContent = cameraErrorMessage(error);
    return;
  }
  video.addEventListener(
    "loadedmetadata",
    () => {
      statusEl.hidden = true;
      resizeDisplay();
      renderFrame();
    },
    { once: true }
  );
})();
