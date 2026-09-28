/**
 * Centralized Configuration for headSize Application
 * @module config
 */

// ============================================================================
// CAMERA & VIDEO SETTINGS
// ============================================================================

/**
 * Camera and MediaPipe configuration
 */
export const CAMERA_CONFIG = {
  // MediaPipe version
  mediaPipeVersion: "0.10.0",
  runningMode: "VIDEO",

  // Video resolution
  videoSize: {
    width: 1280,
    height: 720,
  },

  // Camera selection preferences
  cameraPreferences: {
    // Priority order for camera selection (regex patterns)
    priorities: [
      /front.*wide/i,      // "Front Wide" camera (iPhone, etc.)
      /wide.*front/i,      // Alternative naming
      /front/i,            // Any front camera
      /user/i,             // User-facing camera
    ],
    facingMode: "user",    // Fallback: "user" (front) or "environment" (rear)
  },

  // Physical measurements
  irisDiameterMm: 11.7,    // Average human iris diameter in millimeters

  // Focal length as a fraction of video width (0.8 ≈ 64° horizontal FOV, a
  // typical webcam). Only affects distance; mm values barely depend on it.
  focalLengthNorm: 0.8,

  // MediaPipe's relative depth (z) is about twice as deep as real faces: at 0.5,
  // measurements between points at different depths no longer grow as the head turns.
  // (main.js overrides this to 1 for the GNM fit.)
  depthScale: 0.5,

  // Exponential smoothing of iris diameter (0-1, higher = faster response)
  irisSmoothing: 0.3,

  // Pupil-to-eye-rotation-center distance, used to convert the measured
  // (converged) IPD to far IPD
  eyeRotationRadiusMm: 10,
};

/**
 * MediaPipe Face Landmarker indices for facial features
 * Reference: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
 */
export const HEAD_CONFIG = {
  // Iris boundary points and pupil centers (subject's left/right)
  iris: {
    left: [474, 475, 476, 477],
    right: [469, 470, 471, 472],
  },
  pupil: { left: 473, right: 468 },

  // Eye corners, ordered from the subject's right to left
  eyeCorners: {
    left: [362, 263],
    right: [33, 133],
  },

  // Face edges [right, left]
  faceWidth: [127, 356],

  // Chin -> forehead, defines the face's vertical axis
  faceUp: [152, 10],

  // Nose rows, ordered across the nose
  bridgeRow: [190, 189, 193, 168, 417, 413, 414],
  padRow: [114, 188, 122, 6, 351, 412, 343],

  // Angles as [vertex, armA, armB]
  padAngle: [8, 412, 6],
  flareAngle: [6, 122, 351],
};

/** Display */
export const UI_CONFIG = {
  // Mirror the video (selfie view)
  mirrorEnabled: true,
};
