/**
 * Webcam access and MediaPipe face landmarker
 * @module camera
 */

import {
  FilesetResolver,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

/**
 * Manages camera access and video stream operations
 */
export class CameraManager {
  /**
   * @param {HTMLVideoElement} videoElement - Video element for camera feed
   * @param {Object} config - Camera configuration
   */
  constructor(videoElement, config) {
    this.video = videoElement;
    this.config = config;
  }

  /**
   * Device ID of the highest-priority camera by label, or null if none match.
   * Labels are only readable after camera permission has been granted.
   * @returns {Promise<string|null>}
   */
  async findPreferredCamera() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((device) => device.kind === "videoinput");
    for (const pattern of this.config.cameraPreferences.priorities) {
      const match = videoDevices.find((device) => pattern.test(device.label));
      if (match) return match.deviceId;
    }
    return null;
  }

  /**
   * Open the front camera, then switch to a preferred one (e.g. "Front Wide")
   * if it is a different device
   */
  async initialize() {
    const { videoSize, cameraPreferences } = this.config;
    const size = { width: videoSize.width, height: videoSize.height, resizeMode: "none" };
    const open = (video) => navigator.mediaDevices.getUserMedia({ audio: false, video: { ...size, ...video } });

    let stream = await open({ facingMode: cameraPreferences.facingMode });
    const preferredId = await this.findPreferredCamera();
    const currentId = stream.getVideoTracks()[0]?.getSettings().deviceId;
    if (preferredId && preferredId !== currentId) {
      stream.getTracks().forEach((track) => track.stop());
      stream = await open({ deviceId: { exact: preferredId } });
    }
    this.video.srcObject = stream;
  }
}

/**
 * Manages MediaPipe model loading and inference
 */
export class ModelManager {
  /**
   * @param {Object} faceLandmarker - MediaPipe face landmarker instance
   */
  constructor(faceLandmarker) {
    this.faceLandmarker = faceLandmarker;
    this.lastVideoTime = -1;
    this.faceResults = null;
  }

  /**
   * Initialize MediaPipe models
   * @param {Object} config - Camera configuration object
   * @returns {Promise<ModelManager>} Initialized ModelManager instance
   */
  static async initialize(config) {
    const vision = await FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${config.mediaPipeVersion}/wasm`
    );

    const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: config.runningMode,
      numFaces: 1,
    });

    return new ModelManager(faceLandmarker);
  }

  /**
   * Process video frame and update detection results
   * @param {HTMLVideoElement} video - Video element to process
   * @returns {{faceResults: Object|null}}
   */
  processFrame(video) {
    // Only process if video time has changed
    if (video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = video.currentTime;
      this.faceResults = this.faceLandmarker.detectForVideo(video, Date.now());
    }

    return { faceResults: this.faceResults };
  }
}
