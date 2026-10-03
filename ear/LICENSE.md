# Ear models: non-commercial use only

`BlazeEar_web.onnx` (ear detector) and `EarLandmarker_web.onnx` (55-point ear landmarks)
come from [Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker) and BlazeEar by
Shameem Hameed, licensed CC BY-NC 4.0. They are trained on data licensed for
non-commercial research only (iBUG ears; AudioEar2D, whose images come from FFHQ), so
they are **not** covered by this repository's Apache License 2.0. Use them for
non-commercial research only.

The scripts in this folder (`earlandmarker_inference.js`, `blazeear_inference.js`,
`smoothing.js`) are from the same projects and are CC BY-NC 4.0 as published there.

`BlazeFace_web.onnx` is MediaPipe's published BlazeFace face detector (Apache License
2.0, Google LLC), converted by the BlazeEar project.

For commercial use, drop the ear step or replace these models with ones trained on
commercially licensed data.
