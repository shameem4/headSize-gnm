# Ear models and scripts: licence terms

Everything in this folder comes from [Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)
and [BlazeEar](https://github.com/shameem4/BlazeEar) by Shameem Hameed, whose **code** is
licensed under the Apache License 2.0. Both upstream projects state that their trained
weights are Apache-2.0 as well, so everything here allows commercial use.

| File | Terms | Commercial use |
|---|---|---|
| `earlandmarker_inference.js`, `blazeear_inference.js`, `ear_tracker.js`, `smoothing.js` | Apache-2.0 | yes |
| `BlazeFace_web.onnx` | MediaPipe's published BlazeFace weights, unmodified (Apache-2.0, Google LLC) | yes |
| `EarLandmarker_web.onnx` | Apache-2.0: Ear_Landmarker's README and NOTICE state the shipped weights are Apache-2.0. They were trained only on commissioned annotations of images vetted to exclude non-commercial licences; no iBUG or FFHQ data is in them. Ear_Landmarker's NOTICE discloses that the annotation was seeded with an earlier model's predictions and then hand-corrected. | yes |
| `BlazeEar_web.onnx` | Apache-2.0: BlazeEar's README states its code, architecture and trained weights are Apache-2.0. Trained on Open Images and Roboflow Universe ear datasets (CC BY terms); no training data is redistributed. | yes |

This summarises the upstream statements; it is not legal advice.
