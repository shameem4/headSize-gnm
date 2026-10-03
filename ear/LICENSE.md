# Ear models and scripts: licence terms

Everything in this folder comes from [Ear_Landmarker](https://github.com/shameem4/Ear_Landmarker)
and [BlazeEar](https://github.com/shameem4/BlazeEar) by Shameem Hameed, whose **code** is
licensed under the Apache License 2.0. The **trained weights** are a separate question,
because they inherit the terms of the data they were trained on.

| File | Terms | Commercial use |
|---|---|---|
| `earlandmarker_inference.js`, `blazeear_inference.js`, `smoothing.js` | Apache-2.0 | yes |
| `BlazeFace_web.onnx` | MediaPipe's published BlazeFace weights, unmodified (Apache-2.0, Google LLC) | yes |
| `EarLandmarker_web.onnx` | Trained mostly on non-commercial data: iBUG ears (non-commercial research only) and AudioEar2D (CC BY 4.0 annotations on FFHQ images, which are CC BY-NC-SA 4.0). Ear_Landmarker's README states the weights are **research use only**. | **no** |
| `BlazeEar_web.onnx` | Apache-2.0: BlazeEar's README states its code, architecture and trained weights are Apache-2.0. Trained on Open Images and Roboflow Universe ear datasets (CC BY terms); no training data is redistributed. | yes |

For a commercial product, either drop the landmark step or retrain the landmark model on
commercially licensed data. This is not legal advice.
