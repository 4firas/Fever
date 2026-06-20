# Fever — Models

Place the BlazePose model files here so they get bundled into the app:

```
Models/
└── pose_landmark_full.tflite   # 6.4 MB, MediaPipe Pose (33 landmarks)
```

## Where to get the model

Official MediaPipe model assets are distributed via the MediaPipe tasks-vision
package. The pose landmarker model lives at:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

`pose_landmarker_full.task` is the Tasks-API bundle (recommended for
`MediaPipeTasksVision`). For the raw TFLite path, extract
`pose_landmark_full.tflite` from the legacy MediaPipe pose solution assets.

## Wiring

1. Copy the model into `Fever/Sources/Models/`.
2. Uncomment `resources: [ .copy("Models") ]` in `Package.swift` (or add the
   file to the Xcode target's "Copy Bundle Resources" build phase).
3. Load it in `MediaPipePoseLandmarker` via
   `PoseLandmarkerOptions(modelPath:)`.

> The pipeline ships with `StubPoseLandmarker` so the solver + OSC path can be
> built and exercised without the model or the MediaPipe framework present.
