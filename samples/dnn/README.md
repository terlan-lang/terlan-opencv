# Deep Neural Network Tutorial Inventory

This inventory groups the pinned OpenCV DNN examples by package capability.
Runnable samples use committed in-memory ONNX bytes and deterministic CPU
inputs; they do not download models or read ambient files.

| Tutorial family | Terlan coverage | Status |
| --- | --- | --- |
| Image-to-blob preprocessing | `BlobPreprocessing.terl` | Runnable |
| ONNX model loading and forward execution | `IdentityInference.terl` | Runnable |
| Classification projections | `Classification.terl` | Runnable |
| Segmentation mask projections | `Segmentation.terl` | Runnable |
| Bounded object-detection decoding and NMS | `ObjectDetection.terl` | Runnable |
| Model downloads and framework conversion | Package-manager/tooling concern, not runtime inference | Blocked |
| Trackers, action recognition, pose, and text pipelines | No reviewed end-to-end model and state contracts | Blocked |
| Dynamic shapes, named tensors, and multiple inputs or outputs | No reviewed arbitrary tensor contract | Blocked |
| Accelerator backend selection | CPU-only package contract | Blocked |

New inference tutorials require a committed bounded model fixture and exact
copied-output assertions.
