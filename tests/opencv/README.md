# OpenCV Package Tests

Each generated OpenCV module has package-owned tests with substantive `@test`
entries. Tests are split by API ownership: matrix, core linear algebra and
dense arrays, types, codecs, image processing, components, contours, features,
calibration, machine learning, and DNN. The feature module further separates
detector, descriptor, matcher, copied-result, and lifecycle contracts so one
failed family is identifiable.

`DnnDetectionChecks.terl` owns embedded ONNX detection fixtures and detection
projection checks. It deliberately exposes only `valid()` to keep fixture and
resource-lifecycle details out of the package test orchestrator.

Package generation copies the support module into `generated/src/opencv` and
the complete test suite into `generated/tests/opencv`. The C++ binding
generator does not emit unconditional passing placeholder tests. The external
consumer keeps independent checks under `tests/consumer`.
