# OpenCV Tutorials In Terlan

This tree ports the executable concepts from the OpenCV 4.12 tutorial index to
the public Terlan package API. Samples use deterministic in-memory inputs so
they run without GUI, camera, or filesystem capabilities.

## Current Ports

| OpenCV tutorial area | Inventory | Runnable samples |
| --- | --- | --- |
| Core functionality | `core/README.md` | 5 |
| Image processing | `imgproc/README.md` | 15 |
| Image codecs | `imgcodecs/README.md` | 1 |
| 2D features | `features2d/README.md` | 7 |
| Camera calibration and 3D | `calib3d/README.md` | 3 |
| Deep neural networks | `dnn/README.md` | 5 |
| Machine learning | `ml/README.md` | 2 |

The current runtime matrix contains 38 deterministic samples.

## Pending Capability Areas

The upstream index also covers video I/O, GUI, video analysis, object
detection, computational photography, stitching, G-API, and CUDA. Those
areas are classified in [CAPABILITIES.md](CAPABILITIES.md). The sample tree
must not imitate unsupported behavior with no-op examples or acquire ambient
host capabilities through OpenCV.

`make opencv-samples-check` regenerates the OpenCV package and a separate
`samples.*` package that depends on it, then compile-checks every sample.
`make opencv-samples-runtime-check` executes every emitted tutorial artifact
through the guarded native runtime and compares its output with
`expected-output.tsv`.
