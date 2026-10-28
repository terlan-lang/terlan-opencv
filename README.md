# Terlan OpenCV

Use [OpenCV](https://github.com/opencv/opencv) from Terlan for image
processing, feature detection, camera calibration, machine learning, and ONNX
inference.

## Example

```terlan
module example.Threshold.

import opencv.Imgproc.
import opencv.Mat.
import opencv.Types.{BinaryThreshold, Gray}.
import std.vm.Bytes.

pub make_binary_image(): Bytes ->
    Bytes.from_list([0, 64, 128, 255])
    |> Mat.from_u8(2, 2, Gray)
    |> Imgproc.threshold(100.0, 255.0, BinaryThreshold)
    |> Mat.to_u8().
```

The source is a 2x2 grayscale image with pixel values `0`, `64`, `128`, and
`255`. Applying a threshold of `100` produces the binary image
`[0, 0, 255, 255]`: darker pixels become black and brighter pixels become
white.

## Installation

The first packaged release is coming soon. To use the current version from
source, install Terlan `0.0.7`, Rust, CMake, a C++ compiler, and Git, then clone
the repository:

```sh
git clone https://github.com/terlan-lang/terlan-opencv.git
cd terlan-opencv
```

Add OpenCV `4.13` and prepare the package:

```sh
git clone --branch 4.13.0 --depth 1 \
  https://github.com/opencv/opencv.git upstream/opencv
make opencv-generate TERLC="$(command -v terlc)"
make opencv-build
```

Reference the prepared package from a Terlan project:

```toml
[dependencies]
opencv = { path = "../terlan-opencv/generated" }
```

Compile all included examples to confirm the installation:

```sh
make opencv-samples-check TERLC="$(command -v terlc)"
```

## Available Now

- Matrix creation, copying, conversion, inspection, channel operations,
  arithmetic, masks, and statistics
- Typed matrix exchange with Terlan CUDA and other tensor packages
- Deterministic random matrices, dot products, matrix multiplication, dense
  system solving, inversion, determinants, norms, and traces
- Per-channel statistics, symbolic reductions, normalization, row and column
  sorting, and finite exponential, logarithm, power, and square-root transforms
- Bounded in-memory YAML, XML, and JSON matrix persistence
- Linked runtime version and build details, processor and worker counts, tick
  frequency, and optimized-code-path reporting
- Resizing, color conversion, filtering, thresholding, morphology, edge
  detection, histograms, contours, connected components, and drawing
- Affine and perspective transforms, image pyramids, undistortion, camera
  projection, and pose estimation
- FAST, AGAST, ORB, BRISK, AKAZE, SIFT, KAZE, MSER, Shi-Tomasi, and blob
  feature detection
- Binary and floating-point descriptor matching
- K-nearest-neighbor and linear support-vector-machine classification
- In-memory BMP, PNG, JPEG, WebP, and TIFF encoding and decoding
- In-memory ONNX loading, image preprocessing, CPU inference,
  classification, segmentation, object-detection decoding, and non-maximum
  suppression

## Coming Soon

- Full behavioral parity with the pinned OpenCV 4.13 public runtime API
- Installable package releases
- Video processing and tracking
- Computational photography and image stitching
- Stereo calibration, depth processing, and expanded 3D operations
- More classifiers, clustering, and dimensionality reduction
- Multiple-input and multiple-output neural networks, dynamic tensor shapes,
  and additional model pipelines
- Hardware-accelerated execution

## Platform Support

| Platform | Status |
| --- | --- |
| Linux x86-64 | Supported from source |
| Linux ARM64 | Coming soon |
| macOS on Apple silicon | Coming soon |
| macOS on Intel | Coming soon |
| Windows x86-64 | Coming soon |

Linux x86-64 is the currently validated development platform and the target of
the first packaged artifact. That artifact requires glibc 2.35 or newer and a
system C++ runtime providing `GLIBCXX_3.4.30` or newer. Other platforms will be
listed as supported only after their packaged examples pass on the target OS.

## Samples

The [samples](samples/README.md) directory contains runnable Terlan programs
covering core matrices, image processing, codecs, features, calibration,
machine learning, and DNN inference.

## Repository Scripts

Repository orchestration is written in Terlan and registered in
`terlan.toml`. `make opencv-scripts-check` recursively type-checks, fully
lints, and verifies formatting for repository `.terls` files.
Make runs those scripts from an isolated `_build/scripts` output directory so
their compiler artifacts cannot replace package or sample VM images.
`scripts/CheckSources.terls` also runs the grouped-binding and
function-reference lint gates over every source root.
`scripts/CheckCppValidation.terls` owns isolated compilation and execution of
the header validation fixtures, while `scripts/GeneratePackage.terls` owns
binding generation and deterministic test and sample staging.
`scripts/RuntimeChecks.terls` runs sample expectations, native-helper failure
contracts, generated package tests, and external consumer tests with explicit
runtime environments. `scripts/PackageArtifact.terls` owns artifact assembly,
deterministic replay, verification, and the extracted-package consumer smoke
test. `scripts/FreshWorkspaceCheck.terls` validates immutable inputs, clones
clean compiler, package, and OpenCV repositories, rebuilds the required
compiler tools, runs the package consumer gate, and records verified promotion
evidence. `scripts/Workspace.terls` creates the generated native dependency
link and performs idempotent cleanup through portable Terlan filesystem APIs.

The ONNX fixture freshness check remains in Make because `protoc --encode`
emits binary stdout and the current Terlan process API captures UTF-8 text.

## License

Apache-2.0. See [LICENSE](LICENSE).
