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
- Resizing, color conversion, filtering, thresholding, morphology, edge
  detection, histograms, contours, connected components, and drawing
- Affine and perspective transforms, image pyramids, undistortion, camera
  projection, and pose estimation
- FAST, AGAST, ORB, BRISK, AKAZE, SIFT, KAZE, MSER, Shi-Tomasi, and blob
  feature detection
- Binary and floating-point descriptor matching
- K-nearest-neighbor and linear support-vector-machine classification
- In-memory BMP encoding and decoding
- In-memory ONNX loading, image preprocessing, CPU inference,
  classification, segmentation, object-detection decoding, and non-maximum
  suppression

## Coming Soon

- Installable package releases
- More image codecs
- Video processing and tracking
- Computational photography and image stitching
- Stereo calibration, depth processing, and expanded 3D operations
- More classifiers, clustering, and dimensionality reduction
- Multiple-input and multiple-output neural networks, dynamic tensor shapes,
  and additional model pipelines
- Hardware-accelerated execution

## Samples

The [samples](samples/README.md) directory contains runnable Terlan programs
covering core matrices, image processing, codecs, features, calibration,
machine learning, and DNN inference.

## License

Apache-2.0. See [LICENSE](LICENSE).
