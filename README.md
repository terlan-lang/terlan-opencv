# terlan-opencv

`terlan-opencv` is the external OpenCV package for Terlan. OpenCV remains a
package dependency and is never linked into the Terlan compiler.

The CPU-only package surface uses OpenCV `core`, `imgproc`, `imgcodecs`,
`features`, `calib3d`, `ml`, and `dnn`. It owns matrices, ORB, BRISK, AKAZE, SIFT, affine SIFT, KAZE, and MSER analyses
behind opaque Terlan resources and copies all descriptors and region geometry
across the native boundary. The first codec API supports BMP bytes through
generated `opencv.Codecs.decode` and `opencv.Codecs.encode` functions.
Camera geometry includes bounded image and point undistortion plus copied
affine and perspective point transforms over the shared transform records.
Three or four reviewed point correspondences can also produce copied affine or
perspective transforms through `opencv.Calib3d.estimate_affine_transform` and
`opencv.Calib3d.estimate_perspective_transform`, with no native transform
handle retained.
Bounded 3D points can be projected into copied image coordinates through
`opencv.Calib3d.project_points` using reusable copied `Vector3`, camera, and
distortion records.
Six or more matching 3D and 2D observations can produce a copied camera pose
through fixed-policy `opencv.Calib3d.solve_pose` without retaining solver state.
The same correspondence contract supports fixed-policy RANSAC outlier rejection
through `opencv.Calib3d.solve_pose_robust`; inlier indices remain private.
Exactly four matched observations can select OpenCV's lowest-reprojection AP3P
candidate through `opencv.Calib3d.solve_pose_minimal`; alternative candidates
remain adapter-private.
Four or more matched coplanar observations can select the lowest-error IPPE
solution through `opencv.Calib3d.solve_pose_planar`, with ambiguity data kept
inside the adapter.
Four ordered image corners and a positive marker size can select the
lowest-error fixed-IPPE square-marker pose through
`opencv.Calib3d.solve_square_marker_pose`; canonical object corners and all
alternate candidates remain adapter-private.
An explicit copied initial pose can be refined against the same bounded
correspondences through fixed-policy `opencv.Calib3d.refine_pose` or
`opencv.Calib3d.refine_pose_vvs`; OpenCV's mutable refinement vectors remain
adapter-private.
Two copied camera poses can be combined through `opencv.Calib3d.compose_pose`,
which applies the first pose and then the second without exposing derivative
matrices or retaining native transformation state.
Classical machine learning starts with a bounded, CPU-only
`opencv.Ml.KnnModel`. `train_knn_classifier` copies row-major finite samples
and integer class labels into an opaque OpenCV `cv::ml::KNearest` owner. The
adapter fixes classifier mode, brute-force search, and a validated default
neighbor count; callers cannot mutate those policies, retrain the model, load
or save model files, or select KD-tree execution. `predict` copies query rows
and returns copied integer labels while neighbors and distances remain private.
Model dimensions are immutable, disposal is idempotent, stale handles fail,
and live model ownership is observable for lifecycle validation.
The second classical model is a bounded `opencv.Ml.LinearSvmModel`. Training
copies the same row-major samples and categorical integer labels, requires at
least two classes, and fixes OpenCV to `C_SVC`, the `LINEAR` kernel, `C = 1`,
and bounded iteration-plus-epsilon convergence. Prediction returns copied
integer labels. Auto-training, alternate kernels and formulations, class
weights, raw margins, support-vector access, custom kernels, retraining, and
model serialization remain private or unavailable.
The third model is an in-memory `opencv.Ml.NormalBayesModel`. It reuses the
same copied finite sample, categorical label, immutable dimension, prediction,
ownership, and lifecycle contracts. Training requires at least two classes and
uses OpenCV's maintained normal Bayes classifier with no exposed mutable
configuration. Probability matrices and class ordering, incremental updates,
retraining, and model serialization remain private or unavailable.
The fourth model is a bounded `opencv.Ml.LogisticRegressionModel`. Its adapter
copies categorical integer labels into OpenCV's required floating-point
training matrix, preserves the original labels in copied integer predictions,
and fixes training to batch gradient descent, L2 regularization, a `0.001`
learning rate, and 1000 iteration-plus-epsilon steps. Learned coefficients,
raw scores, mini-batch training, mutable policy, retraining, and serialization
remain private or unavailable.
The fifth model is a bounded `opencv.Ml.DecisionTreeModel`. It reuses the
shared copied classifier state and fixes OpenCV DTrees to 16 categories, depth
8, a minimum sample count of 2, no cross-validation or surrogate splits, the
truncated-pruned-tree policy, `0.01` regression accuracy, and no priors.
Tree roots, nodes, splits, subset storage, pruning controls, mutable policy,
retraining, and serialization remain private or unavailable.
The sixth model is a binary `opencv.Ml.BoostModel`. It reuses the shared
classifier owner, requires exactly two copied categorical classes, and fixes
OpenCV to REAL AdaBoost with 32 weak decision stumps, `0.95` weight trimming,
depth 1, and the same bounded tree policy. Raw confidence scores, weak-tree
topology, alternate boosting algorithms, mutable policy, retraining, and
serialization remain private or unavailable.
The seventh model is a binary `opencv.Ml.SgdSvmModel`. The adapter sorts and
copies exactly two categorical labels, maps them to OpenCV's required signed
responses, and maps predictions back to the original labels. Training is fixed
to average SGD, soft margins, OpenCV's pinned optimal coefficients, 100000
count-plus-epsilon steps, and OpenCV's local seed-zero sampler. Learned
weights, shift, raw margins, alternate SGD and margin modes, mutable policy,
retraining, and serialization remain private or unavailable.
The eighth model is a deterministic `opencv.Ml.RandomForestModel`. It fixes
OpenCV RTrees to 32 trees, depth 8, a minimum sample count of 2, default
square-root feature sampling, no variable-importance calculation, and the
same bounded tree policy. Training serializes the brief use of OpenCV's
process-global RNG, installs a package-private seed, and restores the exact
previous state before returning or propagating failure. Votes, out-of-bag
error, variable importance, tree topology, RNG control, mutable policy,
retraining, and serialization remain private or unavailable.
The ninth model is a deterministic `opencv.Ml.MlpClassifierModel`. It sorts
and copies between 2 and 64 original integer classes, encodes symmetric
one-hot targets, and decodes each finite output row by stable argmax. Its
private topology has one bounded hidden layer, symmetric-sigmoid activation,
backpropagation with fixed `0.1` weight and momentum scales, and 1000
count-plus-epsilon steps. Training shares the serialized, state-restoring RNG
lane. Layer configuration, learned weights, raw outputs, alternate activation
and optimization methods, incremental training, retraining, and serialization
remain private or unavailable.
The tenth model is an unsupervised `opencv.Ml.GaussianMixtureModel`. Training
copies finite row-major samples, accepts an immutable cluster count from 2 to
64 with at least two samples per cluster, and fixes expectation-maximization
to diagonal covariance and 100 count-plus-epsilon steps. Prediction validates
finite log likelihoods and normalized posterior rows, then returns only
model-local cluster indexes. Means, weights, covariance matrices, likelihoods,
posterior probabilities, E-step and M-step initialization, mutable policy,
retraining, and serialization remain private or unavailable.
The first DNN surface provides deterministic CPU preprocessing.
`opencv.Dnn.blob_from_bgr_rgb` accepts an owned unsigned-byte BGR
image, resizes without cropping, scales values by `1/255`, swaps channels to
RGB, and stores one immutable NCHW `CV_32F` blob. Shape accessors and copied
finite values are public. `blob_from_grayscale` applies the same resize,
scaling, ownership, and allocation policy to unsigned-byte single-channel
input without channel swapping, producing `[1,1,height,width]` storage.
`blob_from_bgr_rgb_center_crop` and `blob_from_grayscale_center_crop` use
OpenCV's maintained aspect-resize and center-crop path while preserving the
same scaling, channel, allocation, immutability, and `Blob` ownership rules.
`blob_from_bgr_rgb_letterbox` and `blob_from_grayscale_letterbox` preserve the
source aspect ratio, center the resized image within zero padding, scale by
`1/255`, and publish through the same immutable NCHW `Blob` owner. The BGR
variant swaps to RGB; the grayscale variant retains one channel. Both use
OpenCV's maintained DNN letterbox implementation.
`blob_from_bgr_rgb_normalized` and its center-crop and letterbox variants accept
exactly three RGB means and positive scales, convert BGR input to RGB, and
apply `(RGB / 255 - means) * scales` through OpenCV's parameterized NCHW blob
path. Letterbox padding uses raw zero pixels before the same normalization.
All parameters and endpoint results must be finite and float-representable;
the source image remains unchanged. `blob_from_grayscale_normalized` and its
center-crop and letterbox variants apply the corresponding
`(Gray / 255 - mean) * scale` formula using exactly one finite mean and
positive scale. The aspect variants use OpenCV's maintained center-crop and
letterbox modes; letterbox padding is raw zero before normalization. The
bounded inference surface loads up to
128 MiB of
copied in-memory ONNX bytes, fixes OpenCV's maintained backend and CPU target,
and synchronously returns an immutable output owner with copied finite shape
and values. `output_value_count` inspects contiguous tensor size without a
copy, while `output_values_slice` copies a contiguous range capped at
1,048,576 values so callers can inspect large outputs incrementally. Offsets
are checked, tail slices truncate, and the exact end produces an empty list.
Reproducible shape-specific Identity fixtures cover both RGB and
grayscale execution paths. A rank-two `[1, classes]` output can be projected into a copied
top-one `Classification` or a separately owned, bounded top-k ranking. The
package limits class count to one million and top-k results to 10,000,
requires finite raw scores, orders by descending score, and breaks ties by
the lowest class index. `classification_candidates` applies an inclusive
finite raw-score threshold before returning at most 10,000 entries; no match
is a valid empty owner. `softmax_classifications` converts one finite class
row into numerically stable global probabilities before applying the same
10,000-entry ranking bound. `sigmoid_classifications` independently converts
multi-label logits with an overflow-resistant sigmoid before canonical
ranking. `sigmoid_classification_candidates` applies an inclusive probability
floor and may return an empty owner. Ranked indexes and scores cross as aligned
copies.
For rank-four class logits, `softmax_class_confidence_mask` applies stable
per-pixel softmax and publishes the highest global probability as a newly
owned grayscale byte mask. `softmax_class_probability_mask` applies the same
normalization while projecting one caller-selected class, and
`softmax_class_mask` applies an inclusive probability threshold to that class
without admitting single-channel tensors.
`sigmoid_class_probability_mask` independently normalizes one selected class
and supports single-channel binary logits. `sigmoid_class_mask` applies an
inclusive caller-selected probability threshold to the same stable sigmoid
policy and returns a binary grayscale mask. `sigmoid_any_class_mask` publishes
the union of every matching multi-label class and remains equivalent to the
class mask for one-channel binary output. `sigmoid_class_confidence_mask`
publishes the maximum independently normalized class probability per pixel.
One channel from a rank-four
`[1, channels, height, width]` output can also be
strictly thresholded into a newly owned grayscale byte mask. Channel bounds,
finite thresholds and scores, output dimensions, and allocation size are
validated before any mask byte is published.
The same rank-four output can be projected into a per-pixel class-index mask
when it contains two through 256 classes. Every score must be finite; equal
scores retain the lowest class index, which fits directly in the fresh
grayscale matrix's byte storage.
`class_confidence_mask` accepts the same shape when every value is a
probability in `[0,1]`, selects the highest probability per pixel, and rounds
its `0..255` scaling to the nearest byte. Raw logits are rejected rather than
implicitly normalized.
`ssd_detections` decodes only normalized `[1,1,rows,7]` SSD output rows. It
validates every batch, class, confidence, and box field, removes class-zero
background and below-threshold rows, orders results deterministically, and
publishes copied class, confidence, and box projections through a bounded
`Detections` owner. Non-maximum suppression is not implicit;
`non_max_suppression` applies OpenCV's maintained class-aware implementation
with fixed adaptation disabled, deterministic canonical detection ordering, and
an explicit result bound, returning a fresh independent `Detections` owner.
`normalized_xyxy_detections` accepts `[rows,6]` or single-batch `[1,rows,6]`
output ordered as `[left, top, right, bottom, confidence, class_index]`. It
preserves class zero, validates normalized boxes and confidence plus integral
nonnegative class indexes, applies an inclusive confidence threshold, and
orders copied results by descending confidence, ascending class, then source
row. It does not apply NMS or model-specific coordinate conversion.
`normalized_center_xywh_detections` accepts the same shapes ordered as
`[center_x, center_y, width, height, confidence, class_index]`, requires
positive sizes whose converted corners remain in normalized bounds, and
publishes the same copied XYXY representation. Filtering, ordering, class-zero
handling, ownership, and explicit NMS policy are identical.
`normalized_top_left_xywh_detections` supports
`[left, top, width, height, confidence, class_index]` rows under the same
positive-size, normalized-bound, filtering, ordering, ownership, and explicit
NMS rules. All three six-field operations share one internal validator.
`normalized_objectness_detections` accepts `[rows,5+classes]` or single-batch
`[1,rows,5+classes]` rows ordered as center x, center y, width, height,
objectness, then normalized class probabilities. It selects the lowest-index
maximum class probability, multiplies it by objectness, and applies the
inclusive confidence threshold before canonical copied ordering. The operation
does not infer sigmoid or softmax activation and does not apply NMS.
`normalized_class_score_detections` accepts the corresponding objectness-free
`[rows,4+classes]` and `[1,rows,4+classes]` layouts. The maximum normalized
class probability is the confidence, equal probabilities select the lowest
class index, and box conversion, filtering, ownership, activation, ordering,
and NMS policy remain identical.
`normalized_transposed_class_score_detections` accepts field-major
`[4+classes,rows]` and `[1,4+classes,rows]` output under the same policy. It
reads coordinate and class planes directly without allocating or exposing an
intermediate transpose.
Filesystem loading, mutable blobs, backend and target selection, named
multi-output execution, graph inspection, and borrowed tensor storage remain
unavailable.
Borrowed OpenCV buffers, integer type codes, mutable output parameters,
camera devices, filesystems, and GPU state are not public API.

`bindings/opencv.native-binding.json` is the committed expected normalized
adapter snapshot consumed by `terlc bind cpp`. OCV0.2 is not complete until
the maintained Clang LibTooling extractor reproduces its declarations, types,
and source locations exactly; the package does not treat handwritten JSON as
proof of extraction.

The offline metadata gate validates compile-command provenance, declaration
documentation, and exact source locations:

```sh
make opencv-metadata-check
```

The native gate additionally runs `opencv-metadata-live-check`. It invokes the
maintained Clang LibTooling extractor selected by `CPP_METADATA_EXTRACTOR` and
compares reproduced declarations, structured types, and source locations with
the committed binding metadata before any adapter is accepted.

## Pinned upstream

The reviewed surface targets OpenCV `4.13.0` at commit
`fe38fc608f6acb8b68953438a62305d8318f4fcd`. The immutable source identity and
minimal build policy live in `upstream/opencv.lock.toml` and
`cmake/opencv-cpu.cmake`.

The source is deliberately not vendored. Restore the ignored default checkout
as a detached commit and verify it before configuring:

```sh
git clone --filter=blob:none --no-checkout https://github.com/opencv/opencv.git upstream/opencv
git -C upstream/opencv fetch --depth 1 origin fe38fc608f6acb8b68953438a62305d8318f4fcd
git -C upstream/opencv checkout --detach fe38fc608f6acb8b68953438a62305d8318f4fcd
make opencv-source-check
```

An existing checkout can instead be supplied through `OPENCV_SOURCE`. The
source gate applies the same exact commit check before any configure or build
step.

## Surface check

Run the offline contract gate with:

```sh
make opencv-surface-check
```

The gate verifies the upstream pin, module build scope, declaration source
provenance, one-to-one mapping decisions, and complete transitive type policy.
It does not download or build OpenCV.

## Tutorial samples

The `samples` tree ports the supported portions of the OpenCV tutorial index
to runnable Terlan programs. The ports preserve useful upstream step comments
while replacing GUI, camera, and filesystem assumptions with deterministic
in-memory inputs owned by the current package contract.

Compile-check every sample after regenerating the package with:

```sh
make opencv-samples-check
```

The sample inventory and unsupported tutorial categories are tracked in
`samples/README.md`; admission rules for host I/O, UI, stateful algorithms,
graph runtimes, and accelerators are explicit in `samples/CAPABILITIES.md`.

## Promotion reproducibility

Promotion requires clean, committed compiler, package, and pinned OpenCV
repositories. The fast contract gate validates those revision identities and
writes `_build/promotion/opencv-promotion.json` without rebuilding OpenCV:

```sh
make opencv-promotion-inventory-check
```

The complete promotion gate clones only those committed snapshots into a
temporary workspace, rebuilds `terlc`, `terlan-vm`, and the maintained C++
metadata extractor, and runs the external-consumer package check against the
pinned OpenCV source:

```sh
make opencv-fresh-workspace-check
```

The final evidence report sets `fresh_workspace_verified` only after that
consumer run succeeds. Missing HEADs, dirty worktrees, and revision drift fail
before the expensive native build starts.

## Codec policy check

The OCV1 codec foundation permits only copied byte input and output, starts
with required BMP encode/decode support, and forbids ambient filesystem APIs.
The gate verifies the policy, pinned `imgcodecs` build scope, generated native
link plan, and compile/link probe without requiring OpenCV locally:

```sh
make opencv-codec-policy-check
```

The generated codec API is exercised through the native, package-local, and
external-consumer lanes with:

```sh
make opencv-codec-check OPENCV_SOURCE=/path/to/opencv
```

An existing checkout of the pinned commit can be configured separately with:

```sh
make opencv-configure OPENCV_SOURCE=/path/to/opencv
```

## Native adapter check

OCV0.2 generates `opencv.Mat`, `opencv.Types`, `opencv.Components`,
`opencv.Contours`, `opencv.Features`, `opencv.Imgproc`, `opencv.Calib3d`,
`opencv.Ml`, `opencv.Codecs`, and `opencv.Dnn`, compiles their `cxx::bridge`,
links the prepared `core`, `imgproc`, `imgcodecs`, `features`, `calib3d`, `ml`,
and `dnn`
libraries, and runs the native lifecycle contract:

```sh
make opencv-native-adapter-check OPENCV_SOURCE=/path/to/opencv
```

The check builds and installs the pinned OpenCV checkout into the ignored
package workspace. It verifies copied buffers, shape and channel queries,
clone, resize, color conversion, Gaussian blur, ownership-safe crop, bounded
padding, finite flips and right-angle rotations, bounded fixed and automatic
Otsu thresholding,
copied in-memory ONNX loading, fixed CPU identity inference, output ownership,
deterministic top-one, channel-threshold, and class-index projections,
rectangular erosion and dilation, grayscale Canny edge detection, invalid
symbolic policies, bounded min/max normalization, repeated disposal, stale
handles, and a final zero live-object count. The opt-in
`opencv-native-sanitizer-check` target runs the same contract with
AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection on supported
Linux toolchains.

The OCV1.1 geometry expansion can be exercised explicitly with:

```sh
make opencv-geometry-check OPENCV_SOURCE=/path/to/opencv
make opencv-threshold-check OPENCV_SOURCE=/path/to/opencv
make opencv-morphology-check OPENCV_SOURCE=/path/to/opencv
make opencv-edge-check OPENCV_SOURCE=/path/to/opencv
make opencv-normalization-check OPENCV_SOURCE=/path/to/opencv
make opencv-affine-check OPENCV_SOURCE=/path/to/opencv
make opencv-perspective-check OPENCV_SOURCE=/path/to/opencv
make opencv-calib3d-check OPENCV_SOURCE=/path/to/opencv
make opencv-ml-check OPENCV_SOURCE=/path/to/opencv
make opencv-histogram-check OPENCV_SOURCE=/path/to/opencv
make opencv-components-check OPENCV_SOURCE=/path/to/opencv
make opencv-contours-check OPENCV_SOURCE=/path/to/opencv
make opencv-features-check OPENCV_SOURCE=/path/to/opencv
make opencv-arithmetic-check OPENCV_SOURCE=/path/to/opencv
make opencv-mask-check OPENCV_SOURCE=/path/to/opencv
make opencv-range-check OPENCV_SOURCE=/path/to/opencv
make opencv-mask-logic-check OPENCV_SOURCE=/path/to/opencv
make opencv-mask-points-check OPENCV_SOURCE=/path/to/opencv
make opencv-statistics-check OPENCV_SOURCE=/path/to/opencv
make opencv-filter-check OPENCV_SOURCE=/path/to/opencv
```

## Package consumer check

The package gate executes the deterministic BGR-to-grayscale, resize, blur,
codec, crop, pad, flip, rotation, fixed and automatic Otsu/Triangle thresholding,
primitive and compound morphology,
Canny edge,
normalization, histogram equalization and statistics, connected components,
simplified external contours, bounded Shi-Tomasi, FAST, AGAST, ORB, BRISK,
AKAZE, SIFT, affine-invariant SIFT, KAZE, MSER, and simple dark/bright blob detection, cross-checked
ORB, BRISK, and AKAZE Hamming matching, and cross-checked SIFT and KAZE L2 matching,
copied Hu
shape invariants, two-matrix absolute difference, saturated arithmetic,
bounded scaled per-element multiplication and division, element-wise image extrema,
typed grayscale comparison masks,
immutable mask-driven image selection, bounded per-pixel byte-mask blending,
and weighted addition,
copied per-channel mean absolute error,
checked horizontal and vertical image concatenation,
checked image transposition,
bounded image tiling,
checked grayscale channel extraction,
fixed-arity BGR and BGRA channel reconstruction,
immutable checked channel replacement,
inclusive grayscale range masks, semantic image masking and complete Boolean composition,
copied nonzero mask coordinates, copied per-channel means, standard deviations,
nonzero counts, and exact sums, grayscale extrema and integral images, bounded median and bilateral filtering,
bounded squared-difference template matching,
normalized box and stack filtering, adaptive thresholding, absolute Laplacian and
bounded BGR mean-shift filtering, first-order Sobel and Scharr edge responses,
saturated grayscale gradient magnitude, Manhattan/precise Euclidean mask distance transforms,
immutable exact-value flood filling, copied minimal nonzero-mask bounds, and
copied connected-component and nonzero-mask centroids, enclosing circles, convex hulls, bounded hull
approximations, normalized minimum-area boxes, minimum enclosing triangles,
least-squares fitted lines, canonical fitted ellipses, fixed binary
shape-match scores, and bounded canonical probabilistic Hough line segments,
immutable bounded grayscale line, rectangle, circle, cross, rectangle-fill, and circle-fill rasterization, full-layout byte inversion and binary bitwise composition,
copied 256-byte lookup-table transforms,
bounded per-channel scale/absolute-value conversion,
bounded semantic 3x3 sharpening,
affine and perspective transforms, bounded fixed-camera and optimal-camera lens undistortion,
copied bounded image-point undistortion in pixel coordinates, perspective
transform estimation from four point correspondences, and
Gaussian pyramid down/up
sampling pipeline twice: once
from generated package tests and once from
an independent project using a normal local path dependency.

The offline preflight verifies that both lanes compile and reach the expected
native-helper boundary without requiring an OpenCV checkout:

```sh
make terlan-opencv-package-preflight-check
```

The runtime-availability gate starts the package-owned unlinked guard with a
missing OpenCV worker and requires both Terlan lanes to receive the stable
`opencv.unavailable` error:

```sh
make opencv-runtime-unavailable-check
```

```sh
make terlan-opencv-package-check OPENCV_SOURCE=/path/to/opencv
```

`terlc test` resolves the same dependency-first source closure as `terlc build`
and dispatches native operations through the generated package helper.
