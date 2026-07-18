# Generated Terlan modules

The compiler generates the reviewed `opencv.Mat`, `opencv.Types`,
`opencv.Components`, `opencv.Contours`, `opencv.Features`, `opencv.Imgproc`,
`opencv.Calib3d`, `opencv.Codecs`, `opencv.Ml`, and `opencv.Dnn` modules under
the ignored `generated/src/opencv` workspace.
This directory remains reserved for package-owned Terlan facades that cannot
be expressed by normalized binding metadata; there are no handwritten
substitutes for generated bindings.

`opencv.Dnn` owns immutable preprocessing blobs, bounded in-memory ONNX
networks fixed to OpenCV's CPU backend and target, and immutable inference
outputs. Model bytes, output shapes, and output values cross the boundary by
copy. Output value counts do not copy tensor storage; bounded contiguous slice
projections copy at most 1,048,576 values, truncate at the tail, and permit an
empty slice at the exact end. Fixed RGB and grayscale preprocessing reuse the
same `Blob` owner and publish three-channel or single-channel NCHW values
respectively. Explicit center-crop variants use maintained OpenCV
aspect-resize/crop behavior while
retaining the same scaling, channel conversion, allocation, and ownership
contract. Configurable RGB normalization accepts three finite means and three
positive scales and applies `(RGB / 255 - means) * scales` through OpenCV's
parameterized NCHW preprocessing, with resize, center-crop, and letterbox
variants. Letterbox padding is raw zero before normalization. Fixed RGB and
grayscale letterbox variants use the same parameterized OpenCV
preprocessing to preserve aspect ratio, center the resized image in zero
padding, scale by `1/255`, and return the shared immutable NCHW `Blob` owner.
RGB conversion applies only to BGR input. Configurable grayscale normalization
accepts exactly one finite mean and positive scale and applies
`(Gray / 255 - mean) * scale` through the same immutable owner, with fixed
resize, maintained center-crop, and raw-zero letterbox variants. A single-batch
rank-two output may additionally cross as a copied deterministic
top-one classification record or an independently owned top-k ranking capped
at 10,000 entries. Ranking uses descending finite raw scores and lower-index
tie breaking, and exposes aligned copied index and score lists. Threshold
candidate selection applies an inclusive finite minimum without implicit
normalization and may return an empty owner. Stable-softmax selection computes
global probabilities with subtract-maximum normalization and returns them
through the same independently owned bounded ranking. Stable-sigmoid selection
provides independently normalized probabilities for multi-label logits through
that same owner, with a separate inclusive threshold operation for bounded
candidate selection. Backend
selection, filesystem loading, and borrowed tensor views do not. A rank-four single-batch output can
project one bounded channel into a newly owned grayscale threshold mask or
select the first highest of at most 256 finite classes into a newly owned
grayscale class-index mask. Probability-valued outputs can also project the
highest per-pixel confidence into a newly owned grayscale byte mask without
implicit softmax. Logit-valued outputs use the separate stable-softmax
confidence projection, which normalizes all classes at each pixel before byte
scaling. The companion class-probability projection publishes one selected
class from the same normalization policy, while its binary-mask projection
applies an inclusive finite `[0,1]` threshold. Independent sigmoid probability
projection also accepts one-channel binary segmentation output. Its companion
binary-mask operation applies an inclusive finite `[0,1]` probability
threshold after stable sigmoid normalization. The any-class variant produces
one foreground-union mask across all independently normalized channels, and
the confidence variant byte-scales their maximum probability.
Normalized SSD `[1,1,rows,7]` output can become a bounded `Detections` owner
with copied class-index, confidence, and box projections. Decoding does not
silently apply NMS or a model-specific coordinate transform. Explicit
`non_max_suppression` uses OpenCV's class-aware implementation and returns a
new bounded `Detections` owner without mutating the source.
Generic normalized `[rows,6]` and `[1,rows,6]` output can use
`normalized_xyxy_detections` when fields are ordered as left, top, right,
bottom, confidence, and class index. Class zero is retained, the confidence
threshold is inclusive, and NMS remains an explicit separate operation.
`normalized_center_xywh_detections` accepts the same shapes with center x,
center y, positive width, and positive height fields, validates that converted
corners remain normalized, and returns the same copied XYXY owner.
`normalized_top_left_xywh_detections` provides the corresponding left, top,
positive width, and positive height encoding under the same validation and
ownership contract.
`normalized_objectness_detections` supports normalized center x, center y,
positive width, positive height, objectness, and one or more class-probability
fields in `[rows,5+classes]` and `[1,rows,5+classes]` output. Confidence is the
objectness and highest class-probability product; equal class probabilities
select the lowest index. Activation and NMS remain explicit caller policy.
`normalized_class_score_detections` removes the objectness field and uses the
highest normalized class probability directly as confidence for
`[rows,4+classes]` and `[1,rows,4+classes]` output. It shares the same bounded
decoder, copied result owner, tie rule, coordinate checks, and explicit
activation and NMS policy.
`normalized_transposed_class_score_detections` exposes the field-major
`[4+classes,rows]` and `[1,4+classes,rows]` form through the same decoder. No
temporary transpose, additional owner, activation, or suppression step is
introduced.
`normalized_transposed_objectness_detections` provides the field-major
`[5+classes,rows]` and `[1,5+classes,rows]` objectness form with identical
confidence multiplication, tie selection, coordinate validation, and copied
ownership semantics.

## DNN Expansion Boundary

The current DNN lane is intentionally bounded to one copied image blob, one
in-memory ONNX model, synchronous CPU execution, and one immutable contiguous
output. Generic output access plus classification, segmentation, detection,
and explicit NMS projections are complete for that execution model.

Batching, arbitrary tensors, multiple or named inputs and outputs, dynamic
shapes, cancellation, backend capability selection, graph inspection, and
quantized execution require new ownership or execution contracts. No further
DNN source operation should be added until live metadata reproduction, linked
package execution, and sanitizer execution pass against the pinned OpenCV
checkout.
