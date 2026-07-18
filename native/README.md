# Native adapter

The package-owned C++ wrapper lives in `bindings/cpp`; `terlc bind cpp`
generates the Rust ownership adapter and `cxx::bridge` under the ignored
`generated/native` workspace. Native code must not expose OpenCV pointers,
mutable output parameters, integer type codes, or exception payloads to Terlan
callers.

`runtime_guard` is the unlinked package entry point used by the VM. It proxies
the generated OpenCV-linked worker and converts missing libraries, failed
worker startup, and early worker exit into the stable `opencv.unavailable`
package error.

Generated C++ workers attach a cryptographically random owner identity to every
resource handle. The worker rejects a handle from another process before
looking up its type, ID, or generation, preventing cross-process handle replay.

`opencv_validation.hpp` owns allocation and operation policy independently of
OpenCV. Image dimensions and byte-count multiplication are checked before an
allocation, one owned image is limited to 256 MiB, copied input must match the
exact image size, color conversions must match the source channel layout, and
Gaussian kernels are limited to positive odd dimensions no larger than 255.
Sigma values must be finite and nonnegative.

Fixed DNN preprocessing accepts either BGR input through
`blob_from_bgr_rgb`, which swaps to RGB, or grayscale input through
`blob_from_grayscale`, which preserves its single channel. Both resize without
cropping, scale by `1/255`, produce immutable contiguous NCHW `CV_32F`
storage, and share the checked 256 MiB allocation policy. Their explicit
`blob_from_bgr_rgb_center_crop` and `blob_from_grayscale_center_crop` variants
aspect-resize and center-crop through the same maintained OpenCV operation and
shared package validation without changing source matrices.

The DNN adapter accepts at most 128 MiB of copied ONNX bytes and configures
each loaded network with OpenCV's maintained backend and CPU target. Inference
clones the immutable NCHW input before synchronous execution. Only a finite,
contiguous, single-channel `CV_32F` output fitting the package's 256 MiB owned
storage policy may become an output owner; shape and values are copied before
they reach Terlan. Fixed protobuf-generated RGB and grayscale Identity graphs
exercise both accepted input channel layouts. Classification accepts only
`[1, classes]` output with two through one million finite scores. Top-one
returns a copied record. Bounded top-k returns an independent owner containing
at most 10,000 entries with copied, aligned index and raw-score projections.
`classification_candidates` reuses that owner with an inclusive finite
raw-score threshold and permits an empty result. Both collection operations
order by descending score and retain the lower index on equal scores.
Channel-mask projection accepts only `[1, channels, height, width]` output,
one in-range channel, and a finite threshold. It validates the selected plane
before strictly thresholding scores into `0` or `255` bytes owned by a fresh
grayscale `OpenCvMat`.
Class-index projection accepts the same rank-four layout with two through 256
classes, validates every score before writing output, and selects the first
highest class per pixel into a fresh grayscale `OpenCvMat`.
Class-confidence projection additionally requires every tensor value in
`[0,1]`, selects the highest probability per pixel, and scales it to the
nearest `0..255` byte. It does not apply softmax or accept raw logits.
SSD detection projection accepts only normalized `[1,1,rows,7]` output with
batch index zero, integral bounded class indexes, probabilities and box
coordinates in `[0,1]`, ordered corners, at most 100,000 source rows, and at
most 10,000 selected rows. Class zero is background. Accepted rows are sorted
by descending confidence, class index, and source row before copied parallel
projections are published. Class-aware NMS is an explicit immutable operation
over this owner. It uses OpenCV `NMSBoxesBatched` with fixed `eta = 1`, no
OpenCV top-k truncation, normalized `Rect2d` boxes, canonical detection order,
and package-owned result truncation. It returns a fresh collection and leaves
the source untouched. Model-specific coordinate transforms remain outside the
package contract.

Crop regions are bounds-checked without adding coordinates, then cloned into
fresh ownership so no OpenCV ROI view crosses the boundary. Padding dimensions
use checked addition, the same 256 MiB output limit, and finite constant,
replicate, or reflect policies before `copyMakeBorder` executes.
Finite flip axes and right-angle rotations also return fresh ownership. Their
output dimensions are validated against the same allocation budget before the
OpenCV operation executes.

Fixed threshold operations accept only reviewed modes and finite scalar values
from 0 through 255. Results retain the source shape and channel layout in a new
owner. Automatic and neighborhood-derived thresholds use the narrower
grayscale contracts documented below.

Erosion and dilation share one native morphology path. They use rectangular
odd kernels no larger than 255, one through 64 iterations, and replicate-border
behavior; each operation preserves shape and returns a new owner.

Compound morphology exposes opening, closing, gradient, top-hat, and black-hat
as symbolic operations over the same rectangular-kernel and iteration bounds.
The adapter invokes OpenCV `morphologyEx` once with replicate borders, leaves
the source unchanged, preserves its public unsigned-byte layout, and returns a
fresh matrix owner.

Gaussian pyramid downsampling produces dimensions `(source + 1) / 2`, while
upsampling produces dimensions exactly twice the source. Both operations use
OpenCV's fixed Gaussian pyramid kernel and required default reflected border,
preserve the public unsigned-byte layout, leave the source unchanged, and
return fresh owners. Upsampling rejects any doubled shape that would overflow
OpenCV dimensions or the package-wide 256 MiB allocation bound.

Canny edge detection accepts only one-channel matrices, ordered finite
thresholds no larger than 65,535, Sobel apertures 3/5/7, and an explicit L1/L2
gradient choice. Its output is a fresh one-channel owner with the source shape.

Min/max normalization preserves the source shape and channel layout while
mapping values into an ordered finite range from 0 through 255. Equal,
reversed, non-finite, and out-of-range bounds are rejected before OpenCV runs.

Histogram equalization accepts only valid single-channel unsigned-byte images,
preserves source dimensions, and returns a fresh matrix owner. Color images
must be converted explicitly before equalization.

Contrast-limited adaptive histogram equalization exposes
`opencv.Imgproc.adaptive_equalize_hist(image, clip_limit, tiles_x, tiles_y)`
for bounded single-channel unsigned-byte images. The clip limit must be finite
and greater than zero through 256. Tile counts are one through 64 and cannot
exceed their corresponding image dimensions. The adapter creates a transient
OpenCV CLAHE implementation, applies it into fresh matching storage, and
retains neither the equalizer nor temporary output across the boundary. Color
input, mutable destinations, unbounded tile grids, and a public CLAHE resource
remain private.

Harris corner response exposes
`opencv.Imgproc.harris_response(image, block_size, aperture_size, response_k)`
for bounded single-channel unsigned-byte images. The neighborhood is an odd
size from three through 31 that fits the image, the derivative aperture is
3/5/7, and the finite positive response coefficient is at most 0.25. OpenCV's
`BORDER_REFLECT_101` policy is fixed by the package. The transient `CV_32F`
matrix remains private; finite values cross as a copied row-major
`List[Float]`, leaving the source unchanged and creating no native result
owner.

Minimum-eigenvalue corner response exposes
`opencv.Imgproc.corner_min_eigenvalue(image, block_size, aperture_size)` under
the same grayscale, copied-analysis, odd-neighborhood, derivative-aperture,
and fixed reflected-border policy. Finite row-major values are copied from a
private `CV_32F` result; insignificant negative roundoff is normalized to
zero, while materially negative values are rejected as a native contract
failure. Harris, minimum-eigenvalue, precise distance, and template-match
results share one native shape/type/finiteness copy path.

Pre-corner derivative response exposes
`opencv.Imgproc.pre_corner_response(image, aperture_size)` for the same
bounded grayscale analysis images and derivative apertures 3/5/7. The package
fixes `BORDER_REFLECT_101`, retains the transient `CV_32F` matrix privately,
and returns finite signed row-major `List[Float]` values. The operation leaves
the source unchanged and creates no native result owner.

Corner eigensystem analysis exposes
`opencv.Imgproc.corner_eigensystem(image, block_size, aperture_size)` under the
same bounded neighborhood, derivative-aperture, and reflected-border policy.
Each row-major pixel contributes six copied values: maximum eigenvalue,
minimum eigenvalue, maximum-eigenvalue vector `x/y`, and minimum-eigenvalue
vector `x/y`. The adapter requires finite scalars and ordered nonnegative
eigenvalues, normalizes insignificant negative roundoff to zero, and keeps the
private six-channel `CV_32F` tensor out of the resource model.

Circle detection exposes `opencv.Imgproc.detect_circles(image,
minimum_distance, edge_threshold, center_threshold, minimum_radius,
maximum_radius)` for bounded grayscale analysis images. The package fixes
OpenCV `HOUGH_GRADIENT` with a unit inverse-resolution ratio, requires finite
positive thresholds and separation plus ordered positive bounded radii, and
caps results at 4,096 circles. Results cross as canonical row-major
`[center_x, center_y, radius, ...]` values sorted by center row, center column,
and radius. The source remains unchanged and native detector containers do not
cross the boundary.

Standard line detection exposes
`opencv.Imgproc.detect_lines(image, vote_threshold)` for bounded grayscale
analysis images. The adapter normalizes nonzero bytes to binary, fixes OpenCV
accumulator resolution to one pixel and one degree, and caps results at 4,096
lines. It validates every finite bounded result before sorting and copies
canonical `[rho, theta, ...]` pairs ordered by theta then rho. The source
remains unchanged; Hough method constants, accumulator storage, and native
vectors remain private.

Histogram statistics use OpenCV's 256-bin unsigned-byte calculation and cross
the boundary as an ordinary copied `List[Int]`, not another native resource.
Inputs are grayscale-only and limited to 16,777,216 pixels so every OpenCV
floating-point counter remains an exact integer. The adapter verifies all 256
bins and their total before returning them to the VM.

Per-channel means and population standard deviations use OpenCV's statistical
operations over one-, three-, or four-channel unsigned-byte images. Results
cross as copied `List[Float]` values in channel order, contain exactly the
source channel count, and are validated as finite values from 0 through 255.
Neither operation retains a native result, and both inherit the package-wide
256 MiB owned-image input bound.

Per-channel nonzero counts use OpenCV `extractChannel` and `countNonZero` over
the same gray, BGR, and BGRA layouts. `opencv.Imgproc.non_zero_counts` returns
one copied `Int` per channel in channel order, validates every result against
the source pixel count, supports the complete package-owned image-size policy,
leaves the source unchanged, and retains no extracted channel or result.

Checked channel extraction exposes that maintained OpenCV `extractChannel`
path as `opencv.Imgproc.extract_channel`. The zero-based channel index must be
within the source's gray, BGR, or BGRA channel count. The operation preserves
rows and columns, leaves the source unchanged, and returns a fresh
single-channel unsigned-byte matrix owner. Mutable channel insertion, broad
split/merge arrays, and OpenCV channel container types remain private.

Fixed-arity channel reconstruction complements extraction without publishing a
variable native array. `opencv.Imgproc.merge_bgr` accepts blue, green, and red
grayscale owners; `opencv.Imgproc.merge_bgra` additionally accepts alpha. All
inputs must have identical dimensions and unsigned-byte grayscale types, and
the resulting BGR or BGRA allocation must fit the 256 MiB limit. OpenCV
`merge` receives only package-owned fixed arrays, every input remains
unchanged, aliases are valid, and the result receives fresh ownership.
Arbitrary channel shuffling and mutable insertion remain private.

Immutable channel replacement exposes the maintained OpenCV `insertChannel`
operation as `opencv.Imgproc.replace_channel`. The replacement must be a
same-size unsigned-byte grayscale owner and the zero-based index must identify
a gray, BGR, or BGRA destination channel. The adapter clones the target before
privately inserting the replacement, so both inputs remain unchanged and the
result is a fresh owner with the target's exact layout and private element
type. A grayscale owner may replace its own channel zero because insertion
targets the clone. Mutable channel insertion is not part of the public API.

Per-channel sums use OpenCV `sum` over the same unsigned-byte layouts.
`opencv.Imgproc.sum_channels` validates each native scalar as finite,
nonnegative, integral, and no larger than `pixels * 255` before copying it to
`List[Int]`. The package allocation limit keeps every possible channel sum
exact in OpenCV's double representation. The source remains unchanged and no
OpenCV scalar representation crosses the boundary.

Grayscale extrema use OpenCV `minMaxLoc` without requesting pixel locations.
`opencv.Imgproc.min_max_values` returns exactly `[minimum, maximum]` as a
copied `List[Float]`, validates ordered finite values in the unsigned-byte
domain, rejects color images, retains no native result, and inherits the same
owned-image input bound.

Grayscale integral analysis uses OpenCV `integral` with a private signed
32-bit accumulator. `opencv.Imgproc.integral_values` accepts at most 1,048,576
unsigned-byte pixels, so the maximum accumulated sum remains exact. It returns
the complete `(rows + 1) x (columns + 1)` result as a copied row-major
`List[Int]`, including the zero top row and left column. Color and oversized
inputs are rejected, the source remains unchanged, and no analysis resource is
retained.

Normalized box filtering accepts one-, three-, or four-channel unsigned-byte
images and independently bounded odd kernel dimensions from 1 through 255.
The adapter uses a centered anchor and explicit replicate borders, preserves
shape and private element type, leaves the source unchanged, and returns a
fresh matrix owner.

Stack blur accepts the same public unsigned-byte layouts and independently
bounded odd kernel dimensions. It uses OpenCV's fixed replicate-border
algorithm, preserves shape and private element type, leaves the source
unchanged, and returns a fresh owner. Kernel size one is an exact identity.

Pyramid mean-shift filtering accepts only unsigned-byte BGR images, finite
positive spatial and color radii no greater than 65,535, and pyramid levels
from zero through eight. Termination criteria remain package-fixed. The
operation preserves shape and BGR layout, leaves the source unchanged, and
returns a fresh owner.

Median filtering accepts one-, three-, or four-channel unsigned-byte images
and one odd square kernel from 3 through 255. OpenCV's median filter uses its
defined replicate-border behavior; the adapter validates the input and kernel
first, preserves shape and private element type, leaves the source unchanged,
and returns a fresh matrix owner.

Bilateral filtering accepts grayscale or BGR unsigned-byte images, one odd
diameter from 1 through 255, and finite positive color and spatial sigmas no
greater than 65,535. The adapter selects replicate borders explicitly,
preserves shape and private element type, leaves the source unchanged, and
returns a fresh matrix owner. BGRA images and implicit or unbounded filter
parameters are rejected.

Adaptive thresholding accepts grayscale unsigned-byte images, mean or
Gaussian neighborhood calculations, binary or inverse output, odd block sizes
from 3 through 255, maximum output values from 0 through 255, and finite
constants from -255 through 255. Other fixed-threshold modes and color images
are rejected. The source remains unchanged and each result has fresh ownership
with the same shape and private element type.

Automatic Otsu thresholding accepts grayscale unsigned-byte images, binary or
inverse output, and maximum output values from 0 through 255. OpenCV derives
the global threshold from the source histogram; the package keeps the
`THRESH_OTSU` flag and computed scalar private. The adapter validates the
selected scalar, preserves the source, and returns a fresh shape-preserving
grayscale owner. Color images and nonbinary threshold modes are rejected.

Automatic Triangle thresholding exposes
`opencv.Imgproc.triangle_threshold(image, maximum_value, mode)` through the
same grayscale, maximum-value, binary-mode, source-preservation, and
fresh-owner contract as Otsu. Both operations share one private automatic
threshold execution path that validates OpenCV's selected scalar before
accepting the shape-preserving result. The package fixes `THRESH_TRIANGLE`
privately; dry-run threshold selection, numeric flags, color images, and
nonbinary threshold modes remain outside the public API.

Laplacian edge response accepts grayscale unsigned-byte images and derivative
apertures 1, 3, 5, or 7. The adapter computes into a signed 16-bit intermediate,
converts the absolute response to unsigned bytes, selects replicate borders,
and returns a fresh shape-preserving owner. Color images and all other
apertures are rejected before OpenCV execution.

Sobel edge response accepts grayscale unsigned-byte images, exactly one
first-order x or y derivative, and apertures 1, 3, 5, or 7. The adapter uses
signed 16-bit intermediate values, returns their absolute unsigned-byte
response with the source shape, selects replicate borders explicitly, leaves
the source unchanged, and returns a fresh owner.

Scharr edge response applies OpenCV's fixed higher-accuracy first-order kernel
to one selected x or y derivative. It shares Sobel's grayscale input, signed
intermediate, absolute unsigned-byte output, replicate-border, source
preservation, and fresh-owner contracts without exposing a kernel parameter.

Gradient magnitude composes two same-size grayscale unsigned-byte component
images through `opencv.Imgproc.gradient_magnitude(horizontal, vertical)`.
The adapter privately converts both components to single-precision storage,
uses OpenCV `magnitude` to calculate `sqrt(horizontal^2 + vertical^2)`, and
saturates the result back into a fresh grayscale byte owner. Inputs remain
unchanged and may identify the same matrix. Color images, floating-point matrix
types, output-depth controls, angle output, and mutable destinations remain
private.

Manhattan distance transformation exposes
`opencv.Imgproc.manhattan_distance_transform(mask)` for unsigned-byte
grayscale masks. Every nonzero pixel becomes its L1 distance to the nearest
zero pixel, while zero pixels remain zero. The adapter fixes OpenCV `DIST_L1`,
the required three-pixel mask, and `CV_8U` output privately, so distances above
255 saturate and the result remains a composable fresh grayscale owner. The
source remains unchanged; color input, labels, floating-point output, raw
distance policies, and mutable destinations remain private.

Precise Euclidean distance transformation exposes
`opencv.Imgproc.euclidean_distance_transform(mask)` as copied row-major
`List[Float]` values. The unsigned-byte grayscale mask must contain at least
one zero pixel and is capped at 1,048,576 pixels. The adapter privately uses
OpenCV `DIST_L2`, `DIST_MASK_PRECISE`, and `CV_32F`, verifies the result shape
and every finite nonnegative distance against the image's geometric maximum,
then copies values without retaining the temporary matrix. The source remains
unchanged; color input, all-foreground masks, labels, approximate modes, raw
distance constants, and public floating-point matrices remain private.

Immutable exact-value flood filling exposes
`opencv.Imgproc.flood_fill(image, seed_x, seed_y, fill_value, connectivity)`.
The seed must be in bounds, the fill value is one byte applied uniformly to
every channel, and connectivity is the existing typed four/eight policy. The
adapter clones gray, BGR, or BGRA input before privately invoking OpenCV
`floodFill` with zero lower and upper differences, so only pixels exactly
connected to the seed value are replaced and the source remains unchanged.
OpenCV's mutable image and mask parameters, tolerance scalars, packed flags,
filled rectangle, and in-place result remain private.

Copied nonzero-mask bounds expose `opencv.Imgproc.non_zero_bounds(mask)` for
unsigned-byte grayscale masks capped at 1,048,576 pixels. The adapter invokes
OpenCV `boundingRect` over the grayscale image and returns either an empty
`List[Int]` or exactly `[x, y, width, height]` for the minimal axis-aligned
rectangle containing every nonzero pixel. Native rectangle coordinates and
extent are validated against the source dimensions before copying. The source
remains unchanged and no result resource is retained. Point-set overloads,
rotated rectangles, rectangle drawing, and mutable destinations remain private.

Copied nonzero-mask centroids expose
`opencv.Imgproc.non_zero_centroid(mask)` under the same bounded unsigned-byte
grayscale policy. OpenCV moments use binary-image weighting, so every nonzero
pixel contributes equally regardless of byte value. An empty mask returns an
empty `List[Float]`; otherwise the result is exactly `[x, y]`. The adapter
validates finite and integral foreground mass, finite first moments, and
in-image centroid coordinates before copying. The source remains unchanged and
no result resource is retained. Weighted-intensity moments, the native
`cv::Moments` representation, and mutable outputs remain private.

Copied minimum enclosing circles expose
`opencv.Imgproc.non_zero_enclosing_circle(mask)` under the same bounded
unsigned-byte grayscale policy. The adapter shares one private validated
`findNonZero` collection path with copied point extraction, then invokes
OpenCV `minEnclosingCircle` without publishing the point vector or native
geometry types. Empty masks return an empty `List[Float]`; one nonzero pixel is
normalized to `[x, y, 0.0]`; all other results are copied as
`[center_x, center_y, radius]`. The adapter validates finite in-image centers
and a finite nonnegative radius against the source diagonal. The source remains
unchanged and no result resource is retained. Mutable center/radius outputs,
caller-provided point sets, and circle drawing remain private.

Copied nonzero-mask convex hulls expose
`opencv.Imgproc.non_zero_convex_hull(mask)` under the same bounded
unsigned-byte grayscale policy and shared validated `findNonZero` collection
path. The adapter invokes OpenCV `convexHull` in its default point-returning
orientation, rejects malformed native points, and rotates the result so the
topmost then leftmost vertex is first. Empty masks return an empty `List[Int]`;
otherwise coordinates are flattened as `[x, y, ...]`. Every copied vertex is
validated as unique, in bounds, and nonzero in the source mask, while hulls of
three or more points must remain convex. The source remains unchanged and no
result resource is retained. Caller-provided point sets, hull indices,
clockwise mode, native point vectors, and mutable outputs remain private.

Bounded nonzero-mask hull approximation exposes
`opencv.Imgproc.non_zero_approximate_hull(mask, epsilon)` through the same
bounded grayscale, nonzero-point, convex-hull, and canonical-order policies.
The finite nonnegative epsilon is measured in pixels and cannot exceed the
source diagonal. Empty masks return an empty `List[Int]`; point and collinear
hulls retain their canonical vertices; area-bearing hulls are simplified with
OpenCV `approxPolyDP` in closed-curve mode. Results are flattened as
`[x, y, ...]` after validating nonempty output, source membership, uniqueness,
bounds, and convexity. Zero epsilon preserves the canonical hull, the source
remains unchanged, and no result resource is retained. Arbitrary caller curves,
open-curve approximation, concave contour simplification, native output arrays,
and mutable destinations remain private.

Normalized nonzero-mask minimum-area boxes expose
`opencv.Imgproc.non_zero_min_area_box(mask)` under the same bounded grayscale
policy and shared point collector. OpenCV `minAreaRect` computes the geometry
and `boxPoints` produces clockwise vertices, but the adapter does not publish
the native center, size, or unstable angle convention. Empty masks return an
empty `List[Float]`, one point returns `[x, y]`, collinear masks return two
canonical endpoints, and area-bearing masks return four clockwise vertices
flattened as `[x, y, ...]`. Vertices are rotated so the topmost then leftmost
appears first, near-zero artifacts are normalized to zero, and finite
coordinates are bounded by a source-diagonal envelope because OpenCV permits
border-adjacent rotated boxes to extend outside the source matrix. Four-point
results must remain convex and contain every source point within native
precision. The source remains unchanged and no result resource is retained.
Native `RotatedRect`, angle conventions, caller point sets, and mutable outputs
remain private.

Normalized nonzero-mask minimum enclosing triangles expose
`opencv.Imgproc.non_zero_enclosing_triangle(mask)` through the same bounded
grayscale and shared point-collection policy. Empty masks return an empty
`List[Float]`, one point returns `[x, y]`, and collinear masks return two
canonical endpoints. Area-bearing masks invoke OpenCV `minEnclosingTriangle`
and return three vertices in the default convex-hull orientation, rotated so
the topmost then leftmost vertex appears first. The adapter validates the
reported area against the copied contour, caps it by the enclosing source
rectangle, requires finite source-bounded coordinates and convexity, and
confirms that every source point is enclosed within native precision.
Near-zero artifacts are normalized to zero, the source remains unchanged, and
no result resource is retained. Native output arrays, returned area values,
caller point sets, and mutable outputs remain private.

Canonical nonzero-mask line fitting exposes
`opencv.Imgproc.non_zero_fit_line(mask)` through the same bounded grayscale and
shared point-collection policy. Fewer than two nonzero pixels return an empty
`List[Float]`; otherwise OpenCV `fitLine` runs with fixed L2 distance and the
documented `0.01` radius and angle tolerances. The copied result is
`[vx, vy, x0, y0]`, where the direction is normalized and its sign is selected
toward positive x, or positive y for a vertical line, and the anchor remains
inside the source dimensions. The adapter validates finite values, unit-vector
length, and a source-diagonal residual envelope before normalizing near-zero
artifacts. The source remains unchanged and no result resource is retained.
Alternative distance functions, estimator parameters, native vectors, caller
point sets, 3D fitting, and mutable outputs remain private.

Canonical nonzero-mask ellipse fitting exposes
`opencv.Imgproc.non_zero_fit_ellipse(mask)` through the bounded grayscale and
shared point-collection policy. Fewer than five points and collinear point sets
return an empty `List[Float]`; otherwise OpenCV `fitEllipse` performs the fixed
least-squares fit. The copied result is
`[center_x, center_y, major_diameter, minor_diameter, angle_degrees]`. Axes are
ordered major-first, the major-axis angle is normalized to `[0, 180)`, and a
near-circular fit has canonical angle zero. The adapter validates finite,
positive geometry against source-diagonal coordinate and diameter envelopes,
then normalizes near-zero artifacts. The source remains unchanged and no result
resource is retained. AMS and Direct fitting, native `RotatedRect`, caller
point sets, mutable outputs, and unstable width/height angle conventions remain
private.

Lookup-table transformation accepts every public unsigned-byte image layout and
exactly 256 copied bytes. The one-channel table is applied independently to
every gray, BGR, or BGRA channel through `cv::LUT`; the adapter rejects short or
oversized tables before native execution, leaves the source and caller-owned
table unchanged, and returns a fresh shape-preserving owner.

Scale/absolute-value conversion accepts every public unsigned-byte image layout
and finite scale and offset values bounded to plus or minus 65,535. Each channel
is transformed independently through `cv::convertScaleAbs`, including absolute
value, unsigned-byte saturation, and rounding. The source remains unchanged and
the shape-preserving result has fresh ownership.

Semantic sharpening accepts every public unsigned-byte image layout and a
finite amount from zero through 16. The package owns a centered 3x3 cross kernel,
uses `cv::filter2D` with source-depth output and replicate borders, and keeps
anchor, depth, border, and raw-kernel constants private. Zero is identity;
positive amounts sharpen each channel independently with unsigned-byte
saturation. The source remains unchanged and the result has fresh ownership.

Connected-component analysis is a second uniquely owned native resource. One
bounded OpenCV `connectedComponentsWithStats` call retains labels, stats, and
centroids; callers copy row-major labels, foreground areas, flattened bounds,
and flattened foreground centroid pairs only when needed. Inputs are
grayscale-only, connectivity is limited to four or eight
neighbors, and label output is capped at 1,048,576 pixels. Matrix and analysis
live counts remain independent. Centroid shape, finiteness, bounds, and values
are validated against label-derived coordinate sums before ownership.

External-contour analysis is a third uniquely owned native resource. It uses
OpenCV `findContours` with external retrieval and simple chain approximation,
without publishing OpenCV integer constants. Grayscale input and total copied
points are capped at 1,048,576 pixels. Callers receive flattened point pairs,
cumulative point offsets, flattened bounding rectangles, contour areas, and
closed perimeters, while the source matrix remains independently owned and
unchanged. Floating-point metrics cross as copied `List[Float]` values and do
not retain native vector storage. Convex-hull areas are computed once during
analysis, validated as finite values no smaller than their corresponding
contour areas, and copied in the same contour order. Empty and degenerate
contours remain explicit as empty projections or zero area rather than
sentinel values. `opencv.Contours.solidities(contours)` derives normalized
area-to-convex-area ratios from the retained values without another OpenCV
call or duplicate native storage. Results remain in `[0.0, 1.0]`, with
degenerate zero-area hulls normalized to `0.0`.
`opencv.Contours.circularities(contours)` similarly derives normalized
`4 * pi * area / perimeter^2` values from retained metrics, using `0.0` for
zero-perimeter contours and retaining no additional native state.
`opencv.Contours.extents(contours)` derives contour-area to axis-aligned
bounding-box-area ratios from the retained bounds. Contour construction
validates that native areas fit those bounds, and copied extents remain in
`[0.0, 1.0]` without additional state or OpenCV execution. Geometric contour
centroids are computed once from polygon moments and retained in the same
owner; zero-area contours use the arithmetic mean of their retained vertices.
`opencv.Contours.centroids(contours)` copies finite, bounds-validated `x, y`
pairs in contour order. `opencv.Contours.hu_moments(contours)` lazily computes
and copies the seven raw Hu invariants for each retained contour. Values remain
in contour order, must be finite, and do not expose `cv::Moments`; the adapter
caps the combined copied scalar count through the existing analysis bound and
retains no additional native state. `opencv.Contours.minimum_area_boxes`
reuses the package's validated minimum-area-box geometry and returns exactly
four canonical `x, y` corners per contour. The topmost then leftmost corner is
first; point contours repeat one corner four times and line contours emit both
endpoints twice, preserving an eight-scalar record for every contour. Native
`RotatedRect` angles remain private, output is bounded by the existing copied
analysis limit, and no box resource or duplicate geometry is retained.
`opencv.Contours.minimum_enclosing_circles` similarly reuses the validated
nonzero-mask circle primitive and copies `[center_x, center_y, radius]` for each
contour. Point contours normalize to zero radius; all centers and radii must be
finite and source-bounded, combined output is capped, and no native circle
state is retained. `opencv.Contours.minimum_enclosing_triangles` reuses the
validated mask-triangle primitive and emits exactly three canonical `x, y`
vertices per contour. Point contours repeat their vertex three times and line
contours repeat the second endpoint, preserving a six-scalar record. Native
area outputs and triangle arrays remain private, copied values are bounded by
the analysis cap, and no triangle state is retained.
`opencv.Contours.fitted_lines` reuses the canonical L2 line fitter and copies
`[vx, vy, x0, y0]` per contour. Multi-point directions are unit length with a
deterministic sign and a source-bounded anchor; point contours use
`[0.0, 0.0, x, y]` because no direction exists. Output remains fixed at four
scalars per contour, bounded by the analysis cap, and retains no fitted-line
state. `opencv.Contours.fitted_ellipses` reuses the canonical least-squares
ellipse eligibility and execution policy and copies
`[center_x, center_y, major_diameter, minor_diameter, angle_degrees]` per
contour. Eligible contours have positive ordered axes and an angle in
`[0, 180)`; ineligible point, line, collinear, or too-small contours retain
their validated centroid and use zero axes and angle. The fixed five-scalar
records are capped and retain no ellipse state.
`opencv.Contours.match_scores(contours, reference_index)` compares every
retained contour against one validated contour index and copies fixed
`[I1, I2, I3]` records in contour order. It reuses the same fixed OpenCV
methods, zero parameter, finite nonnegative validation, and near-zero
normalization as binary-mask shape matching. The reference record is therefore
three zeros, invalid or empty indexes fail before matching, copied output is
capped, and no comparison state is retained.
`opencv.Contours.approximate(contours, epsilon)` applies OpenCV's closed
Douglas-Peucker polygon approximation to every retained contour and returns a
new uniquely owned `Contours` resource. The source owner remains unchanged.
The new owner recomputes and validates bounds, areas, perimeters, convex areas,
and centroids through the same construction path as contour extraction, so no
derived projection can become stale. Epsilon is finite, nonnegative, and
bounded by the retained source-image diagonal. Zero tolerance preserves exact
points; point, line, and empty analyses remain valid independent owners.
`opencv.Contours.convex_hulls(contours)` applies OpenCV convex-hull extraction
to each retained contour and returns another uniquely owned `Contours`
resource. Concave geometry is replaced by its hull while contour ordering is
preserved. The source remains unchanged, and the new owner's area equals its
recomputed convex area. Point, line, convex, and empty inputs preserve their
degenerate geometry without sentinel records or borrowed native storage.
`opencv.Contours.convexity_defects(contours, contour_index)` inspects one
validated retained contour and copies fixed
`[start_x, start_y, end_x, end_y, farthest_x, farthest_y, depth]` records.
Coordinates refer to retained source pixels and depth is converted from
OpenCV's private fixed-point representation into bounded pixel units. Convex,
point, and line contours return an empty list; empty or out-of-range indexes
fail before native execution. Hull indexes and native `Vec4i` records never
cross the package boundary.
`opencv.Contours.signed_distances(contours, x, y)` evaluates one finite point
inside the retained source-image pixel-center bounds against every contour.
Each copied distance is positive inside its polygon, zero on the boundary, and
negative outside, with magnitude bounded by the source diagonal. Point and
line degenerates retain OpenCV's geometric distance behavior, empty analyses
return an empty list, and the source contour owner remains unchanged. Native
`Point2d` values and `pointPolygonTest` flags remain adapter-private.
`opencv.Contours.filled_mask(contours, value)` closes the analysis-to-image
round trip by rendering every retained external contour into a fresh
source-sized grayscale matrix. The fill value is an integer from zero through
255; the background is zero, and OpenCV's filled, eight-connected line policy
is fixed by the adapter. Rectangle, point, and line contours rasterize without
special sentinel forms, empty analyses produce an all-zero matrix, and the
returned `Mat` remains independently owned after its contour source is
disposed.
`opencv.Contours.outline_mask(contours, value, thickness)` renders every
retained contour with a fixed eight-connected line policy. The positive
thickness is bounded by the shared drawing-style validator, while the value
uses the same zero-through-255 contract as filled masks.
`opencv.Contours.selected_filled_mask(contours, contour_index, value)` renders
exactly one validated contour and rejects empty or out-of-range selections.
Both operations return fresh source-sized grayscale matrices whose ownership
and lifetime are independent of the contour analysis.

Shi-Tomasi feature detection uses OpenCV `goodFeaturesToTrack` directly on a
live grayscale matrix and returns flattened `x, y` coordinates as a copied
`List[Float]`. Inputs are capped at 1,048,576 pixels and 4,096 requested
corners. Quality must be finite and greater than zero through one, and minimum
distance must be finite, nonnegative, and no larger than one million. The
adapter verifies output count, finiteness, and image bounds before copying.
`opencv.Imgproc.fast_keypoints(image, threshold, nonmax_suppression,
max_keypoints)` uses OpenCV's reviewed `features` module with a fixed
`TYPE_9_16` detector. Grayscale inputs remain capped at 1,048,576 pixels,
thresholds are positive bytes, and copied output is capped at 4,096 points.
Candidates are ranked by response before truncation and returned as unique,
row-major flattened `x, y` coordinates. Native `cv::KeyPoint` values never
cross the package boundary.

`opencv.Imgproc.agast_keypoints(image, threshold, nonmax_suppression,
max_keypoints)` applies the same copied-coordinate, response-ranked, bounded
output contract to OpenCV's maintained AGAST detector. The neighborhood is
fixed to `OAST_9_16`; thresholds are positive bytes and output is capped at
4,096 points. FAST and AGAST share native validation, canonicalization, and
projection code while remaining explicit operations with no detector object
crossing the package boundary.

`opencv.Features.detect_orb(image, max_features)` creates one uniquely owned
analysis containing both canonical keypoints and their row-aligned ORB
descriptors. The grayscale image is capped at 1,048,576 pixels and the request
at 4,096 features. The adapter fixes the ORB pyramid, patch, score, and FAST
threshold policy, validates every returned keypoint and 32-byte descriptor
row, then sorts both through the same permutation. `keypoints` copies flattened
`x, y` pairs, while `descriptors` copies row-major bytes; neither native
`cv::KeyPoint` nor `cv::Mat` descriptor storage crosses the boundary. The
analysis remains valid after its source matrix is disposed and has an explicit
`dispose` and `live_count` lifecycle contract.

`opencv.Features.detect_brisk(image, threshold, octaves, max_features)`
creates a distinct owned BRISK analysis over the same private binary-feature
representation. Thresholds are positive bytes, octaves range from zero through
eight, and output is capped at 4,096 rows. The adapter uses OpenCV's maintained
BRISK implementation with a fixed `1.0` pattern scale, verifies its unsigned
byte Hamming descriptor contract, ranks detections by response before bounded
descriptor computation, and canonicalizes each keypoint with its 64-byte row.
The `brisk_keypoints`, `brisk_descriptor_width`, and `brisk_descriptors`
projections are copied and remain valid after source disposal. A separate
`BriskFeatures` type prevents accidental use with the ORB-only matcher.

`opencv.Features.match_brisk(query, train, max_matches)` applies the same
private cross-checked `NORM_HAMMING` engine and canonical copied projections to
BRISK's 512-bit descriptor rows. The result is a distinct `BriskMatches` owner,
so neither ORB feature nor ORB match handles can enter this path. Distances are
validated from zero through 512, indices remain unique and in range, results
are sorted by distance and descriptor-row indices, and the retained owner has
no borrow of either source analysis.

`opencv.Features.detect_akaze(image, threshold, max_features)` creates a
distinct owned AKAZE analysis with a deliberately fixed binary policy:
rotated full-width MLDB descriptors, three descriptor channels, four octaves,
four octave layers, and `DIFF_PM_G2` diffusivity. Thresholds must be finite and
greater than zero through one, and output is capped at 4,096 rows. The adapter
validates OpenCV's configured policy, Hamming norm, unsigned-byte storage, and
61-byte descriptor width before canonicalizing each keypoint with its row.
Copied keypoint and descriptor projections remain valid after source disposal.

`opencv.Features.match_akaze(query, train, max_matches)` applies the shared
private cross-checked Hamming matcher to the full 486-bit MLDB payload stored
in each 61-byte row. It returns an independent `AkazeMatches` owner with
copied, canonical indices, distances, and endpoint coordinates.
Family-specific `AkazeFeatures` and `AkazeMatches` types prevent ORB or BRISK
handles from entering this path.

`opencv.Features.detect_sift(image, contrast_threshold, max_features)` adds a
floating-point descriptor lane without introducing duplicate feature storage
or match records. It fixes three octave layers, edge threshold `10`, sigma
`1.6`, `CV_32F` descriptors, and default non-precise upscaling. Contrast must
be finite and greater than zero through one; output is capped at 4,096 rows.
Each owned `SiftFeatures` resource carries canonical keypoints and their
row-aligned 128-element descriptors, exposed only through copied float lists.

`opencv.Features.detect_affine_sift(image, contrast_threshold, max_features)`
wraps that maintained SIFT backend with OpenCV `AffineFeature`. It fixes the
tilt range to powers zero through two, the tilt step to square root of two,
and the rotation-step base to `72`. The private view vectors are validated,
round-tripped through the mutable detector, and never cross the boundary.
Response-ranked truncation keeps descriptors aligned before the shared
canonical projection. The operation returns the existing `SiftFeatures`
owner, so copied projections, L2 matching, disposal, and lifecycle accounting
remain one coherent type contract rather than a parallel affine hierarchy.

`opencv.Features.match_sift(query, train, max_matches)` uses the same shared
feature and match ownership model as the binary families while fixing
cross-checked `NORM_L2` semantics. It returns an independent `SiftMatches`
owner with finite float distances, unique bounded indices, and copied endpoint
coordinates. Binary and float families remain distinct public resource types.

`opencv.Features.detect_kaze(image, threshold, max_features)` creates a
rotation-invariant, non-extended KAZE analysis with four octaves, four octave
layers, and `DIFF_PM_G2` diffusivity. Thresholds must be finite and greater
than zero through one, and retained output is capped at 4,096 rows after
deterministic response ranking. Each `KazeFeatures` owner keeps canonical
keypoints aligned with copied 64-element `CV_32F` descriptor rows.

`opencv.Features.match_kaze(query, train, max_matches)` uses the shared
cross-checked `NORM_L2` matcher and returns a distinct `KazeMatches` owner.
Finite distances, unique bounded indices, canonical ordering, copied endpoint
coordinates, and source-independent lifetime follow the same private float
feature contract as SIFT without making either public resource interchangeable.

`opencv.Features.detect_mser(image, delta, minimum_area, maximum_area,
max_regions)` detects maximally stable extremal regions from bounded grayscale
images. The package fixes maximum variation `0.25`, minimum diversity `0.2`,
maximum evolution `200`, area threshold `1.01`, minimum margin `0.003`, edge
blur size `5`, and both-pass execution. Caller bounds must keep the maximum
retained point total within 1,048,576 pixels.

The resulting `MserRegions` owner canonicalizes unique region points in
row-major order, verifies exact bounding rectangles, ranks stronger regions
before truncation, and then orders retained regions by bounds. Points,
cumulative offsets, and bounds are copied through shared private region
projection helpers also used by contour resources. No source image or native
MSER detector remains borrowed by the result.

`opencv.Features.simple_blob_keypoints(image, blob_value, minimum_area,
maximum_area, minimum_distance, max_blobs)` detects dark (`0`) or bright
(`255`) grayscale blobs and returns copied `[x, y, diameter, response]` rows.
Threshold scanning is fixed from `0` through `256` in steps of `10` with two
required repetitions. Color and caller-bounded area filtering are enabled;
circularity, inertia, convexity, and contour collection are disabled.

The adapter round-trips and verifies the complete OpenCV parameter object,
validates every keypoint, ranks candidates before bounded truncation, and
orders retained rows by center. The mutable detector, parameter object, and
native keypoints never cross the package boundary.

`opencv.Features.match_orb(query, train, max_matches)` uses OpenCV's maintained
brute-force matcher with `NORM_HAMMING` and cross-checking, matching the fixed
two-point ORB descriptor policy. It returns an independent `OrbMatches` owner
instead of exposing `cv::DMatch`. Matches are validated for finite integral
zero-through-256 distance, unique query and train rows, and bounded indices;
then ordered by distance, query row, and train row before truncation at 4,096.
Copied index pairs, distances, and query/train endpoint coordinates remain
row-aligned and usable after either source feature analysis is disposed.

Probabilistic line detection exposes
`opencv.Imgproc.line_segments(mask, vote_threshold, minimum_length,
maximum_gap)` for the same bounded grayscale-analysis input. The adapter
normalizes all nonzero bytes to binary, invokes OpenCV `HoughLinesP` with
package-owned one-pixel and one-degree accumulator resolution, and copies at
most 4,096 segments as flattened `[x1, y1, x2, y2, ...]` integers. Vote count
is positive and bounded by the longest image dimension; length and gap are
nonnegative and bounded by the image diagonal. Endpoints are validated,
oriented lexicographically, sorted, and deduplicated before copying so native
detector ordering does not enter the Terlan contract. The source remains
unchanged, and accumulator constants, native vectors, and mutable outputs stay
private.

Immutable line rasterization exposes
`opencv.Imgproc.draw_line(image, start_x, start_y, end_x, end_y, value,
thickness)` for bounded unsigned-byte grayscale images. Both endpoints must be
inside the image, value is one byte, and thickness is from one through 255.
The adapter clones the source before invoking OpenCV `line` with fixed
eight-connected rasterization and zero fractional shift, validates the
shape-preserving result through the shared owner helper, and returns a fresh
`Mat`. The source remains unchanged. OpenCV's mutable destination, scalar,
line-type, and fixed-point parameters remain private.

Immutable rectangle rasterization exposes
`opencv.Imgproc.draw_rectangle(image, x, y, width, height, value, thickness)`
under the same grayscale value, thickness, cloning, and result-ownership
policy. Width and height are positive, and the complete rectangle must remain
inside the source image. The adapter uses OpenCV's rectangle overload with a
checked `Rect`, fixed eight-connected rasterization, and zero fractional
shift. Coordinates are validated with subtraction-based containment checks so
malformed extents cannot overflow. The source remains unchanged, and mutable
destination and native drawing policy remain private.

Immutable circle rasterization exposes
`opencv.Imgproc.draw_circle(image, center_x, center_y, radius, value,
thickness)` under the shared grayscale value, thickness, cloning, and fresh
ownership policy. Radius zero is a one-pixel point and therefore requires
thickness one. Positive radii include half the stroke thickness in the
containment margin, so the complete rasterized circle remains inside the
source. The adapter fixes eight-connected rasterization and zero fractional
shift; mutable destinations, fill mode, antialiasing, and subpixel coordinates
remain private.

Immutable rectangle filling exposes
`opencv.Imgproc.fill_rectangle(image, x, y, width, height, value)` as a
separate semantic operation. Positive extents must remain fully contained,
and the value must fit one byte. The adapter clones the source and invokes
OpenCV rectangle rasterization with package-owned `FILLED` and `LINE_8`
policy before returning a fresh matching owner. This avoids publishing the
negative-thickness sentinel accepted by OpenCV and keeps mutation private.

Immutable circle filling exposes
`opencv.Imgproc.fill_circle(image, center_x, center_y, radius, value)` through
the shared contained-circle and byte-value policy. Radius zero normalizes to a
single pixel. The adapter clones the source and invokes OpenCV circle
rasterization with package-owned `FILLED`, `LINE_8`, and zero-shift policy,
then returns a fresh matching owner. OpenCV's negative-thickness sentinel,
mutable destination, antialiasing, and subpixel coordinates remain private.

Full-layout byte inversion exposes `opencv.Imgproc.invert(image)` for gray,
BGR, and BGRA unsigned-byte owners. OpenCV `bitwise_not` writes every
`255 - value` result into fresh matching storage while the source remains
unchanged. Applying the operation twice restores the original bytes. This is
separate from `mask_not`, whose one-channel-only contract belongs to mask
algebra rather than general image transformation.

Full-layout binary bitwise composition exposes `opencv.Imgproc.bitwise_and`,
`bitwise_or`, and `bitwise_xor` for two compatible gray, BGR, or BGRA
unsigned-byte owners. One shared native path validates equal dimensions and
layout, dispatches to the selected OpenCV bitwise primitive, and returns fresh
matching storage without changing either input. These operations remain
distinct from the one-channel `mask_and`, `mask_or`, and `mask_xor` contracts.

Immutable cross rasterization exposes `opencv.Imgproc.draw_cross(image,
center_x, center_y, size, value, thickness)` for bounded unsigned-byte
grayscale owners. Size is positive, odd, and no greater than 255; the complete
marker and stroke must remain inside the source. The adapter clones before
invoking OpenCV `drawMarker` with package-owned `MARKER_CROSS` and `LINE_8`
policy and returns fresh matching ownership. Marker enums, mutable
destinations, antialiasing, and subpixel coordinates remain private.

Immutable axis-aligned ellipse rasterization exposes
`opencv.Imgproc.draw_ellipse(image, center_x, center_y, radius_x, radius_y,
value, thickness)` and `opencv.Imgproc.fill_ellipse(image, center_x, center_y,
radius_x, radius_y, value)`. Both radii are positive and the complete ellipse
must remain inside a bounded unsigned-byte grayscale source; outlines also
include half the stroke thickness in that containment margin. The adapter
clones before invoking OpenCV `ellipse` with package-owned zero rotation, full
zero-through-360-degree arc, `LINE_8`, zero shift, and either positive
thickness or `FILLED`, then returns fresh matching ownership. Arbitrary arcs,
rotation, mutable destinations, antialiasing, and subpixel coordinates remain
private.

Squared-difference template matching uses OpenCV `matchTemplate` with the
package-fixed `TM_SQDIFF` mode. Image and pattern must have the same public
unsigned-byte layout, and the pattern must fit inside the image. The operation
returns finite nonnegative scores in row-major order for the complete
`(image rows - pattern rows + 1) x (image columns - pattern columns + 1)`
search area; lower scores are better matches. Copied output is capped at
1,048,576 scores and work is capped at 67,108,864 channel-pixel comparisons.
Both matrices remain unchanged and no native result resource is retained.

Binary-shape description uses OpenCV moments and `HuMoments` over the same
bounded grayscale-analysis input policy. `opencv.Imgproc.hu_moments` always
returns exactly seven finite invariants as a copied `List[Float]`; an empty
mask returns seven zeros. No OpenCV moment structure or temporary vector is
retained across the boundary.

Binary shape comparison exposes
`opencv.Imgproc.shape_match_scores(first, second)` for two independently sized,
bounded, nonempty unsigned-byte grayscale masks. Both inputs are normalized to
binary matrices before OpenCV `matchShapes` runs with fixed I1, I2, and I3
methods and the required zero parameter. The copied result is exactly three
finite nonnegative values in `[I1, I2, I3]` order, with native near-zero noise
normalized to zero. Nonzero byte magnitudes therefore cannot change shape
identity, inputs remain unchanged, same-handle comparison is supported, and no
result resource is retained. Raw matching constants, caller parameters,
contour storage, intensity-weighted matching, and mutable outputs remain
private.

Absolute image difference is the first generated two-resource operation. Both
matrices remain independently owned and are borrowed immutably for one call;
rows, columns, channels, and the private OpenCV element type must match. The
result is a fresh matrix owner. Secondary handles receive the same owner,
type, generation, and stale-handle validation as method receivers, including
when both arguments identify the same live matrix.

Saturated image arithmetic reuses the same immutable two-resource contract.
`opencv.Imgproc.add_saturated` and `opencv.Imgproc.subtract_saturated` accept
only identical gray, BGR, or BGRA unsigned-byte images and return a fresh owner
with the same shape and private type. OpenCV `add` clamps overflow to 255 and
`subtract` clamps underflow to zero. Same-handle operands are valid, both
inputs remain unchanged, and no mask, output depth, raw arithmetic flag, or
mutable destination is published.

Bounded scaled multiplication exposes OpenCV `multiply` as
`opencv.Imgproc.multiply_scaled(first, second, scale)`. Both operands must be
identical gray, BGR, or BGRA unsigned-byte images, and `scale` must be finite
and range from zero through one. Multiplication is per element, the scaled
result saturates to the unsigned-byte range, and a fresh same-layout owner is
returned. Same-handle operands are valid and both sources remain unchanged.
Output-depth controls, matrix-expression overloads, non-element-wise matrix
multiplication, and mutable destinations remain private.

Bounded scaled division exposes OpenCV `divide` as
`opencv.Imgproc.divide_scaled(first, second, scale)`. Both operands must be
identical gray, BGR, or BGRA unsigned-byte images, and `scale` must be finite
and range from zero through 255. Each output byte is the saturated result of
`first * scale / second`; a zero denominator produces zero under OpenCV's
integer-array contract. Same-handle operands are valid, both inputs remain
unchanged, and the result is a fresh same-layout owner. Scalar division,
output-depth controls, matrix expressions, and mutable destinations remain
private.

Element-wise image extrema also reuse the immutable image-pair contract.
`opencv.Imgproc.minimum` and `opencv.Imgproc.maximum` select each gray, BGR,
or BGRA byte independently through OpenCV `min` and `max`. Inputs must have
identical dimensions, channels, and private unsigned-byte type. Same-handle
inputs produce an equal but independently owned result, both sources remain
unchanged, and scalar overloads and mutable output arrays remain private.

Typed grayscale comparison exposes OpenCV `compare` only as
`opencv.Imgproc.compare_mask`. Equal, unequal, less, less-or-equal, greater,
and greater-or-equal are semantic atoms lowered privately to OpenCV codes.
Both inputs must be same-size unsigned-byte grayscale owners. True pixels are
255 and false pixels are zero in a fresh grayscale owner; inputs remain
unchanged. Scalar comparison, multi-channel comparison, raw integer flags,
and mutable output remain private.

Immutable mask selection exposes `opencv.Imgproc.select_mask(when_true,
when_false, mask)` as a three-resource composition operation. Image branches
must have identical gray, BGR, or BGRA unsigned-byte storage, and the mask must
be same-size unsigned-byte grayscale. Every nonzero mask pixel selects the
true branch; zero selects the false branch. The adapter clones the false branch
before privately invoking OpenCV `copyTo`, so all inputs remain unchanged and
the result has fresh ownership. Branch and mask aliasing are valid; mutable
destinations and broader OpenCV mask layouts remain private.

Per-pixel byte-mask blending exposes `opencv.Imgproc.blend_mask(first, second,
weights)` for compatible gray, BGR, or BGRA unsigned-byte images and one
bounded same-size unsigned-byte grayscale weight map. A weight of 255 selects
the first image, zero selects the second image, and intermediate values
interpolate between them. The adapter privately converts the byte map into
complementary single-channel floating-point maps before invoking OpenCV
`blendLinear`. All inputs remain unchanged, aliased image inputs are valid,
and the result has fresh matching ownership. Floating-point source matrices,
caller-provided weight pairs, and mutable output destinations remain private.

Per-channel mean absolute error reuses that immutable two-resource contract
without allocating a result matrix. OpenCV `absdiff` computes unsigned-byte
element differences and `mean` reduces them independently in gray, BGR, or
BGRA channel order. `opencv.Imgproc.mean_absolute_error_channels` returns one
copied finite value from zero through 255 per channel. Both inputs remain
unchanged, equal handles return zeros, and no native result resource survives
the call.

Horizontal and vertical concatenation extend the immutable two-resource
contract to checked image composition. Horizontal inputs must have equal row
counts, channel layouts, and private element types; vertical inputs must have
equal column counts, layouts, and private types. The growing dimension is
added with overflow checks and the result must fit the package-wide 256 MiB
allocation limit. `opencv.Imgproc.concat_horizontal` and
`opencv.Imgproc.concat_vertical` call OpenCV `hconcat` and `vconcat`, leave
both inputs unchanged, permit both handles to identify the same matrix, and
return a fresh owner with the exact validated shape and source type. OpenCV's
variadic input-array forms remain private.

Image transposition accepts every public unsigned-byte layout and swaps the
validated row and column dimensions under the same 256 MiB allocation policy.
`opencv.Imgproc.transpose` invokes OpenCV `transpose`, preserves the source
channel layout and private element type, leaves the source unchanged, and
returns a fresh matrix owner. The package exposes only two-dimensional image
transposition; OpenCV's general multidimensional transpose surface remains
private.

Image tiling accepts every public unsigned-byte layout and positive vertical
and horizontal repetition counts. `opencv.Imgproc.tile` validates both output
dimension multiplications against OpenCV's integer dimensions and the final
storage against the package-wide 256 MiB limit before invoking OpenCV
`repeat`. The source remains unchanged, its channel layout and private element
type are preserved, and the repeated image receives fresh ownership. Counts
of one are valid and still produce an independent result owner.

Weighted image addition reuses that two-resource contract. Both weights must
be finite with absolute values no larger than 16, and the finite additive
offset is bounded to -255 through 255. OpenCV saturation remains explicit in
the operation semantics, both inputs remain unchanged, and every result has
fresh ownership with the same shape, channels, and private element type.

Semantic masking borrows a second matrix without requiring its channel layout
to equal the image. The image must have one, three, or four unsigned-byte
channels; the mask must be a same-size one-channel unsigned-byte matrix.
Nonzero mask bytes retain complete source pixels, zero bytes clear them, and
the operation returns a fresh image owner without modifying either input.

Inclusive range selection creates those masks from valid one-channel
unsigned-byte images. Lower and upper bounds are ordinary integers from 0
through 255 with `lower <= upper`; equal bounds select one exact value. The
result contains only 0 or 255, retains source dimensions under fresh
ownership, and composes directly with semantic masking.

Mask composition exposes bytewise intersection, union, exclusive-or, and
complement only for equal-size one-channel unsigned-byte matrices. Binary
operations allow both handles to identify the same live mask. Inputs remain
unchanged and each result has independent ownership, preserving binary masks
produced by range selection while also defining exact behavior for arbitrary
mask bytes. Exclusive-or with the same handle produces an all-zero mask.

Nonzero mask extraction uses OpenCV `findNonZero` over the same one-channel
analysis policy and returns flattened row-major `x, y` coordinates as a copied
`List[Int]`. Inputs are capped at 1,048,576 pixels, empty masks return an empty
list, and no native point vector or additional resource survives the call.

Affine and perspective transforms cross the boundary as copied value records.
Every coefficient must be finite and have absolute value no larger than one
million; output dimensions, interpolation, border policy, and allocation are
validated before OpenCV runs. Each transform returns a fresh matrix owner.

`opencv.Calib3d.undistort(image, intrinsics, distortion)` accepts copied
`CameraIntrinsics` and five-coefficient `DistortionCoefficients` records. Focal
lengths must be positive, finite, and bounded; the principal point must lie
inside the image; and every radial or tangential coefficient must be finite
with absolute value no larger than ten. The CPU operation accepts one-, three-,
or four-channel unsigned-byte images up to 1,048,576 pixels, preserves the
source camera matrix and image geometry, and returns a fresh matrix owner
without retaining or modifying the source.

`opencv.Calib3d.undistort_optimal(image, intrinsics, distortion, alpha,
center_principal_point)` additionally asks OpenCV to compute its maintained
optimal camera matrix for the same output dimensions. Alpha must be finite in
the inclusive range zero through one: zero retains only valid corrected pixels,
while one retains all source pixels and may introduce black virtual pixels.
The Boolean policy either centers the output principal point or lets OpenCV
choose it. The computed matrix and optional valid-pixel ROI remain private;
the operation preserves the source layout and returns a fresh matrix owner.

`opencv.Calib3d.undistort_points(points, intrinsics, distortion)` accepts a
flattened copied `[x, y, ...]` list and returns corrected coordinates in the
same pixel coordinate space. The adapter validates an even coordinate count,
at most 4,096 points, finite coordinates bounded to absolute value one
million, and the same finite camera/distortion coefficient limits used by
image correction. Empty input returns an empty VM-owned list. The input is a
call-scoped CXX slice; no OpenCV point vector or camera matrix survives the
call.

`opencv.Calib3d.perspective_transform_points(points, transform)` applies the
existing copied nine-field perspective transform to at most 4,096 flattened
point pairs. Input and output coordinates must remain finite and within the
million-unit bound; the operation returns a copied `List[Float]` and owns no
native handle.

`opencv.Calib3d.affine_transform_points(points, transform)` applies the
existing copied six-field affine transform under the same point-count,
coordinate, output, and ownership policy. It uses OpenCV's maintained core
vector transform rather than duplicating affine arithmetic in the adapter.

`opencv.Calib3d.estimate_perspective_transform(source_points,
destination_points)` accepts exactly four finite, million-bounded source and
destination pairs. Duplicate points and any collinear triple are rejected
before OpenCV runs. The adapter calls the maintained
`getPerspectiveTransform`, validates the finite bounded nonsingular 3x3
result, then projects one temporary unique owner through reviewed getters into
the existing `PerspectiveTransform` record. The temporary is dropped during
the call; no transform handle or OpenCV matrix survives the boundary.

`opencv.Calib3d.estimate_affine_transform(source_points, destination_points)`
accepts exactly three finite, million-bounded source and destination pairs.
Duplicate or collinear triangles are rejected before OpenCV runs. The adapter
calls the maintained `getAffineTransform`, validates the finite bounded
nonsingular 2x3 result, then projects one temporary unique owner through
reviewed getters into the existing `AffineTransform` record. The temporary is
dropped during the call; no transform handle or OpenCV matrix survives the
boundary.

`opencv.Calib3d.project_points(object_points, rotation, translation,
intrinsics, distortion)` accepts at most 4,096 flattened `[x, y, z, ...]`
object points plus copied `Vector3`, `CameraIntrinsics`, and
`DistortionCoefficients` records. Coordinates and pose components must be
finite and bounded to absolute value one million; camera and distortion values
reuse the existing finite package policy. The adapter calls maintained
`projectPoints` with private OpenCV vectors and matrices, requires one output
point per input point, and copies finite million-bounded image coordinates back
to the VM. Empty input returns an empty list, and no OpenCV point, matrix, or
Jacobian survives the call.

`opencv.Calib3d.solve_pose(object_points, image_points, intrinsics,
distortion)` accepts six through 4,096 matching 3D and 2D correspondences. The
adapter fixes OpenCV to `SOLVEPNP_ITERATIVE`, disables caller-provided initial
guesses, and keeps its vectors and camera matrices private. It rejects
malformed dimensions, mismatched cardinality, non-finite or million-excessive
coordinates, and invalid camera models before solving. A successful finite,
bounded Rodrigues rotation and translation are copied into `CameraPose`; the
temporary native result is dropped before returning to the VM.

`opencv.Calib3d.solve_pose_robust(object_points, image_points, intrinsics,
distortion)` shares the same six through 4,096 correspondence, camera, copied
result, and rejection contract. It calls the maintained classic
`solvePnPRansac` overload with no initial guess, `SOLVEPNP_ITERATIVE`, 100
iterations, an 8-pixel reprojection threshold, and confidence `0.99`. At least
six inliers must remain. Inlier indices and all mutable OpenCV outputs stay
private, and only the finite bounded `CameraPose` crosses the boundary.

`opencv.Calib3d.solve_pose_minimal(object_points, image_points, intrinsics,
distortion)` accepts exactly four matching finite, million-bounded 3D and 2D
observations. It calls maintained `solveP3P` with fixed `SOLVEPNP_AP3P` and
copies the first candidate, which OpenCV orders by reprojection error. Empty,
missing, additional, mismatched, degenerate, or invalid observations are
rejected. All candidate vectors remain private and only one validated finite
bounded `CameraPose` crosses the boundary.

`opencv.Calib3d.solve_pose_planar(object_points, image_points, intrinsics,
distortion)` accepts four through 4,096 matched observations whose 3D points
span one nondegenerate plane and whose image points span a two-dimensional
region. It calls maintained `solvePnPGeneric` with fixed `SOLVEPNP_IPPE`, no
initial guess, and private reprojection-error output. The adapter explicitly
selects the finite nonnegative minimum-error candidate and returns one copied
bounded `CameraPose`; alternate poses and error arrays never cross the native
boundary.

`opencv.Calib3d.solve_square_marker_pose(image_points, marker_size, intrinsics,
distortion)` accepts exactly four finite, million-bounded image corners plus a
finite marker size greater than zero and no larger than one million. Corners
must be supplied in the order top-left, top-right, bottom-right, bottom-left as
viewed in the marker's object plane. The adapter privately constructs the
OpenCV-required object points `[-s/2, s/2, 0]`, `[s/2, s/2, 0]`,
`[s/2, -s/2, 0]`, `[-s/2, -s/2, 0]`, then calls maintained
`solvePnPGeneric` with fixed `SOLVEPNP_IPPE_SQUARE`. It shares planar pose's
finite nonnegative minimum-error selection and returns one copied bounded
`CameraPose`; object points, alternate candidates, and error arrays remain
private.

`opencv.Calib3d.refine_pose(object_points, image_points, initial_pose,
intrinsics, distortion)` accepts the same six through 4,096 correspondence
contract plus one finite, million-bounded copied `CameraPose`. The adapter
calls maintained `solvePnPRefineLM` with a fixed 20-step count-and-epsilon
termination policy. OpenCV mutates only private vector copies; the caller's
initial record is unchanged, and the finite bounded result is copied into a
fresh `CameraPose` before the temporary native owner is dropped.

`opencv.Calib3d.refine_pose_vvs(object_points, image_points, initial_pose,
intrinsics, distortion)` provides the maintained virtual visual servoing
companion under the same input, ownership, and termination contract. Its VVS
gain is fixed to `1.0`; caller-controlled convergence or gain parameters are
not part of the public package API. Both refiners share one validation and
copied-result path in the adapter.

`opencv.Calib3d.compose_pose(first, second)` applies `first` and then `second`
using the maintained `composeRT` operation. Both copied poses must contain
finite, million-bounded Rodrigues rotation and translation components. The
adapter keeps all OpenCV vectors and optional derivative matrices private,
validates the composed result, and returns a fresh copied `CameraPose` without
retaining native transformation state.

Codec input and output are copied buffers. The first codec surface accepts only
BMP signatures, limits encoded buffers to 64 MiB, limits decoded matrix storage
to 256 MiB before VM ownership, and accepts only finite decode-mode and format
discriminants. Empty or malformed decode results and unavailable codecs use
finite package statuses rather than OpenCV payloads.

`opencv_failure.hpp` contains every exception before it can cross CXX. Each
thread stores only a finite integer status; it never retains `what()` text or
another native payload. Generated helpers consume and clear that status after a
null result and map it through the manifest's reviewed `opencv.*` code/message
table. Invalid arguments, allocation failure, standard native exceptions,
unknown exceptions, and non-contiguous output therefore remain distinguishable
without trusting native diagnostics. The status probe is part of the private
CXX bridge and is not generated into public Terlan modules.

The offline surface and metadata gates are mutation-tested. They reject drift
in the pinned upstream identity, enabled module set, CMake policy, reviewed
declarations and source locations, mapping and skip inventories, adapter
namespace/header/source identity, and compile database. Missing committed
metadata inputs fail before generation or native compilation begins.

`native_install_check` validates a prepared installation before generated code
is built. It requires the core, imgproc, imgcodecs, features, and calib3d headers and
exact native library files, compiles the committed C++14 probe, and links
matrix, image-processing, byte-codec, feature-detection, and camera-correction symbols against
the selected artifacts.
Missing headers, libraries, compilers, incompatible headers, and missing
symbols therefore have separate stable diagnostics. Compiler and linker output
is not reflected into package errors.
