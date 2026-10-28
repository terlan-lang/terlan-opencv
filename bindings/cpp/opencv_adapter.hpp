#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#if defined(TERLAN_OPENCV_METADATA)
#include "opencv_metadata_rust.hpp"
#else
#include "rust/cxx.h"
#endif
#include "opencv_ml.hpp"
namespace terlan_opencv {
class OpenCvDnnBlob;
/// Defines the supported unsigned-byte layouts for owned matrices.
enum class PixelLayout : std::int32_t {
  Gray = 1,
  Bgr = 3,
  Bgra = 4,
};

/// Defines every scalar depth supported by an OpenCV matrix.
enum class MatrixDepth : std::int32_t {
  U8 = 0,
  I8 = 1,
  U16 = 2,
  I16 = 3,
  I32 = 4,
  F32 = 5,
  F64 = 6,
  F16 = 7,
};

/// Defines the supported resize interpolation policies.
enum class Interpolation : std::int32_t {
  Nearest = 0,
  Linear = 1,
  Area = 2,
  Cubic = 3,
  Lanczos4 = 4,
  LinearExact = 5,
  NearestExact = 6,
};

/// Defines the supported color conversion policies.
enum class ColorConversion : std::int32_t {
  BgrToGray = 0,
  GrayToBgr = 1,
  BgrToBgra = 2,
  BgraToBgr = 3,
};

/// Defines supported encoded image formats.
enum class ImageFormat : std::int32_t {
  Bmp = 0, Png = 1, Jpeg = 2, Webp = 3, Tiff = 4,
};

/// Defines supported image decode output modes.
enum class DecodeMode : std::int32_t {
  Unchanged = -1,
  Grayscale = 0,
  Color = 1,
};

/// Holds copied image dimensions in width-height order.
struct ImageSize final {
  std::int64_t width;
  std::int64_t height;
};

/// Holds one copied two-dimensional affine transform in row-major order.
struct AffineTransform final {
  double m00; /** Returns m00. */ double m00_value() const noexcept { return m00; }
  double m01; /** Returns m01. */ double m01_value() const noexcept { return m01; }
  double m02; /** Returns m02. */ double m02_value() const noexcept { return m02; }
  double m10; /** Returns m10. */ double m10_value() const noexcept { return m10; }
  double m11; /** Returns m11. */ double m11_value() const noexcept { return m11; }
  double m12; /** Returns m12. */ double m12_value() const noexcept { return m12; }
};

class OpenCvComponents;
class OpenCvOrbFeatures;
class OpenCvBriskFeatures;
class OpenCvBriskMatches;
class OpenCvAkazeFeatures;
class OpenCvSiftFeatures;
class OpenCvKazeFeatures;
class OpenCvMserRegions;
/// Uniquely owns one process-local OpenCV matrix.
class OpenCvMat final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvMat(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned OpenCV matrix.
  ~OpenCvMat() noexcept;

  /// Returns the matrix row count.
  std::int64_t rows() const noexcept;
  /// Returns the matrix column count.
  std::int64_t cols() const noexcept;
  /// Returns the matrix channel count.
  std::int64_t channels() const noexcept;
  /// Returns the OpenCV scalar depth code.
  std::int64_t depth_code() const noexcept;
  /// Returns the byte width of one scalar channel value.
  std::int64_t element_size() const noexcept;
  /// Returns the byte width of one matrix element across all channels.
  std::int64_t pixel_size() const noexcept;
  /// Returns the OpenCV matrix type code.
  std::int64_t type_code() const noexcept;
  /// Returns the number of matrix dimensions.
  std::int64_t dimension_count() const noexcept;
  /// Returns the number of matrix elements excluding channels.
  std::int64_t element_count() const noexcept;
  /// Returns the stride for one dimension in scalar values.
  std::int64_t step_scalars(std::int64_t dimension) const noexcept;
  /// Returns the complete contiguous matrix storage size in bytes.
  std::int64_t byte_count() const noexcept;
  /// Reports whether the matrix has no elements.
  bool empty() const noexcept;
  /// Reports whether the matrix aliases a parent matrix.
  bool is_submatrix() const noexcept;
  /// Reports whether matrix bytes occupy one contiguous region.
  bool is_contiguous() const noexcept;
  /// Copies one checked row interval into a fresh independent owner.
  std::unique_ptr<OpenCvMat> row_range_copy_owned(
      std::int64_t start, std::int64_t end) const noexcept;
  /// Copies one checked column interval into a fresh independent owner.
  std::unique_ptr<OpenCvMat> col_range_copy_owned(
      std::int64_t start, std::int64_t end) const noexcept;
  /// Copies one checked diagonal into a fresh column-matrix owner.
  std::unique_ptr<OpenCvMat> diagonal_copy_owned(
      std::int64_t offset) const noexcept;
  /// Reshapes contiguous storage and copies it into a fresh owner.
  std::unique_ptr<OpenCvMat> reshape_copy_owned(
      std::int64_t rows, std::int64_t channels) const noexcept;
  /// Deep-copies the matrix into a new owner.
  std::unique_ptr<OpenCvMat> clone_owned() const noexcept; /** Creates one fixed-policy NCHW RGB inference blob from this BGR image. */ std::unique_ptr<OpenCvDnnBlob> blob_from_bgr_rgb_owned(std::int64_t width, std::int64_t height) const noexcept; /** Creates one fixed-policy NCHW inference blob from this grayscale image. */ std::unique_ptr<OpenCvDnnBlob> blob_from_grayscale_owned(std::int64_t width, std::int64_t height) const noexcept; /** Creates one center-cropped NCHW RGB inference blob from this BGR image. */ std::unique_ptr<OpenCvDnnBlob> blob_from_bgr_rgb_center_crop_owned(std::int64_t width, std::int64_t height) const noexcept; /** Creates one center-cropped NCHW inference blob from this grayscale image. */ std::unique_ptr<OpenCvDnnBlob> blob_from_grayscale_center_crop_owned(std::int64_t width, std::int64_t height) const noexcept; /** Creates one per-channel normalized NCHW RGB inference blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_bgr_rgb_normalized_owned(std::int64_t width, std::int64_t height, rust::Slice<const double> means, rust::Slice<const double> scales) const noexcept; /** Creates one center-cropped per-channel normalized NCHW RGB blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_bgr_rgb_normalized_center_crop_owned(std::int64_t width, std::int64_t height, rust::Slice<const double> means, rust::Slice<const double> scales) const noexcept; /** Creates one aspect-preserving zero-padded NCHW RGB blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_bgr_rgb_letterbox_owned(std::int64_t width, std::int64_t height) const noexcept; /** Creates one aspect-preserving zero-padded grayscale NCHW blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_grayscale_letterbox_owned(std::int64_t width, std::int64_t height) const noexcept; /** Creates one letterboxed per-channel normalized NCHW RGB blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_bgr_rgb_normalized_letterbox_owned(std::int64_t width, std::int64_t height, rust::Slice<const double> means, rust::Slice<const double> scales) const noexcept; /** Creates one normalized NCHW grayscale blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_grayscale_normalized_owned(std::int64_t width, std::int64_t height, rust::Slice<const double> means, rust::Slice<const double> scales) const noexcept; /** Creates one center-cropped normalized NCHW grayscale blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_grayscale_normalized_center_crop_owned(std::int64_t width, std::int64_t height, rust::Slice<const double> means, rust::Slice<const double> scales) const noexcept; /** Creates one letterboxed normalized NCHW grayscale blob. */ std::unique_ptr<OpenCvDnnBlob> blob_from_grayscale_normalized_letterbox_owned(std::int64_t width, std::int64_t height, rust::Slice<const double> means, rust::Slice<const double> scales) const noexcept;
  /// Computes the finite flattened dot product with one compatible matrix.
  double dot_value(const OpenCvMat& other) const noexcept;
  /// Multiplies two compatible floating-point matrices into a fresh owner.
  std::unique_ptr<OpenCvMat> matrix_multiply_owned(
      const OpenCvMat& other) const noexcept;
  /// Inverts one square floating-point matrix using a finite method policy.
  std::unique_ptr<OpenCvMat> matrix_inverse_owned(
      std::int64_t method) const noexcept;
  /// Solves one floating-point linear system into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> matrix_solve_owned(
      const OpenCvMat& right_hand_side, std::int64_t method) const noexcept;
  /// Computes the finite determinant of one square floating-point matrix.
  double determinant_value() const noexcept;
  /// Computes an L1, L2, or infinity norm selected by a finite policy.
  double norm_value(std::int64_t norm) const noexcept;
  /// Copies one finite diagonal sum per matrix channel.
  std::unique_ptr<std::vector<double>> trace_values_owned() const noexcept;
  /// Serializes this matrix into one bounded in-memory OpenCV document.
  std::unique_ptr<std::string> serialize_owned(
      std::int64_t format) const noexcept;
  /// Copies one finite sum per matrix channel.
  std::unique_ptr<std::vector<double>> sum_values_owned() const noexcept;
  /// Copies one finite arithmetic mean per matrix channel.
  std::unique_ptr<std::vector<double>> mean_values_owned() const noexcept;
  /// Copies one finite population standard deviation per matrix channel.
  std::unique_ptr<std::vector<double>>
  standard_deviation_values_owned() const noexcept;
  /// Normalizes the matrix to one finite target norm.
  std::unique_ptr<OpenCvMat> normalize_norm_owned(
      double target, std::int64_t norm) const noexcept;
  /// Reduces one matrix dimension using a symbolic operation.
  std::unique_ptr<OpenCvMat> reduce_owned(
      std::int64_t dimension, std::int64_t operation) const noexcept;
  /// Sorts each row or column into fresh ownership.
  std::unique_ptr<OpenCvMat> sort_owned(
      std::int64_t dimension, std::int64_t order) const noexcept;
  /// Returns source indexes for each independently sorted row or column.
  std::unique_ptr<OpenCvMat> sort_indices_owned(
      std::int64_t dimension, std::int64_t order) const noexcept;
  /// Applies element-wise exponential to a finite floating matrix.
  std::unique_ptr<OpenCvMat> exponential_owned() const noexcept;
  /// Applies element-wise natural logarithm to a positive floating matrix.
  std::unique_ptr<OpenCvMat> logarithm_owned() const noexcept;
  /// Raises every finite floating matrix value to one finite exponent.
  std::unique_ptr<OpenCvMat> power_owned(double exponent) const noexcept;
  /// Applies element-wise square root to a nonnegative floating matrix.
  std::unique_ptr<OpenCvMat> square_root_owned() const noexcept;
  /// Converts scalar depth with checked scale and offset into a new owner.
  std::unique_ptr<OpenCvMat> convert_depth_owned(
      std::int64_t depth, double scale, double offset) const noexcept;
  /// Copies contiguous matrix bytes into an owned standard vector.
  std::unique_ptr<std::vector<std::uint8_t>> to_u8() const noexcept;
  /// Copies contiguous matrix storage into an owned byte vector.
  std::unique_ptr<std::vector<std::uint8_t>> to_bytes() const noexcept;
  /// Copies a contiguous matrix into a pointer-free tensor exchange packet.
  std::unique_ptr<std::vector<std::uint8_t>> to_tensor_packet(
      rust::Str consumer) const noexcept;
  /// Resizes the matrix into a new owner.
  std::unique_ptr<OpenCvMat> resize_owned(std::int64_t width, std::int64_t height, std::int64_t interpolation) const noexcept;
  /// Computes one bounded absolute first-order Sobel response into a new owner.
  std::unique_ptr<OpenCvMat> sobel_edges_owned(std::int64_t x_order, std::int64_t y_order, std::int64_t kernel_size) const noexcept;
  /// Converts the matrix color space into a new owner.
  std::unique_ptr<OpenCvMat> cvt_color_owned(std::int64_t conversion) const noexcept;
  /// Applies Gaussian blur into a new owner.
  std::unique_ptr<OpenCvMat> gaussian_blur_owned(std::int64_t kernel_width, std::int64_t kernel_height, double sigma_x) const noexcept;
  /// Computes one bounded absolute first-order Scharr response into a new owner.
  std::unique_ptr<OpenCvMat> scharr_edges_owned(std::int64_t x_order, std::int64_t y_order) const noexcept;
  /// Encodes this matrix into copied bytes using a reviewed image format.
  std::unique_ptr<std::vector<std::uint8_t>> encode_owned(std::int64_t format) const noexcept;
  /// Applies one bounded stack blur into a new owner.
  std::unique_ptr<OpenCvMat> stack_blur_owned(std::int64_t kernel_width, std::int64_t kernel_height) const noexcept;
  /// Applies bounded pyramid mean-shift filtering into a new BGR owner.
  std::unique_ptr<OpenCvMat> mean_shift_filter_owned(double spatial_radius, double color_radius, std::int64_t max_level) const noexcept;
  /// Copies a validated rectangular region into a new owner.
  std::unique_ptr<OpenCvMat> crop_owned(
      std::int64_t x, std::int64_t y, std::int64_t width, std::int64_t height) const noexcept;
  /// Adds validated borders into a new owner.
  std::unique_ptr<OpenCvMat> pad_owned(
      std::int64_t top, std::int64_t bottom, std::int64_t left, std::int64_t right, std::int64_t border_mode) const noexcept;
  /// Flips this matrix into a new owner using a finite axis policy.
  std::unique_ptr<OpenCvMat> flip_owned(std::int64_t mode) const noexcept;
  /// Rotates this matrix by a finite right-angle policy into a new owner.
  std::unique_ptr<OpenCvMat> rotate_owned(std::int64_t mode) const noexcept;
  /// Applies a finite unsigned-byte threshold policy into a new owner.
  std::unique_ptr<OpenCvMat> threshold_owned(
      double threshold_value, double maximum_value,
      std::int64_t mode) const noexcept;
  /// Erodes this matrix with a bounded rectangular kernel into a new owner.
  std::unique_ptr<OpenCvMat> erode_owned(
      std::int64_t kernel_width, std::int64_t kernel_height,
      std::int64_t iterations) const noexcept;
  /// Dilates this matrix with a bounded rectangular kernel into a new owner.
  std::unique_ptr<OpenCvMat> dilate_owned(
      std::int64_t kernel_width, std::int64_t kernel_height,
      std::int64_t iterations) const noexcept;
  /// Detects Canny edges in a grayscale matrix into a new owner.
  std::unique_ptr<OpenCvMat> canny_owned(
      double low_threshold, double high_threshold,
      std::int64_t aperture_size, bool l2_gradient) const noexcept;
  /// Min-max normalizes typed values into a finite ordered range.
  std::unique_ptr<OpenCvMat> normalize_min_max_owned(
      double minimum, double maximum) const noexcept;
  /// Applies a bounded affine transform into a newly owned matrix.
  std::unique_ptr<OpenCvMat> warp_affine_owned(
      double m00, double m01, double m02, double m10, double m11,
      double m12, std::int64_t width, std::int64_t height,
      std::int64_t interpolation, std::int64_t border_mode) const noexcept;
  /// Applies a bounded perspective transform into a newly owned matrix.
  std::unique_ptr<OpenCvMat> warp_perspective_owned(
      double m00, double m01, double m02, double m10, double m11,
      double m12, double m20, double m21, double m22, std::int64_t width,
      std::int64_t height, std::int64_t interpolation,
      std::int64_t border_mode) const noexcept;
  /// Equalizes one grayscale histogram into a newly owned matrix.
  std::unique_ptr<OpenCvMat> equalize_hist_owned() const noexcept;
  /// Copies one grayscale image histogram into 256 integer bins.
  std::unique_ptr<std::vector<std::int64_t>> histogram_u8_owned() const noexcept;
  /// Computes one bounded connected-component analysis.
  std::unique_ptr<OpenCvComponents> connected_components_owned(
      std::int64_t connectivity) const noexcept;
  /// Finds simplified external contours in one bounded grayscale mask.
  std::unique_ptr<class OpenCvContours>
  find_external_contours_owned() const noexcept;
  /// Detects bounded Shi-Tomasi corners and copies flattened coordinates.
  std::unique_ptr<std::vector<double>> good_features_to_track_owned(
      std::int64_t max_corners, double quality_level,
      double minimum_distance) const noexcept;
  /// Copies the seven Hu invariants for one bounded grayscale mask.
  std::unique_ptr<std::vector<double>> hu_moments_owned() const noexcept;
  /// Computes absolute per-element difference into a new matrix owner.
  std::unique_ptr<OpenCvMat> abs_diff_owned(const OpenCvMat& other) const noexcept;
  /// Combines two compatible matrices with bounded weights and offset.
  std::unique_ptr<OpenCvMat> add_weighted_owned(const OpenCvMat& other,
      double first_weight, double second_weight, double offset) const noexcept;
  /// Applies a same-size one-channel mask into a new matrix owner.
  std::unique_ptr<OpenCvMat> apply_mask_owned(const OpenCvMat& mask) const noexcept;
  /// Selects an inclusive grayscale range into a new one-channel mask.
  std::unique_ptr<OpenCvMat> in_range_owned(
      std::int64_t lower, std::int64_t upper) const noexcept;
  /// Intersects two same-size one-channel masks into a new owner.
  std::unique_ptr<OpenCvMat> mask_and_owned(const OpenCvMat& other) const noexcept;
  /// Unites two same-size one-channel masks into a new owner.
  std::unique_ptr<OpenCvMat> mask_or_owned(const OpenCvMat& other) const noexcept;
  /// Inverts every byte of a one-channel mask into a new owner.
  std::unique_ptr<OpenCvMat> mask_not_owned() const noexcept;
  /// Copies flattened coordinates for every nonzero mask byte.
  std::unique_ptr<std::vector<std::int64_t>> non_zero_points_owned() const noexcept;
  /// Copies one finite arithmetic mean for each image channel.
  std::unique_ptr<std::vector<double>> mean_channels_owned() const noexcept;
  /// Copies one finite population standard deviation per image channel.
  std::unique_ptr<std::vector<double>> stddev_channels_owned() const noexcept;
  /// Copies the finite minimum and maximum grayscale values.
  std::unique_ptr<std::vector<double>> min_max_values_owned() const noexcept;
  /// Applies one bounded odd-kernel median filter into a new owner.
  std::unique_ptr<OpenCvMat> median_blur_owned(std::int64_t kernel_size) const noexcept;
  /// Applies one bounded edge-preserving bilateral filter into a new owner.
  std::unique_ptr<OpenCvMat> bilateral_filter_owned(
      std::int64_t diameter, double sigma_color, double sigma_space) const noexcept;
  /// Applies one bounded local grayscale threshold into a new owner.
  std::unique_ptr<OpenCvMat> adaptive_threshold_owned(double maximum_value,
      std::int64_t adaptive_method, std::int64_t threshold_mode, std::int64_t block_size, double constant) const noexcept;
  /// Applies one bounded normalized box filter into a new owner.
  std::unique_ptr<OpenCvMat> box_blur_owned(std::int64_t kernel_width, std::int64_t kernel_height) const noexcept;
  /// Computes one bounded absolute grayscale Laplacian into a new owner.
  std::unique_ptr<OpenCvMat> laplacian_edges_owned(std::int64_t kernel_size) const noexcept;
  /// Applies one bounded compound morphology operation into a new owner.
  std::unique_ptr<OpenCvMat> morphology_owned(
      std::int64_t operation, std::int64_t kernel_width,
      std::int64_t kernel_height, std::int64_t iterations) const noexcept;
  /// Blurs and downsamples this image into a new pyramid-level owner.
  std::unique_ptr<OpenCvMat> pyr_down_owned() const noexcept;
  /// Upsamples and blurs this image into a new pyramid-level owner.
  std::unique_ptr<OpenCvMat> pyr_up_owned() const noexcept;
  /// Applies one copied 256-byte lookup table into a new image owner.
  std::unique_ptr<OpenCvMat> apply_lut_owned(rust::Slice<const std::uint8_t> table) const noexcept;
  /// Scales typed values, takes absolute values, and returns a byte matrix.
  std::unique_ptr<OpenCvMat> scale_abs_owned(double scale, double offset) const noexcept;
  /// Sharpens this image with one bounded package-owned kernel.
  std::unique_ptr<OpenCvMat> sharpen_owned(double amount) const noexcept;
  /// Copies one bounded grayscale integral image in row-major order.
  std::unique_ptr<std::vector<std::int64_t>> integral_values_owned() const noexcept;
  /// Copies one nonzero element count for each image channel.
  std::unique_ptr<std::vector<std::int64_t>> non_zero_counts_owned() const noexcept;
  /// Copies one exact unsigned-byte sum for each image channel.
  std::unique_ptr<std::vector<std::int64_t>> sum_channels_owned() const noexcept;
  /// Exclusively combines two compatible masks into a new owner.
  std::unique_ptr<OpenCvMat> mask_xor_owned(const OpenCvMat& other) const noexcept;
  /// Copies bounded squared-difference template-match scores.
  std::unique_ptr<std::vector<double>> match_template_sqdiff_owned(const OpenCvMat& pattern) const noexcept;
  /// Copies one mean absolute error value for each compatible image channel.
  std::unique_ptr<std::vector<double>> mean_absolute_error_channels_owned(
      const OpenCvMat& other) const noexcept;
  /// Applies automatic Otsu binarization into a new grayscale owner.
  std::unique_ptr<OpenCvMat> otsu_threshold_owned(
      double maximum_value, std::int64_t threshold_mode) const noexcept;
  /// Concatenates two compatible images horizontally into a new owner.
  std::unique_ptr<OpenCvMat> concat_horizontal_owned(const OpenCvMat& other) const noexcept;
  /// Concatenates two compatible images vertically into a new owner.
  std::unique_ptr<OpenCvMat> concat_vertical_owned(const OpenCvMat& other) const noexcept;
  /// Transposes this image into a checked fresh owner.
  std::unique_ptr<OpenCvMat> transpose_owned() const noexcept;
  /// Tiles this image by positive vertical and horizontal repetition counts.
  std::unique_ptr<OpenCvMat> tile_owned(std::int64_t vertical_count, std::int64_t horizontal_count) const noexcept;
  /// Extracts one checked channel into a fresh grayscale image owner.
  std::unique_ptr<OpenCvMat> extract_channel_owned(std::int64_t channel_index) const noexcept;
  /// Merges three single-channel typed inputs into a fresh three-channel owner.
  std::unique_ptr<OpenCvMat> merge_bgr_owned(const OpenCvMat& green, const OpenCvMat& red) const noexcept;
  /// Merges four single-channel typed inputs into a fresh four-channel owner.
  std::unique_ptr<OpenCvMat> merge_bgra_owned(const OpenCvMat& green, const OpenCvMat& red, const OpenCvMat& alpha) const noexcept;
  /// Replaces one checked channel with a compatible single-channel source.
  std::unique_ptr<OpenCvMat> replace_channel_owned(const OpenCvMat& replacement, std::int64_t channel_index) const noexcept;
  /// Adds two compatible typed matrices into a fresh owner.
  std::unique_ptr<OpenCvMat> add_saturated_owned(const OpenCvMat& other) const noexcept;
  /// Subtracts two compatible typed matrices into a fresh owner.
  std::unique_ptr<OpenCvMat> subtract_saturated_owned(const OpenCvMat& other) const noexcept;
  /// Selects the per-element minimum of two compatible images into a fresh owner.
  std::unique_ptr<OpenCvMat> minimum_owned(const OpenCvMat& other) const noexcept;
  /// Selects the per-element maximum of two compatible images into a fresh owner.
  std::unique_ptr<OpenCvMat> maximum_owned(const OpenCvMat& other) const noexcept;
  /// Compares two compatible typed matrices into a fresh byte mask owner.
  std::unique_ptr<OpenCvMat> compare_mask_owned(const OpenCvMat& other, std::int64_t mode) const noexcept;
  /// Selects between two compatible images through a grayscale mask into a fresh owner.
  std::unique_ptr<OpenCvMat> select_mask_owned(const OpenCvMat& other, const OpenCvMat& mask) const noexcept;
  /// Multiplies two compatible typed matrices by a bounded scale.
  std::unique_ptr<OpenCvMat> multiply_scaled_owned(const OpenCvMat& other, double scale) const noexcept;
  /// Divides two compatible typed matrices by a bounded scale.
  std::unique_ptr<OpenCvMat> divide_scaled_owned(const OpenCvMat& other, double scale) const noexcept;
  /// Combines two grayscale vector components into a fresh magnitude owner.
  std::unique_ptr<OpenCvMat> gradient_magnitude_owned(const OpenCvMat& other) const noexcept;
  /// Calculates saturated Manhattan distance to the nearest zero mask pixel.
  std::unique_ptr<OpenCvMat> manhattan_distance_transform_owned() const noexcept;
  /// Copies precise Euclidean distances to the nearest zero mask pixel.
  std::unique_ptr<std::vector<double>> euclidean_distance_transform_values_owned() const noexcept;
  /// Flood-fills one exact-value connected region into a fresh image owner.
  std::unique_ptr<OpenCvMat> flood_fill_owned(std::int64_t seed_x, std::int64_t seed_y, std::int64_t fill_value, std::int64_t connectivity) const noexcept;
  /// Copies the minimal axis-aligned bounds of all nonzero mask pixels.
  std::unique_ptr<std::vector<std::int64_t>> non_zero_bounds_owned() const noexcept;
  /// Copies the binary centroid of all nonzero mask pixels.
  std::unique_ptr<std::vector<double>> non_zero_centroid_owned() const noexcept;
  /// Copies the minimum circle enclosing all nonzero mask pixels.
  std::unique_ptr<std::vector<double>> non_zero_enclosing_circle_owned() const noexcept;
  /// Applies automatic Triangle binarization into a new grayscale owner.
  std::unique_ptr<OpenCvMat> triangle_threshold_owned(double maximum_value, std::int64_t threshold_mode) const noexcept;
  /// Blends two compatible images through one bounded byte weight mask.
  std::unique_ptr<OpenCvMat> blend_mask_owned(const OpenCvMat& other, const OpenCvMat& weights) const noexcept;
  /// Copies the convex hull of all nonzero mask pixels in canonical order.
  std::unique_ptr<std::vector<std::int64_t>> non_zero_convex_hull_owned() const noexcept;
  /// Copies the normalized minimum-area box enclosing all nonzero mask pixels.
  std::unique_ptr<std::vector<double>> non_zero_min_area_box_owned() const noexcept;
  /// Copies the normalized minimum triangle enclosing all nonzero mask pixels.
  std::unique_ptr<std::vector<double>> non_zero_enclosing_triangle_owned() const noexcept;
  /// Copies a canonical least-squares line fitted to all nonzero mask pixels.
  std::unique_ptr<std::vector<double>> non_zero_fit_line_owned() const noexcept;
  /// Copies a canonical least-squares ellipse fitted to nonzero mask pixels.
  std::unique_ptr<std::vector<double>> non_zero_fit_ellipse_owned() const noexcept;
  /// Copies a Douglas-Peucker approximation of the nonzero-pixel convex hull.
  std::unique_ptr<std::vector<std::int64_t>> non_zero_approximate_hull_owned(double epsilon) const noexcept;
  /// Copies fixed I1, I2, and I3 binary shape-match scores.
  std::unique_ptr<std::vector<double>> shape_match_scores_owned(const OpenCvMat& other) const noexcept;
  /// Copies canonical probabilistic Hough line segments from a grayscale mask.
  std::unique_ptr<std::vector<std::int64_t>> line_segments_owned(std::int64_t vote_threshold, std::int64_t minimum_length, std::int64_t maximum_gap) const noexcept;
  /// Draws one bounded grayscale line into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> draw_line_owned(std::int64_t start_x, std::int64_t start_y, std::int64_t end_x, std::int64_t end_y, std::int64_t value, std::int64_t thickness) const noexcept;
  /// Draws one bounded grayscale rectangle into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> draw_rectangle_owned(std::int64_t x, std::int64_t y, std::int64_t width, std::int64_t height, std::int64_t value, std::int64_t thickness) const noexcept;
  /// Draws one bounded grayscale circle into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> draw_circle_owned(std::int64_t center_x, std::int64_t center_y, std::int64_t radius, std::int64_t value, std::int64_t thickness) const noexcept;
  /// Fills one bounded grayscale rectangle into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> fill_rectangle_owned(std::int64_t x, std::int64_t y, std::int64_t width, std::int64_t height, std::int64_t value) const noexcept;
  /// Fills one bounded grayscale circle into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> fill_circle_owned(std::int64_t center_x, std::int64_t center_y, std::int64_t radius, std::int64_t value) const noexcept;
  /// Inverts every scalar bit pattern into a fresh typed matrix owner.
  std::unique_ptr<OpenCvMat> invert_owned() const noexcept;
  /// Inverts selected scalar bit patterns through a same-size byte mask.
  std::unique_ptr<OpenCvMat> invert_masked_owned(const OpenCvMat& mask) const noexcept;
  /// Applies bitwise conjunction to two compatible typed matrices.
  std::unique_ptr<OpenCvMat> bitwise_and_owned(const OpenCvMat& other) const noexcept;
  /// Applies masked bitwise conjunction to compatible typed matrices.
  std::unique_ptr<OpenCvMat> bitwise_and_masked_owned(const OpenCvMat& other, const OpenCvMat& mask) const noexcept;
  /// Applies bitwise disjunction to two compatible typed matrices.
  std::unique_ptr<OpenCvMat> bitwise_or_owned(const OpenCvMat& other) const noexcept;
  /// Applies masked bitwise disjunction to compatible typed matrices.
  std::unique_ptr<OpenCvMat> bitwise_or_masked_owned(const OpenCvMat& other, const OpenCvMat& mask) const noexcept;
  /// Applies bitwise exclusive-or to two compatible typed matrices.
  std::unique_ptr<OpenCvMat> bitwise_xor_owned(const OpenCvMat& other) const noexcept;
  /// Applies masked bitwise exclusive-or to compatible typed matrices.
  std::unique_ptr<OpenCvMat> bitwise_xor_masked_owned(const OpenCvMat& other, const OpenCvMat& mask) const noexcept;
  /// Draws one bounded grayscale cross into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> draw_cross_owned(std::int64_t center_x, std::int64_t center_y, std::int64_t size, std::int64_t value, std::int64_t thickness) const noexcept;
  /// Draws one bounded axis-aligned grayscale ellipse into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> draw_ellipse_owned(std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x, std::int64_t radius_y, std::int64_t value, std::int64_t thickness) const noexcept;
  /// Fills one bounded axis-aligned grayscale ellipse into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> fill_ellipse_owned(std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x, std::int64_t radius_y, std::int64_t value) const noexcept;
  /// Applies bounded local contrast equalization into a fresh matrix owner.
  std::unique_ptr<OpenCvMat> adaptive_equalize_hist_owned(double clip_limit, std::int64_t tiles_x, std::int64_t tiles_y) const noexcept;
  /// Copies one bounded grayscale Harris corner-response map.
  std::unique_ptr<std::vector<double>> harris_response_owned(std::int64_t block_size, std::int64_t aperture_size, double response_k) const noexcept;
  /// Copies one bounded grayscale minimum-eigenvalue corner-response map.
  std::unique_ptr<std::vector<double>> corner_min_eigenvalue_owned(std::int64_t block_size, std::int64_t aperture_size) const noexcept;
  /// Copies one bounded grayscale pre-corner derivative-response map.
  std::unique_ptr<std::vector<double>> pre_corner_response_owned(std::int64_t aperture_size) const noexcept;
  /// Copies bounded grayscale corner eigenvalues and eigenvectors.
  std::unique_ptr<std::vector<double>> corner_eigensystem_owned(std::int64_t block_size, std::int64_t aperture_size) const noexcept;
  /// Detects bounded grayscale circles and copies center/radius triples.
  std::unique_ptr<std::vector<double>> detect_circles_owned(double minimum_distance, double edge_threshold, double center_threshold, std::int64_t minimum_radius, std::int64_t maximum_radius) const noexcept;
  /// Detects bounded grayscale lines and copies canonical rho/theta pairs.
  std::unique_ptr<std::vector<double>> detect_lines_owned(std::int64_t vote_threshold) const noexcept;
  /// Detects bounded FAST keypoints and copies canonical coordinates.
  std::unique_ptr<std::vector<double>> fast_keypoints_owned(std::int64_t threshold, bool nonmax_suppression, std::int64_t max_keypoints) const noexcept;
  /// Extracts bounded ORB keypoints and descriptors into one owner.
  std::unique_ptr<OpenCvOrbFeatures> orb_features_owned(std::int64_t max_features) const noexcept;
  /// Extracts bounded BRISK keypoints and descriptors into one owner.
  std::unique_ptr<OpenCvBriskFeatures> brisk_features_owned(std::int64_t threshold, std::int64_t octaves, std::int64_t max_features) const noexcept;
  /// Extracts bounded fixed-policy AKAZE keypoints and descriptors.
  std::unique_ptr<OpenCvAkazeFeatures> akaze_features_owned(double threshold, std::int64_t max_features) const noexcept;
  /// Detects bounded OAST-9-16 AGAST keypoints and copies canonical coordinates.
  std::unique_ptr<std::vector<double>> agast_keypoints_owned(std::int64_t threshold, bool nonmax_suppression, std::int64_t max_keypoints) const noexcept;
  /// Extracts bounded fixed-policy SIFT keypoints and descriptors.
  std::unique_ptr<OpenCvSiftFeatures> sift_features_owned(double contrast_threshold, std::int64_t max_features) const noexcept; /** Extracts bounded affine-invariant SIFT keypoints and descriptors. */ std::unique_ptr<OpenCvSiftFeatures> affine_sift_features_owned(double contrast_threshold, std::int64_t max_features) const noexcept;
  /// Extracts bounded fixed-policy KAZE keypoints and descriptors.
  std::unique_ptr<OpenCvKazeFeatures> kaze_features_owned(double threshold, std::int64_t max_features) const noexcept;
  /// Detects bounded maximally stable extremal regions.
  std::unique_ptr<OpenCvMserRegions> mser_regions_owned(std::int64_t delta, std::int64_t minimum_area, std::int64_t maximum_area, std::int64_t max_regions) const noexcept;
  /// Detects bounded grayscale blobs and copies canonical keypoint rows.
  std::unique_ptr<std::vector<double>> simple_blob_keypoints_owned(std::int64_t blob_value, std::int64_t minimum_area, std::int64_t maximum_area, double minimum_distance, std::int64_t max_blobs) const noexcept; /** Compensates bounded radial and tangential lens distortion. */ std::unique_ptr<OpenCvMat> undistort_owned(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3) const noexcept; /** Compensates lens distortion using OpenCV's optimal camera matrix. */ std::unique_ptr<OpenCvMat> undistort_optimal_owned(double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2, double k3, double alpha, bool center_principal_point) const noexcept;
 private:
  std::unique_ptr<Impl> impl_;
};
/// Defines supported image-border policies.
enum class BorderMode : std::int32_t {
  Constant = 0,
  Replicate = 1,
  Reflect = 2,
};
/// Defines supported image flip axes.
enum class FlipMode : std::int32_t {
  Both = -1,
  Vertical = 0,
  Horizontal = 1,
};
/// Defines supported right-angle image rotations.
enum class RotateMode : std::int32_t {
  Clockwise90 = 0,
  Rotate180 = 1,
  CounterClockwise90 = 2,
};
/// Defines supported fixed threshold operations.
enum class ThresholdMode : std::int32_t {
  Binary = 0,
  BinaryInverse = 1,
  Truncate = 2,
  ToZero = 3,
  ToZeroInverse = 4,
};
/// Defines supported adaptive-threshold neighborhood calculations.
enum class AdaptiveThresholdMethod : std::int32_t {
  Mean = 0,
  Gaussian = 1,
};
/// Defines supported compound morphology operations.
enum class MorphologyOperation : std::int32_t {
  Open = 0,
  Close = 1,
  Gradient = 2,
  TopHat = 3,
  BlackHat = 4,
};
/// Defines supported dense linear-system decomposition policies.
enum class DecompositionMethod : std::int32_t {
  Lu = 0,
  Svd = 1,
  Eigen = 2,
  Cholesky = 3,
  Qr = 4,
  NormalLu = 5,
  NormalSvd = 6,
};
/// Defines supported matrix norm policies.
enum class MatrixNorm : std::int32_t {
  L1 = 0,
  L2 = 1,
  Infinity = 2,
};
/// Defines supported portable in-memory matrix serialization formats.
enum class SerializationFormat : std::int32_t {
  Yaml = 0,
  Xml = 1,
  Json = 2,
};
/// Defines the matrix axis collapsed by a reduction.
enum class ReductionDimension : std::int32_t {
  AcrossRows = 0,
  AcrossColumns = 1,
};
/// Defines supported matrix reduction operations.
enum class ReductionOperation : std::int32_t {
  Sum = 0,
  Average = 1,
  Maximum = 2,
  Minimum = 3,
};
/// Defines whether sorting is performed independently by row or column.
enum class SortDimension : std::int32_t {
  EachRow = 0,
  EachColumn = 1,
};
/// Defines the ordering applied by matrix sorting.
enum class SortOrder : std::int32_t {
  Ascending = 0,
  Descending = 1,
};
/// Copies unsigned bytes into a newly owned matrix.
std::unique_ptr<OpenCvMat> mat_from_u8(
    rust::Slice<const std::uint8_t> bytes, std::int64_t rows,
    std::int64_t cols, std::int64_t layout) noexcept;
/// Copies raw scalar bytes into a typed newly owned matrix.
std::unique_ptr<OpenCvMat> mat_from_bytes(
    rust::Slice<const std::uint8_t> bytes, std::int64_t rows,
    std::int64_t cols, std::int64_t channels,
    std::int64_t depth) noexcept;
/// Copies a pointer-free contiguous tensor packet into a newly owned matrix.
std::unique_ptr<OpenCvMat> mat_from_tensor_packet(
    rust::Slice<const std::uint8_t> packet) noexcept;
/// Creates a zero-filled owned matrix.
std::unique_ptr<OpenCvMat> mat_zeros(
    std::int64_t rows, std::int64_t cols, std::int64_t layout) noexcept;
/// Creates a typed zero-filled owned matrix.
std::unique_ptr<OpenCvMat> mat_zeros_typed(
    std::int64_t rows, std::int64_t cols, std::int64_t channels,
    std::int64_t depth) noexcept;
/// Creates an all-one typed matrix.
std::unique_ptr<OpenCvMat> mat_ones_typed(
    std::int64_t rows, std::int64_t cols, std::int64_t channels,
    std::int64_t depth) noexcept;
/// Creates one single-channel typed identity matrix.
std::unique_ptr<OpenCvMat> mat_identity_typed(
    std::int64_t rows, std::int64_t cols, std::int64_t depth) noexcept;
/// Creates deterministic uniformly distributed floating-point values.
std::unique_ptr<OpenCvMat> mat_random_uniform(
    std::int64_t rows, std::int64_t cols, std::int64_t depth, double lower,
    double upper, std::int64_t seed) noexcept;
/// Creates deterministic normally distributed floating-point values.
std::unique_ptr<OpenCvMat> mat_random_normal(
    std::int64_t rows, std::int64_t cols, std::int64_t depth, double mean,
    double standard_deviation, std::int64_t seed) noexcept;
/// Parses one bounded in-memory OpenCV matrix document.
std::unique_ptr<OpenCvMat> mat_deserialize(
    rust::Str document, std::int64_t format) noexcept;
/// Copies the linked OpenCV runtime version.
std::unique_ptr<std::string> opencv_version_owned() noexcept;
/// Copies bounded OpenCV build configuration information.
std::unique_ptr<std::string> opencv_build_information_owned() noexcept;
/// Returns the logical processor count observed by OpenCV.
std::int64_t opencv_processor_count() noexcept;
/// Returns the current OpenCV parallel worker count.
std::int64_t opencv_thread_count() noexcept;
/// Returns the monotonic OpenCV tick frequency in ticks per second.
double opencv_tick_frequency() noexcept;
/// Reports whether optimized OpenCV code paths are enabled.
bool opencv_optimized_enabled() noexcept;
/// Decodes copied BMP, PNG, or JPEG bytes into a newly owned matrix.
std::unique_ptr<OpenCvMat> decode_image(
    rust::Slice<const std::uint8_t> bytes, std::int64_t mode) noexcept;
/// Returns the number of live adapter-owned matrices.
std::int64_t live_mat_count() noexcept;
/// Returns and clears the current thread's finite native failure status.
std::int64_t take_last_failure() noexcept;
/// Holds one copied two-dimensional perspective transform in row-major order.
struct PerspectiveTransform final {
  double m00; /** Returns m00. */ double m00_value() const noexcept { return m00; }
  double m01; /** Returns m01. */ double m01_value() const noexcept { return m01; }
  double m02; /** Returns m02. */ double m02_value() const noexcept { return m02; }
  double m10; /** Returns m10. */ double m10_value() const noexcept { return m10; }
  double m11; /** Returns m11. */ double m11_value() const noexcept { return m11; }
  double m12; /** Returns m12. */ double m12_value() const noexcept { return m12; }
  double m20; /** Returns m20. */ double m20_value() const noexcept { return m20; }
  double m21; /** Returns m21. */ double m21_value() const noexcept { return m21; }
  double m22; /** Returns m22. */ double m22_value() const noexcept { return m22; }
};
/// Defines supported connected-component neighborhood policies.
enum class Connectivity : std::int32_t {
  Four = 4,
  Eight = 8,
};

/// Uniquely owns one connected-component analysis result.
class OpenCvComponents final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvComponents(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned component analysis.
  ~OpenCvComponents() noexcept;

  /// Returns the number of foreground components.
  std::int64_t foreground_count() const noexcept;
  /// Copies row-major labels, including zero-valued background labels.
  std::unique_ptr<std::vector<std::int64_t>> labels_owned() const noexcept;
  /// Copies foreground component areas in label order.
  std::unique_ptr<std::vector<std::int64_t>> areas_owned() const noexcept;
  /// Copies flattened x, y, width, and height foreground bounds.
  std::unique_ptr<std::vector<std::int64_t>> bounds_owned() const noexcept;
  /// Copies flattened x and y foreground centroids in label order.
  std::unique_ptr<std::vector<double>> centroids_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned component analyses.
std::int64_t live_component_analysis_count() noexcept;

/// Uniquely owns one simplified external-contour analysis.
class OpenCvContours final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvContours(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned contour analysis.
  ~OpenCvContours() noexcept;

  /// Returns the number of external contours.
  std::int64_t count() const noexcept;
  /// Copies flattened x and y coordinates in contour order.
  std::unique_ptr<std::vector<std::int64_t>> points_owned() const noexcept;
  /// Copies cumulative point offsets, including the initial zero.
  std::unique_ptr<std::vector<std::int64_t>> offsets_owned() const noexcept;
  /// Copies flattened x, y, width, and height contour bounds.
  std::unique_ptr<std::vector<std::int64_t>> bounds_owned() const noexcept;
  /// Copies contour areas in contour order.
  std::unique_ptr<std::vector<double>> areas_owned() const noexcept;
  /// Copies closed contour perimeters in contour order.
  std::unique_ptr<std::vector<double>> perimeters_owned() const noexcept;
  /// Copies convex-hull areas in contour order.
  std::unique_ptr<std::vector<double>> convex_areas_owned() const noexcept;
  /// Copies normalized contour solidities in contour order.
  std::unique_ptr<std::vector<double>> solidities_owned() const noexcept;
  /// Copies normalized contour circularities in contour order.
  std::unique_ptr<std::vector<double>> circularities_owned() const noexcept;
  /// Copies normalized contour extents in contour order.
  std::unique_ptr<std::vector<double>> extents_owned() const noexcept;
  /// Copies flattened geometric contour centroids in contour order.
  std::unique_ptr<std::vector<double>> centroids_owned() const noexcept;
  /// Copies seven raw Hu invariants per contour in contour order.
  std::unique_ptr<std::vector<double>> hu_moments_owned() const noexcept;
  /// Copies four canonical minimum-area-box corners per contour.
  std::unique_ptr<std::vector<double>> minimum_area_boxes_owned() const noexcept;
  /// Copies one minimum-enclosing-circle center and radius per contour.
  std::unique_ptr<std::vector<double>> minimum_enclosing_circles_owned() const noexcept;
  /// Copies three canonical minimum-enclosing-triangle vertices per contour.
  std::unique_ptr<std::vector<double>> minimum_enclosing_triangles_owned() const noexcept;
  /// Copies one canonical fitted-line record per contour.
  std::unique_ptr<std::vector<double>> fitted_lines_owned() const noexcept;
  /// Copies one canonical fitted-ellipse record per contour.
  std::unique_ptr<std::vector<double>> fitted_ellipses_owned() const noexcept;
  /// Copies three shape-match scores per contour against one reference index.
  std::unique_ptr<std::vector<double>> shape_match_scores_owned(
      std::int64_t reference_index) const noexcept;
  /// Creates an independently owned Douglas-Peucker approximation.
  std::unique_ptr<OpenCvContours> approximate_owned(double epsilon) const noexcept;
  /// Creates independently owned convex hulls for every contour.
  std::unique_ptr<OpenCvContours> convex_hulls_owned() const noexcept;
  /// Copies convexity-defect geometry for one contour index.
  std::unique_ptr<std::vector<double>> convexity_defects_owned(
      std::int64_t contour_index) const noexcept;
  /// Copies one signed point-to-polygon distance per contour.
  std::unique_ptr<std::vector<double>> signed_distances_owned(
      double x, double y) const noexcept;
  /// Renders every contour into a fresh source-sized grayscale mask.
  std::unique_ptr<OpenCvMat> filled_mask_owned(std::int64_t value) const noexcept;
  /// Renders every contour outline into a fresh grayscale mask.
  std::unique_ptr<OpenCvMat> outline_mask_owned(
      std::int64_t value, std::int64_t thickness) const noexcept;
  /// Renders one selected filled contour into a fresh grayscale mask.
  std::unique_ptr<OpenCvMat> selected_filled_mask_owned(
      std::int64_t contour_index, std::int64_t value) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned contour analyses.
std::int64_t live_contour_analysis_count() noexcept;

/// Defines supported grayscale image comparison operations.
enum class ComparisonMode : std::int32_t {
  Equal = 0,
  NotEqual = 1,
  Less = 2,
  LessOrEqual = 3,
  Greater = 4,
  GreaterOrEqual = 5,
};

class OpenCvOrbMatches;

/// Uniquely owns canonical ORB keypoints and their descriptor rows.
class OpenCvOrbFeatures final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvOrbFeatures(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned ORB analysis.
  ~OpenCvOrbFeatures() noexcept;

  /// Returns the number of keypoint and descriptor rows.
  std::int64_t count() const noexcept;
  /// Copies flattened x and y coordinates in descriptor-row order.
  std::unique_ptr<std::vector<double>> keypoints_owned() const noexcept;
  /// Returns the fixed number of bytes in each ORB descriptor row.
  std::int64_t descriptor_width() const noexcept;
  /// Copies row-major ORB descriptor bytes in keypoint order.
  std::unique_ptr<std::vector<std::uint8_t>> descriptors_owned() const noexcept;
  /// Cross-checks descriptors through bounded Hamming matching.
  std::unique_ptr<OpenCvOrbMatches> match_owned(
      const OpenCvOrbFeatures& other, std::int64_t max_matches) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned ORB analyses.
std::int64_t live_orb_feature_count() noexcept;

/// Uniquely owns canonical cross-checked ORB matches.
class OpenCvOrbMatches final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvOrbMatches(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned ORB match analysis.
  ~OpenCvOrbMatches() noexcept;

  /// Returns the number of retained matches.
  std::int64_t count() const noexcept;
  /// Copies flattened query and train descriptor-row indices.
  std::unique_ptr<std::vector<std::int64_t>> indices_owned() const noexcept;
  /// Copies integral Hamming distances in canonical match order.
  std::unique_ptr<std::vector<std::int64_t>> distances_owned() const noexcept;
  /// Copies query and train x/y endpoints in canonical match order.
  std::unique_ptr<std::vector<double>> coordinates_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned ORB match analyses.
std::int64_t live_orb_match_count() noexcept;

/// Uniquely owns canonical BRISK keypoints and descriptor rows.
class OpenCvBriskFeatures final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvBriskFeatures(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned BRISK analysis.
  ~OpenCvBriskFeatures() noexcept;

  /// Returns the number of keypoint and descriptor rows.
  std::int64_t count() const noexcept;
  /// Copies flattened x and y coordinates in descriptor-row order.
  std::unique_ptr<std::vector<double>> keypoints_owned() const noexcept;
  /// Returns the fixed number of bytes in each BRISK descriptor row.
  std::int64_t descriptor_width() const noexcept;
  /// Copies row-major BRISK descriptor bytes in keypoint order.
  std::unique_ptr<std::vector<std::uint8_t>> descriptors_owned() const noexcept;
  /// Cross-checks BRISK descriptors through bounded Hamming matching.
  std::unique_ptr<OpenCvBriskMatches> match_owned(
      const OpenCvBriskFeatures& other,
      std::int64_t max_matches) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned BRISK analyses.
std::int64_t live_brisk_feature_count() noexcept;

/// Uniquely owns canonical cross-checked BRISK matches.
class OpenCvBriskMatches final {
 public:
  struct Impl;

  /// Constructs an owner from private canonical match records.
  explicit OpenCvBriskMatches(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned BRISK match analysis.
  ~OpenCvBriskMatches() noexcept;

  /// Returns the number of retained matches.
  std::int64_t count() const noexcept;
  /// Copies flattened query and train descriptor-row indices.
  std::unique_ptr<std::vector<std::int64_t>> indices_owned() const noexcept;
  /// Copies integral Hamming distances in canonical match order.
  std::unique_ptr<std::vector<std::int64_t>> distances_owned() const noexcept;
  /// Copies query and train x/y endpoints in canonical match order.
  std::unique_ptr<std::vector<double>> coordinates_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned BRISK match analyses.
std::int64_t live_brisk_match_count() noexcept;

class OpenCvAkazeMatches;

/// Uniquely owns canonical AKAZE keypoints and descriptor rows.
class OpenCvAkazeFeatures final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvAkazeFeatures(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned AKAZE analysis.
  ~OpenCvAkazeFeatures() noexcept;

  /// Returns the number of keypoint and descriptor rows.
  std::int64_t count() const noexcept;
  /// Copies flattened x and y coordinates in descriptor-row order.
  std::unique_ptr<std::vector<double>> keypoints_owned() const noexcept;
  /// Returns the fixed number of bytes in each AKAZE descriptor row.
  std::int64_t descriptor_width() const noexcept;
  /// Copies row-major AKAZE descriptor bytes in keypoint order.
  std::unique_ptr<std::vector<std::uint8_t>> descriptors_owned() const noexcept;
  /// Cross-checks AKAZE descriptors through bounded Hamming matching.
  std::unique_ptr<OpenCvAkazeMatches> match_owned(
      const OpenCvAkazeFeatures& other,
      std::int64_t max_matches) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned AKAZE analyses.
std::int64_t live_akaze_feature_count() noexcept;

/// Uniquely owns canonical cross-checked AKAZE matches.
class OpenCvAkazeMatches final {
 public:
  struct Impl;

  /// Constructs an owner from private canonical match records.
  explicit OpenCvAkazeMatches(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned AKAZE match analysis.
  ~OpenCvAkazeMatches() noexcept;

  /// Returns the number of retained matches.
  std::int64_t count() const noexcept;
  /// Copies flattened query and train descriptor-row indices.
  std::unique_ptr<std::vector<std::int64_t>> indices_owned() const noexcept;
  /// Copies integral Hamming distances in canonical match order.
  std::unique_ptr<std::vector<std::int64_t>> distances_owned() const noexcept;
  /// Copies query and train x/y endpoints in canonical match order.
  std::unique_ptr<std::vector<double>> coordinates_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned AKAZE match analyses.
std::int64_t live_akaze_match_count() noexcept;

class OpenCvSiftMatches;

/// Uniquely owns canonical SIFT keypoints and floating-point descriptors.
class OpenCvSiftFeatures final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvSiftFeatures(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned SIFT analysis.
  ~OpenCvSiftFeatures() noexcept;

  /// Returns the number of keypoint and descriptor rows.
  std::int64_t count() const noexcept;
  /// Copies flattened x and y coordinates in descriptor-row order.
  std::unique_ptr<std::vector<double>> keypoints_owned() const noexcept;
  /// Returns the fixed number of floating-point elements per descriptor.
  std::int64_t descriptor_width() const noexcept;
  /// Copies row-major SIFT descriptor values in keypoint order.
  std::unique_ptr<std::vector<double>> descriptors_owned() const noexcept;
  /// Cross-checks descriptors through bounded L2 matching.
  std::unique_ptr<OpenCvSiftMatches> match_owned(
      const OpenCvSiftFeatures& other,
      std::int64_t max_matches) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned SIFT analyses.
std::int64_t live_sift_feature_count() noexcept;

/// Uniquely owns canonical cross-checked SIFT descriptor matches.
class OpenCvSiftMatches final {
 public:
  struct Impl;

  /// Constructs an owner from canonical copied match records.
  explicit OpenCvSiftMatches(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned SIFT match analysis.
  ~OpenCvSiftMatches() noexcept;

  /// Returns the number of retained matches.
  std::int64_t count() const noexcept;
  /// Copies flattened query and train descriptor-row indices.
  std::unique_ptr<std::vector<std::int64_t>> indices_owned() const noexcept;
  /// Copies finite L2 distances in canonical match order.
  std::unique_ptr<std::vector<double>> distances_owned() const noexcept;
  /// Copies query and train x/y endpoints in canonical match order.
  std::unique_ptr<std::vector<double>> coordinates_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned SIFT match analyses.
std::int64_t live_sift_match_count() noexcept;

class OpenCvKazeMatches;

/// Uniquely owns canonical KAZE keypoints and floating-point descriptors.
class OpenCvKazeFeatures final {
 public:
  struct Impl;

  /// Constructs an owner from its private OpenCV representation.
  explicit OpenCvKazeFeatures(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned KAZE analysis.
  ~OpenCvKazeFeatures() noexcept;

  /// Returns the number of keypoint and descriptor rows.
  std::int64_t count() const noexcept;
  /// Copies flattened x and y coordinates in descriptor-row order.
  std::unique_ptr<std::vector<double>> keypoints_owned() const noexcept;
  /// Returns the fixed number of floating-point elements per descriptor.
  std::int64_t descriptor_width() const noexcept;
  /// Copies row-major KAZE descriptor values in keypoint order.
  std::unique_ptr<std::vector<double>> descriptors_owned() const noexcept;
  /// Cross-checks descriptors through bounded L2 matching.
  std::unique_ptr<OpenCvKazeMatches> match_owned(
      const OpenCvKazeFeatures& other,
      std::int64_t max_matches) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned KAZE analyses.
std::int64_t live_kaze_feature_count() noexcept;

/// Uniquely owns canonical cross-checked KAZE descriptor matches.
class OpenCvKazeMatches final {
 public:
  struct Impl;

  /// Constructs an owner from canonical copied match records.
  explicit OpenCvKazeMatches(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned KAZE match analysis.
  ~OpenCvKazeMatches() noexcept;

  /// Returns the number of retained matches.
  std::int64_t count() const noexcept;
  /// Copies flattened query and train descriptor-row indices.
  std::unique_ptr<std::vector<std::int64_t>> indices_owned() const noexcept;
  /// Copies finite L2 distances in canonical match order.
  std::unique_ptr<std::vector<double>> distances_owned() const noexcept;
  /// Copies query and train x/y endpoints in canonical match order.
  std::unique_ptr<std::vector<double>> coordinates_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned KAZE match analyses.
std::int64_t live_kaze_match_count() noexcept;

/// Uniquely owns canonical maximally stable extremal regions.
class OpenCvMserRegions final {
 public:
  struct Impl;

  /// Constructs an owner from validated canonical regions and bounds.
  explicit OpenCvMserRegions(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned MSER analysis.
  ~OpenCvMserRegions() noexcept;

  /// Returns the number of retained extremal regions.
  std::int64_t count() const noexcept;
  /// Copies flattened row-major x/y points for every retained region.
  std::unique_ptr<std::vector<std::int64_t>> points_owned() const noexcept;
  /// Copies cumulative region-point offsets including the initial zero.
  std::unique_ptr<std::vector<std::int64_t>> offsets_owned() const noexcept;
  /// Copies flattened x/y/width/height bounds in region order.
  std::unique_ptr<std::vector<std::int64_t>> bounds_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live adapter-owned MSER analyses.
std::int64_t live_mser_region_count() noexcept;

/// Holds copied pinhole-camera intrinsic parameters.
struct CameraIntrinsics final {
  double fx;  ///< Horizontal focal length.
  double fy;  ///< Vertical focal length.
  double cx;  ///< Principal-point x coordinate.
  double cy;  ///< Principal-point y coordinate.
};

/// Holds copied five-coefficient lens distortion parameters.
struct DistortionCoefficients final {
  double k1;  ///< First radial coefficient.
  double k2;  ///< Second radial coefficient.
  double p1;  ///< First tangential coefficient.
  double p2;  ///< Second tangential coefficient.
  double k3;  ///< Third radial coefficient.
};

/// Corrects bounded flattened image points into the same pixel coordinate space.
std::unique_ptr<std::vector<double>> undistort_points_owned(
    rust::Slice<const double> points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept;

/// Applies one bounded perspective transform to flattened image points.
std::unique_ptr<std::vector<double>> perspective_transform_points_owned(
    rust::Slice<const double> points, double m00, double m01, double m02,
    double m10, double m11, double m12, double m20, double m21,
    double m22) noexcept;

/// Applies one bounded affine transform to flattened image points.
std::unique_ptr<std::vector<double>> affine_transform_points_owned(
    rust::Slice<const double> points, double m00, double m01, double m02,
    double m10, double m11, double m12) noexcept;

/// Estimates a copied perspective transform from four point correspondences.
std::unique_ptr<PerspectiveTransform> estimate_perspective_transform_owned(
    rust::Slice<const double> source_points,
    rust::Slice<const double> destination_points) noexcept;

/// Estimates a copied affine transform from three point correspondences.
std::unique_ptr<AffineTransform> estimate_affine_transform_owned(
    rust::Slice<const double> source_points,
    rust::Slice<const double> destination_points) noexcept;

/// Holds one copied three-dimensional vector.
struct Vector3 final {
  double x;  ///< X component.
  double y;  ///< Y component.
  double z;  ///< Z component.
};

/// Projects bounded three-dimensional points into image coordinates.
std::unique_ptr<std::vector<double>> project_points_owned(
    rust::Slice<const double> object_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3) noexcept;

/// Holds one copied camera pose as Rodrigues rotation and translation vectors.
struct CameraPose final {
  double rx; /** Returns rx. */ double rx_value() const noexcept { return rx; }
  double ry; /** Returns ry. */ double ry_value() const noexcept { return ry; }
  double rz; /** Returns rz. */ double rz_value() const noexcept { return rz; }
  double tx; /** Returns tx. */ double tx_value() const noexcept { return tx; }
  double ty; /** Returns ty. */ double ty_value() const noexcept { return ty; }
  double tz; /** Returns tz. */ double tz_value() const noexcept { return tz; }
};

/// Estimates one copied camera pose from matching 3D and 2D points.
std::unique_ptr<CameraPose> solve_pose_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept;

/// Refines one copied camera pose from an explicit finite initial estimate.
std::unique_ptr<CameraPose> refine_pose_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3) noexcept;

/// Refines one copied camera pose with fixed virtual visual servoing policy.
std::unique_ptr<CameraPose> refine_pose_vvs_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3) noexcept;

/// Composes two copied camera poses, applying the first pose then the second.
std::unique_ptr<CameraPose> compose_pose_owned(
    double first_rx, double first_ry, double first_rz, double first_tx,
    double first_ty, double first_tz, double second_rx, double second_ry,
    double second_rz, double second_tx, double second_ty,
    double second_tz) noexcept;

/// Estimates one copied camera pose with fixed robust outlier rejection.
std::unique_ptr<CameraPose> solve_pose_robust_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept;

/// Estimates the best copied camera pose from exactly four correspondences.
std::unique_ptr<CameraPose> solve_pose_minimal_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept;

/// Estimates the lowest-error copied pose for coplanar observations.
std::unique_ptr<CameraPose> solve_pose_planar_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept;

/// Estimates the lowest-error copied pose for one square marker.
std::unique_ptr<CameraPose> solve_square_marker_pose_owned(
    rust::Slice<const double> image_points, double marker_size, double fx,
    double fy, double cx, double cy, double k1, double k2, double p1,
    double p2, double k3) noexcept;
}  // namespace terlan_opencv

#include "opencv_dnn.hpp"
