#ifndef TERLAN_OPENCV_VALIDATION_HPP
#define TERLAN_OPENCV_VALIDATION_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace terlan_opencv {
namespace validation {

/// Maximum byte size of one image allocation owned by this package.
constexpr std::size_t kMaxOwnedImageBytes = 256U * 1024U * 1024U;

/// Maximum channel count encoded by OpenCV's public matrix type representation.
constexpr int kMaxMatrixChannels = 512;

/// Maximum byte size of one encoded image entering or leaving this package.
constexpr std::size_t kMaxEncodedImageBytes = 64U * 1024U * 1024U;

/// Deterministic JPEG quality used by the package encoder.
constexpr int kJpegEncodeQuality = 90;

/// Deterministic PNG compression level used by the package encoder.
constexpr int kPngCompressionLevel = 3;

/// Deterministic WebP quality used by the package encoder.
constexpr int kWebpEncodeQuality = 90;

/// Deterministic uncompressed TIFF policy used by the package encoder.
constexpr int kTiffCompressionNone = 1;

/// Maximum supported Gaussian-kernel dimension.
constexpr std::int64_t kMaxKernelDimension = 255;

/// Maximum supported morphology iteration count.
constexpr std::int64_t kMaxMorphologyIterations = 64;

/// Maximum supported Canny derivative threshold.
constexpr double kMaxCannyThreshold = 65535.0;

/// Maximum accepted bilateral-filter color or spatial sigma.
constexpr double kMaxFilterSigma = 65535.0;

/// Maximum accepted spatial or color radius for mean-shift filtering.
constexpr double kMaxMeanShiftRadius = 65535.0;

/// Maximum accepted Gaussian-pyramid level for mean-shift filtering.
constexpr std::int64_t kMaxMeanShiftPyramidLevel = 8;

/// Maximum magnitude of one scale-and-absolute-value parameter.

/// Maximum accepted semantic sharpening amount.
constexpr double kMaxSharpenAmount = 16.0;

/// Maximum absolute constant subtracted by adaptive thresholding.
constexpr double kMaxAdaptiveThresholdConstant = 255.0;

/// Maximum absolute transform coefficient accepted by the package adapter.
constexpr double kMaxTransformCoefficient = 1000000.0;

/// Maximum number of copied two-dimensional points in one native operation.
constexpr std::size_t kMaxCopiedPointCount = 4096;

/// Maximum pixel count whose histogram counters remain exact in OpenCV floats.
constexpr std::int64_t kMaxExactHistogramPixels = 16777216;

/// Maximum pixel count copied from one image-analysis result.
constexpr std::int64_t kMaxCopiedAnalysisPixels = 1048576;
static_assert(
    kMaxCopiedAnalysisPixels <=
        std::numeric_limits<std::int32_t>::max() / 255,
    "copied analysis pixels must remain exact in signed 32-bit sums");

/// Maximum channel-pixel comparisons performed by one template match.
constexpr std::int64_t kMaxTemplateMatchWork = 64 * 1024 * 1024;

/// Maximum number of point features copied from one detector invocation.
constexpr std::int64_t kMaxDetectedCorners = 4096;

/// Maximum number of canonical line segments copied from one detector call.
constexpr std::size_t kMaxDetectedLineSegments = 4096U;

/// Maximum number of canonical lines copied from one detector call.
constexpr std::size_t kMaxDetectedLines = 4096U;

/// Maximum number of canonical circles copied from one detector call.
constexpr std::size_t kMaxDetectedCircles = 4096U;

/// Maximum accepted point-separation distance in image coordinates.
constexpr double kMaxFeatureDistance = 1000000.0;

/// Maximum accepted odd corner-response neighborhood dimension.
constexpr std::int64_t kMaxCornerBlockSize = 31;

/// Maximum accepted Harris free response coefficient.
constexpr double kMaxHarrisResponseK = 0.25;

/// Maximum absolute coefficient accepted by weighted image arithmetic.
constexpr double kMaxImageWeight = 16.0;

/// Maximum absolute additive offset accepted by weighted image arithmetic.
constexpr double kMaxImageOffset = 255.0;

/// Describes a checked image allocation.
struct ImageShape final {
  /// Whether all dimensions and the resulting allocation are valid.
  bool valid;
  /// Exact byte count required by the image.
  std::size_t byte_count;
  /// Number of channels represented by each pixel.
  int channels;
};

/// Describes the validated output of a color conversion.
struct ColorConversionShape final {
  /// Whether the conversion accepts the supplied input image.
  bool valid;
  /// Number of channels produced by the conversion.
  int output_channels;
  /// Exact output byte count.
  std::size_t output_byte_count;
};

/// Returns the channel count for one public pixel-layout discriminant.
constexpr int layout_channels(std::int64_t layout) noexcept {
  return layout == 1 ? 1 : (layout == 3 ? 3 : (layout == 4 ? 4 : 0));
}

/// Checks typed dimensions and computes an overflow-safe storage byte count.
constexpr ImageShape matrix_shape(std::int64_t rows, std::int64_t cols,
                                  int channels,
                                  std::size_t element_size) noexcept {
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (rows <= 0 || cols <= 0 || rows > max_dimension ||
      cols > max_dimension || channels <= 0 ||
      channels > kMaxMatrixChannels || element_size == 0) {
    return {false, 0, 0};
  }
  const auto row_count = static_cast<std::size_t>(rows);
  const auto col_count = static_cast<std::size_t>(cols);
  const auto channel_count = static_cast<std::size_t>(channels);
  if (row_count > kMaxOwnedImageBytes / col_count) {
    return {false, 0, 0};
  }
  const auto pixel_count = row_count * col_count;
  if (pixel_count > kMaxOwnedImageBytes / channel_count) {
    return {false, 0, 0};
  }
  const auto scalar_count = pixel_count * channel_count;
  if (scalar_count > kMaxOwnedImageBytes / element_size) {
    return {false, 0, 0};
  }
  return {true, scalar_count * element_size, channels};
}

/// Checks dimensions and computes an overflow-safe, policy-bounded byte count.
inline ImageShape image_shape(std::int64_t rows, std::int64_t cols,
                              int channels) noexcept {
  return matrix_shape(rows, cols, channels, 1);
}

/// Checks dimensions using one public pixel-layout discriminant.
inline ImageShape image_shape_for_layout(std::int64_t rows,
                                         std::int64_t cols,
                                         std::int64_t layout) noexcept {
  return image_shape(rows, cols, layout_channels(layout));
}

/// Computes the checked default output shape for Gaussian-pyramid downsampling.
inline ImageShape pyramid_down_shape(std::int64_t rows, std::int64_t cols,
                                     int channels) noexcept {
  const auto input = image_shape(rows, cols, channels);
  return input.valid ? image_shape((rows + 1) / 2, (cols + 1) / 2, channels)
                     : ImageShape{false, 0, 0};
}

/// Computes the checked default output shape for Gaussian-pyramid upsampling.
inline ImageShape pyramid_up_shape(std::int64_t rows, std::int64_t cols,
                                   int channels) noexcept {
  const auto input = image_shape(rows, cols, channels);
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (!input.valid || rows > max_dimension / 2 || cols > max_dimension / 2) {
    return {false, 0, 0};
  }
  return image_shape(rows * 2, cols * 2, channels);
}

/// Checks that a copied byte buffer exactly fills a valid image shape.
inline bool valid_buffer_size(const ImageShape& shape,
                              std::size_t actual) noexcept {
  return shape.valid && shape.byte_count == actual;
}

/// Checks the finite decode-mode discriminants exposed by the package.
constexpr bool valid_decode_mode(std::int64_t mode) noexcept {
  return mode == -1 || mode == 0 || mode == 1;
}

/// Checks the finite image-format discriminants exposed by the package.
constexpr bool valid_image_format(std::int64_t format) noexcept {
  return format >= 0 && format <= 4;
}

/// Checks a bounded encoded payload for one reviewed image signature.
inline bool valid_encoded_image_input(const std::uint8_t* bytes,
                                      std::size_t size) noexcept {
  if (bytes == nullptr || size < 2 || size > kMaxEncodedImageBytes) {
    return false;
  }
  const bool bmp = bytes[0] == static_cast<std::uint8_t>('B') &&
                   bytes[1] == static_cast<std::uint8_t>('M');
  const bool jpeg = size >= 3 && bytes[0] == 0xffU && bytes[1] == 0xd8U &&
                    bytes[2] == 0xffU;
  const std::uint8_t png_signature[] = {0x89U, 0x50U, 0x4eU, 0x47U,
                                        0x0dU, 0x0aU, 0x1aU, 0x0aU};
  bool png = size >= sizeof(png_signature);
  for (std::size_t index = 0; png && index < sizeof(png_signature); ++index) {
    png = bytes[index] == png_signature[index];
  }
  const bool webp = size >= 12 && bytes[0] == 'R' && bytes[1] == 'I' &&
                    bytes[2] == 'F' && bytes[3] == 'F' &&
                    bytes[8] == 'W' && bytes[9] == 'E' &&
                    bytes[10] == 'B' && bytes[11] == 'P';
  const bool little_endian_tiff =
      size >= 4 && bytes[0] == 'I' && bytes[1] == 'I' && bytes[2] == 42U &&
      bytes[3] == 0U;
  const bool big_endian_tiff =
      size >= 4 && bytes[0] == 'M' && bytes[1] == 'M' && bytes[2] == 0U &&
      bytes[3] == 42U;
  return bmp || png || jpeg || webp || little_endian_tiff || big_endian_tiff;
}

/// Checks one encoded output size against package policy.
constexpr bool valid_encoded_output_size(std::size_t size) noexcept {
  return size > 0 && size <= kMaxEncodedImageBytes;
}

/// Checks decoded matrix storage without overflowing byte-count arithmetic.
constexpr bool valid_decoded_storage(std::size_t elements,
                                     std::size_t element_size) noexcept {
  return elements > 0 && element_size > 0 &&
         elements <= kMaxOwnedImageBytes / element_size;
}

/// Checks a crop region without overflowing its right or bottom edge.
constexpr bool valid_crop_region(std::int64_t rows, std::int64_t cols,
                                 std::int64_t x, std::int64_t y,
                                 std::int64_t width,
                                 std::int64_t height) noexcept {
  return rows > 0 && cols > 0 && x >= 0 && y >= 0 && width > 0 &&
         height > 0 && x < cols && y < rows && width <= cols - x &&
         height <= rows - y;
}

/// Checks the finite border-mode discriminants exposed by the package.
constexpr bool valid_border_mode(std::int64_t mode) noexcept {
  return mode == 0 || mode == 1 || mode == 2;
}

/// Computes an overflow-safe, policy-bounded padded output shape.
inline ImageShape padded_image_shape(std::int64_t rows, std::int64_t cols,
                                     int channels, std::int64_t top,
                                     std::int64_t bottom, std::int64_t left,
                                     std::int64_t right) noexcept {
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (rows <= 0 || cols <= 0 || channels <= 0 || top < 0 || bottom < 0 ||
      left < 0 || right < 0 || rows > max_dimension || cols > max_dimension ||
      top > max_dimension - rows || left > max_dimension - cols) {
    return {false, 0, 0};
  }
  const auto rows_with_top = rows + top;
  const auto cols_with_left = cols + left;
  if (bottom > max_dimension - rows_with_top ||
      right > max_dimension - cols_with_left) {
    return {false, 0, 0};
  }
  return image_shape(rows_with_top + bottom, cols_with_left + right, channels);
}

/// Checks the finite flip-mode discriminants exposed by the package.
constexpr bool valid_flip_mode(std::int64_t mode) noexcept {
  return mode == -1 || mode == 0 || mode == 1;
}

/// Checks the finite rotation-mode discriminants exposed by the package.
constexpr bool valid_rotate_mode(std::int64_t mode) noexcept {
  return mode == 0 || mode == 1 || mode == 2;
}

/// Checks the finite threshold-mode discriminants exposed by the package.
constexpr bool valid_threshold_mode(std::int64_t mode) noexcept {
  return mode >= 0 && mode <= 4;
}

/// Checks one semantic grayscale comparison-mode discriminant.
constexpr bool valid_comparison_mode(std::int64_t mode) noexcept {
  return mode >= 0 && mode <= 5;
}

/// Checks one finite nonnegative per-element image multiplication scale.
inline bool valid_multiply_scale(double scale) noexcept {
  return std::isfinite(scale) && scale >= 0.0 && scale <= 1.0;
}

/// Checks one finite byte-domain per-element image division scale.
inline bool valid_divide_scale(double scale) noexcept {
  return std::isfinite(scale) && scale >= 0.0 && scale <= 255.0;
}

/// Checks the finite adaptive-threshold method discriminants.
constexpr bool valid_adaptive_threshold_method(std::int64_t method) noexcept {
  return method == 0 || method == 1;
}

/// Checks symbolic threshold modes that produce binary output.
constexpr bool valid_binary_threshold_mode(std::int64_t mode) noexcept {
  return mode == 0 || mode == 1;
}

/// Checks one unsigned-byte threshold scalar without allowing NaN or infinity.
inline bool valid_threshold_value(double value) noexcept {
  return std::isfinite(value) && value >= 0.0 && value <= 255.0;
}

/// Computes the checked output shape for one finite image rotation.
inline ImageShape rotated_image_shape(std::int64_t rows, std::int64_t cols,
                                      int channels,
                                      std::int64_t mode) noexcept {
  if (!valid_rotate_mode(mode)) {
    return {false, 0, 0};
  }
  return mode == 1 ? image_shape(rows, cols, channels)
                   : image_shape(cols, rows, channels);
}

/// Checks one Gaussian-kernel dimension against package policy.
constexpr bool valid_kernel_dimension(std::int64_t value) noexcept {
  return value > 0 && value <= kMaxKernelDimension && value % 2 == 1;
}

/// Checks one morphology iteration count against package policy.
constexpr bool valid_morphology_iterations(std::int64_t value) noexcept {
  return value > 0 && value <= kMaxMorphologyIterations;
}

/// Checks one finite compound morphology operation discriminant.
constexpr bool valid_morphology_operation(std::int64_t operation) noexcept {
  return operation >= 0 && operation <= 4;
}

/// Checks image shape, rectangular kernel, and iterations for morphology.
inline bool valid_morphology(std::int64_t rows, std::int64_t cols,
                             int channels, std::int64_t kernel_width,
                             std::int64_t kernel_height,
                             std::int64_t iterations) noexcept {
  return image_shape(rows, cols, channels).valid &&
         valid_kernel_dimension(kernel_width) &&
         valid_kernel_dimension(kernel_height) &&
         valid_morphology_iterations(iterations);
}

/// Checks one compound morphology configuration against package policy.
inline bool valid_compound_morphology(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t operation, std::int64_t kernel_width,
    std::int64_t kernel_height, std::int64_t iterations) noexcept {
  return valid_morphology_operation(operation) &&
         valid_morphology(rows, cols, channels, kernel_width, kernel_height,
                          iterations);
}

/// Checks one finite Canny derivative threshold against package policy.
inline bool valid_canny_threshold(double value) noexcept {
  return std::isfinite(value) && value >= 0.0 && value <= kMaxCannyThreshold;
}

/// Checks one supported Sobel aperture for Canny edge detection.
constexpr bool valid_canny_aperture(std::int64_t value) noexcept {
  return value == 3 || value == 5 || value == 7;
}

/// Checks one supported Laplacian derivative aperture.
constexpr bool valid_laplacian_kernel(std::int64_t value) noexcept {
  return value == 1 || valid_canny_aperture(value);
}

/// Checks that an image is a valid single-channel unsigned-byte matrix.
inline bool valid_grayscale_image(std::int64_t rows, std::int64_t cols,
                                  int channels) noexcept {
  return channels == 1 && image_shape(rows, cols, channels).valid;
}

/// Checks an image shape with one supported public unsigned-byte layout.
inline bool valid_u8_image(std::int64_t rows, std::int64_t cols,
                           int channels) noexcept {
  return (channels == 1 || channels == 3 || channels == 4) &&
         image_shape(rows, cols, channels).valid;
}

/// Checks one public image and odd median-filter kernel against policy.
inline bool valid_median_blur(std::int64_t rows, std::int64_t cols,
                              int channels,
                              std::int64_t kernel_size) noexcept {
  return valid_u8_image(rows, cols, channels) && kernel_size >= 3 &&
         valid_kernel_dimension(kernel_size);
}

/// Checks one public image and centered normalized box-filter kernel.
inline bool valid_box_blur(std::int64_t rows, std::int64_t cols, int channels,
                           std::int64_t kernel_width,
                           std::int64_t kernel_height) noexcept {
  return valid_u8_image(rows, cols, channels) &&
         valid_kernel_dimension(kernel_width) &&
         valid_kernel_dimension(kernel_height);
}

/// Checks one public image and bounded odd stack-blur kernel.
inline bool valid_stack_blur(std::int64_t rows, std::int64_t cols,
                             int channels, std::int64_t kernel_width,
                             std::int64_t kernel_height) noexcept {
  return valid_box_blur(rows, cols, channels, kernel_width, kernel_height);
}

/// Checks one bounded BGR pyramid mean-shift filter configuration.
inline bool valid_mean_shift_filter(std::int64_t rows, std::int64_t cols,
                                    int channels, double spatial_radius,
                                    double color_radius,
                                    std::int64_t max_level) noexcept {
  return channels == 3 && image_shape(rows, cols, channels).valid &&
         std::isfinite(spatial_radius) && spatial_radius > 0.0 &&
         spatial_radius <= kMaxMeanShiftRadius &&
         std::isfinite(color_radius) && color_radius > 0.0 &&
         color_radius <= kMaxMeanShiftRadius && max_level >= 0 &&
         max_level <= kMaxMeanShiftPyramidLevel;
}

/// Checks one public image and a complete one-channel unsigned-byte table.
inline bool valid_lookup_table(std::int64_t rows, std::int64_t cols,
                               int channels,
                               std::size_t table_size) noexcept {
  return valid_u8_image(rows, cols, channels) && table_size == 256U;
}

/// Checks one public image and a bounded nonnegative sharpening amount.
inline bool valid_sharpen(std::int64_t rows, std::int64_t cols, int channels,
                          double amount) noexcept {
  return valid_u8_image(rows, cols, channels) && std::isfinite(amount) &&
         amount >= 0.0 && amount <= kMaxSharpenAmount;
}

/// Checks one explicit bilateral-filter configuration against policy.
inline bool valid_bilateral_filter(std::int64_t rows, std::int64_t cols,
                                    int channels, std::int64_t diameter,
                                    double sigma_color,
                                    double sigma_space) noexcept {
  const bool valid_channels = channels == 1 || channels == 3;
  return valid_channels && image_shape(rows, cols, channels).valid &&
         valid_kernel_dimension(diameter) && std::isfinite(sigma_color) &&
         sigma_color > 0.0 && sigma_color <= kMaxFilterSigma &&
         std::isfinite(sigma_space) && sigma_space > 0.0 &&
         sigma_space <= kMaxFilterSigma;
}

/// Checks one explicit adaptive-threshold configuration against policy.
inline bool valid_adaptive_threshold(
    std::int64_t rows, std::int64_t cols, int channels,
    double maximum_value, std::int64_t adaptive_method,
    std::int64_t threshold_mode, std::int64_t block_size,
    double constant) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         valid_threshold_value(maximum_value) &&
         valid_adaptive_threshold_method(adaptive_method) &&
         valid_binary_threshold_mode(threshold_mode) && block_size >= 3 &&
         valid_kernel_dimension(block_size) && std::isfinite(constant) &&
         constant >= -kMaxAdaptiveThresholdConstant &&
         constant <= kMaxAdaptiveThresholdConstant;
}

/// Checks one automatic global threshold configuration against package policy.
inline bool valid_automatic_threshold(std::int64_t rows, std::int64_t cols,
                                      int channels, double maximum_value,
                                      std::int64_t threshold_mode) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         valid_threshold_value(maximum_value) &&
         valid_binary_threshold_mode(threshold_mode);
}

/// Checks one grayscale image and derivative aperture against policy.
inline bool valid_laplacian_edges(std::int64_t rows, std::int64_t cols,
                                  int channels,
                                  std::int64_t kernel_size) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         valid_laplacian_kernel(kernel_size);
}

/// Checks that exactly one first-order image derivative axis is selected.
constexpr bool valid_first_order_derivative(std::int64_t x_order,
                                            std::int64_t y_order) noexcept {
  return (x_order == 1 && y_order == 0) ||
         (x_order == 0 && y_order == 1);
}

/// Checks one grayscale first-order Sobel derivative against package policy.
inline bool valid_sobel_edges(std::int64_t rows, std::int64_t cols,
                              int channels, std::int64_t x_order,
                              std::int64_t y_order,
                              std::int64_t kernel_size) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         valid_first_order_derivative(x_order, y_order) &&
         valid_laplacian_kernel(kernel_size);
}

/// Checks one grayscale first-order Scharr derivative against package policy.
inline bool valid_scharr_edges(std::int64_t rows, std::int64_t cols,
                               int channels, std::int64_t x_order,
                               std::int64_t y_order) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         valid_first_order_derivative(x_order, y_order);
}

/// Checks a grayscale image whose 256-bin counters remain exactly representable.
inline bool valid_histogram_image(std::int64_t rows, std::int64_t cols,
                                  int channels) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         rows <= kMaxExactHistogramPixels / cols;
}

/// Checks one finite connected-component neighborhood policy.
constexpr bool valid_connectivity(std::int64_t connectivity) noexcept {
  return connectivity == 4 || connectivity == 8;
}

/// Checks exact-value flood-fill bounds and semantic policies.
inline bool valid_flood_fill(std::int64_t rows, std::int64_t cols,
                             int channels, std::int64_t seed_x,
                             std::int64_t seed_y, std::int64_t fill_value,
                             std::int64_t connectivity) noexcept {
  return valid_u8_image(rows, cols, channels) &&
         (channels == 1 || channels == 3) && seed_x >= 0 &&
         seed_x < cols && seed_y >= 0 && seed_y < rows && fill_value >= 0 &&
         fill_value <= 255 && valid_connectivity(connectivity);
}

/// Checks a grayscale image whose copied analysis remains policy-bounded.
inline bool valid_analysis_image(std::int64_t rows, std::int64_t cols,
                                 int channels) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         rows <= kMaxCopiedAnalysisPixels / cols;
}

/// Checks a grayscale image whose integral values remain bounded and exact.
inline bool valid_integral_image(std::int64_t rows, std::int64_t cols,
                                 int channels) noexcept {
  return valid_analysis_image(rows, cols, channels);
}

/// Checks compatible template-match inputs and bounds copied output and work.
inline bool valid_template_match(std::int64_t image_rows,
                                 std::int64_t image_cols,
                                 int image_channels,
                                 std::int64_t pattern_rows,
                                 std::int64_t pattern_cols,
                                 int pattern_channels) noexcept {
  if (!valid_u8_image(image_rows, image_cols, image_channels) ||
      !valid_u8_image(pattern_rows, pattern_cols, pattern_channels) ||
      image_channels != pattern_channels || pattern_rows > image_rows ||
      pattern_cols > image_cols) {
    return false;
  }
  const std::int64_t result_rows = image_rows - pattern_rows + 1;
  const std::int64_t result_cols = image_cols - pattern_cols + 1;
  if (result_rows > kMaxCopiedAnalysisPixels / result_cols) {
    return false;
  }
  const std::int64_t pattern_pixels = pattern_rows * pattern_cols;
  if (pattern_pixels > kMaxTemplateMatchWork / image_channels) {
    return false;
  }
  const std::int64_t work_per_result = pattern_pixels * image_channels;
  const std::int64_t result_pixels = result_rows * result_cols;
  return result_pixels <= kMaxTemplateMatchWork / work_per_result;
}

/// Checks that two images have equal valid dimensions and channel counts.
inline bool valid_image_pair(std::int64_t first_rows,
                             std::int64_t first_cols, int first_channels,
                             std::int64_t second_rows,
                             std::int64_t second_cols,
                             int second_channels) noexcept {
  return first_rows == second_rows && first_cols == second_cols &&
         first_channels == second_channels &&
         image_shape(first_rows, first_cols, first_channels).valid &&
         image_shape(second_rows, second_cols, second_channels).valid;
}

/// Computes the checked output shape for horizontal image concatenation.
inline ImageShape horizontal_concat_shape(
    std::int64_t first_rows, std::int64_t first_cols, int first_channels,
    std::int64_t second_rows, std::int64_t second_cols,
    int second_channels) noexcept {
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (!valid_u8_image(first_rows, first_cols, first_channels) ||
      !valid_u8_image(second_rows, second_cols, second_channels) ||
      first_rows != second_rows || first_channels != second_channels ||
      first_cols > max_dimension - second_cols) {
    return {false, 0, 0};
  }
  return image_shape(first_rows, first_cols + second_cols, first_channels);
}

/// Computes the checked output shape for vertical image concatenation.
inline ImageShape vertical_concat_shape(
    std::int64_t first_rows, std::int64_t first_cols, int first_channels,
    std::int64_t second_rows, std::int64_t second_cols,
    int second_channels) noexcept {
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (!valid_u8_image(first_rows, first_cols, first_channels) ||
      !valid_u8_image(second_rows, second_cols, second_channels) ||
      first_cols != second_cols || first_channels != second_channels ||
      first_rows > max_dimension - second_rows) {
    return {false, 0, 0};
  }
  return image_shape(first_rows + second_rows, first_cols, first_channels);
}

/// Computes the checked output shape for a transposed image.
inline ImageShape transposed_image_shape(std::int64_t rows,
                                         std::int64_t cols,
                                         int channels) noexcept {
  if (!valid_u8_image(rows, cols, channels)) {
    return {false, 0, 0};
  }
  return image_shape(cols, rows, channels);
}

/// Computes the checked output shape for positive two-dimensional tiling.
inline ImageShape tiled_image_shape(std::int64_t rows, std::int64_t cols,
                                    int channels,
                                    std::int64_t vertical_count,
                                    std::int64_t horizontal_count) noexcept {
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (!valid_u8_image(rows, cols, channels) || vertical_count <= 0 ||
      horizontal_count <= 0 || vertical_count > max_dimension / rows ||
      horizontal_count > max_dimension / cols) {
    return {false, 0, 0};
  }
  return image_shape(rows * vertical_count, cols * horizontal_count, channels);
}

/// Checks a zero-based channel index for one valid unsigned-byte image.
inline bool valid_channel_index(std::int64_t rows, std::int64_t cols,
                                int channels,
                                std::int64_t channel_index) noexcept {
  return valid_u8_image(rows, cols, channels) && channel_index >= 0 &&
         channel_index < channels;
}

/// Computes a checked BGR or BGRA output shape for grayscale channel merging.
inline ImageShape merged_image_shape(std::int64_t rows, std::int64_t cols,
                                     int output_channels) noexcept {
  if (!valid_grayscale_image(rows, cols, 1) ||
      (output_channels != 3 && output_channels != 4)) {
    return {false, 0, 0};
  }
  return image_shape(rows, cols, output_channels);
}

/// Checks an immutable image-channel replacement contract.
inline bool valid_channel_replacement(
    std::int64_t image_rows, std::int64_t image_cols, int image_channels,
    std::int64_t replacement_rows, std::int64_t replacement_cols,
    int replacement_channels, std::int64_t channel_index) noexcept {
  return valid_channel_index(image_rows, image_cols, image_channels,
                             channel_index) &&
         valid_grayscale_image(replacement_rows, replacement_cols,
                               replacement_channels) &&
         image_rows == replacement_rows && image_cols == replacement_cols;
}

/// Checks an image and same-size single-channel mask without requiring equal channels.
inline bool valid_mask_pair(std::int64_t image_rows, std::int64_t image_cols,
                            int image_channels, std::int64_t mask_rows,
                            std::int64_t mask_cols,
                            int mask_channels) noexcept {
  return valid_u8_image(image_rows, image_cols, image_channels) &&
         mask_channels == 1 &&
         image_rows == mask_rows && image_cols == mask_cols &&
         image_shape(mask_rows, mask_cols, mask_channels).valid;
}

/// Checks two compatible images and one bounded same-size byte weight mask.
inline bool valid_blend_mask(
    std::int64_t first_rows, std::int64_t first_cols, int first_channels,
    std::int64_t second_rows, std::int64_t second_cols, int second_channels,
    std::int64_t mask_rows, std::int64_t mask_cols,
    int mask_channels) noexcept {
  return valid_image_pair(first_rows, first_cols, first_channels, second_rows,
                          second_cols, second_channels) &&
         valid_mask_pair(first_rows, first_cols, first_channels, mask_rows,
                         mask_cols, mask_channels) &&
         valid_analysis_image(mask_rows, mask_cols, mask_channels);
}

/// Checks an inclusive unsigned-byte range over a valid grayscale image.
inline bool valid_u8_range(std::int64_t rows, std::int64_t cols, int channels,
                           std::int64_t lower,
                           std::int64_t upper) noexcept {
  return valid_grayscale_image(rows, cols, channels) && lower >= 0 &&
         upper <= 255 && lower <= upper;
}

/// Checks bounded finite coefficients for weighted image arithmetic.
inline bool valid_weighted_addition(double first_weight,
                                    double second_weight,
                                    double offset) noexcept {
  return std::isfinite(first_weight) &&
         std::abs(first_weight) <= kMaxImageWeight &&
         std::isfinite(second_weight) &&
         std::abs(second_weight) <= kMaxImageWeight &&
         std::isfinite(offset) && std::abs(offset) <= kMaxImageOffset;
}

/// Checks image shape and bounded parameters for Shi-Tomasi corner detection.
inline bool valid_corner_detection(std::int64_t rows, std::int64_t cols,
                                   int channels,
                                   std::int64_t max_corners,
                                   double quality_level,
                                   double minimum_distance) noexcept {
  return valid_analysis_image(rows, cols, channels) && max_corners > 0 &&
         max_corners <= kMaxDetectedCorners &&
         std::isfinite(quality_level) && quality_level > 0.0 &&
         quality_level <= 1.0 && std::isfinite(minimum_distance) &&
         minimum_distance >= 0.0 &&
         minimum_distance <= kMaxFeatureDistance;
}

/// Checks image shape and bounded parameters for threshold keypoint detection.
inline bool valid_threshold_keypoint_detection(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t threshold, std::int64_t max_keypoints) noexcept {
  return valid_analysis_image(rows, cols, channels) && threshold > 0 &&
         threshold <= 255 && max_keypoints > 0 &&
         max_keypoints <= kMaxDetectedCorners;
}

/// Checks one bounded grayscale corner-derivative image request.
inline bool valid_corner_derivative_image(std::int64_t rows,
                                          std::int64_t cols, int channels,
                                          std::int64_t aperture_size) noexcept {
  return valid_analysis_image(rows, cols, channels) &&
         valid_canny_aperture(aperture_size);
}

/// Checks one bounded grayscale derivative corner-response request.
inline bool valid_corner_response(std::int64_t rows, std::int64_t cols,
                                  int channels, std::int64_t block_size,
                                  std::int64_t aperture_size) noexcept {
  const auto smaller_dimension = rows < cols ? rows : cols;
  return valid_corner_derivative_image(rows, cols, channels, aperture_size) &&
         block_size >= 3 && block_size <= kMaxCornerBlockSize &&
         block_size <= smaller_dimension && block_size % 2 == 1;
}

/// Checks one bounded grayscale Harris corner-response request.
inline bool valid_harris_response(std::int64_t rows, std::int64_t cols,
                                  int channels, std::int64_t block_size,
                                  std::int64_t aperture_size,
                                  double response_k) noexcept {
  return valid_corner_response(rows, cols, channels, block_size,
                               aperture_size) &&
         std::isfinite(response_k) && response_k > 0.0 &&
         response_k <= kMaxHarrisResponseK;
}

/// Checks bounded standard Hough line-detection parameters.
inline bool valid_line_detection(std::int64_t rows, std::int64_t cols,
                                 int channels,
                                 std::int64_t vote_threshold) noexcept {
  if (!valid_analysis_image(rows, cols, channels)) {
    return false;
  }
  const auto maximum_votes = rows > cols ? rows : cols;
  return vote_threshold > 0 && vote_threshold <= maximum_votes;
}

/// Checks bounded probabilistic Hough line-segment detection parameters.
inline bool valid_line_segment_detection(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t vote_threshold, std::int64_t minimum_length,
    std::int64_t maximum_gap) noexcept {
  if (!valid_line_detection(rows, cols, channels, vote_threshold)) {
    return false;
  }
  const double maximum_span =
      std::hypot(static_cast<double>(cols - 1),
                 static_cast<double>(rows - 1));
  return minimum_length >= 0 &&
         static_cast<double>(minimum_length) <= maximum_span &&
         maximum_gap >= 0 &&
         static_cast<double>(maximum_gap) <= maximum_span;
}

/// Checks bounded probabilistic circle-detection parameters.
inline bool valid_circle_detection(
    std::int64_t rows, std::int64_t cols, int channels,
    double minimum_distance, double edge_threshold,
    double center_threshold, std::int64_t minimum_radius,
    std::int64_t maximum_radius) noexcept {
  if (!valid_analysis_image(rows, cols, channels)) {
    return false;
  }
  const double maximum_span =
      std::hypot(static_cast<double>(cols - 1),
                 static_cast<double>(rows - 1));
  return std::isfinite(minimum_distance) && minimum_distance >= 1.0 &&
         minimum_distance <= maximum_span &&
         std::isfinite(edge_threshold) && edge_threshold > 0.0 &&
         edge_threshold <= kMaxCannyThreshold &&
         std::isfinite(center_threshold) && center_threshold > 0.0 &&
         center_threshold <= kMaxCannyThreshold && minimum_radius >= 1 &&
         maximum_radius >= minimum_radius &&
         static_cast<double>(maximum_radius) <= maximum_span;
}

/// Checks a finite pixel tolerance bounded by one analysis-image diagonal.
inline bool valid_hull_approximation(std::int64_t rows, std::int64_t cols,
                                     int channels, double epsilon) noexcept {
  return valid_analysis_image(rows, cols, channels) &&
         std::isfinite(epsilon) && epsilon >= 0.0 &&
         epsilon <= std::hypot(static_cast<double>(cols - 1),
                               static_cast<double>(rows - 1));
}

/// Checks one finite coordinate against bounded image pixel centers.
inline bool valid_analysis_point(std::int64_t rows, std::int64_t cols,
                                 double x, double y) noexcept {
  return rows > 0 && cols > 0 && std::isfinite(x) && std::isfinite(y) &&
         x >= 0.0 && y >= 0.0 && x <= static_cast<double>(cols - 1) &&
         y <= static_cast<double>(rows - 1);
}

/// Checks two independently sized grayscale masks for bounded shape matching.
inline bool valid_shape_match(std::int64_t first_rows,
                              std::int64_t first_cols, int first_channels,
                              std::int64_t second_rows,
                              std::int64_t second_cols,
                              int second_channels) noexcept {
  return valid_analysis_image(first_rows, first_cols, first_channels) &&
         valid_analysis_image(second_rows, second_cols, second_channels);
}

/// Checks image shape and parameters for deterministic Canny edge detection.
inline bool valid_canny(std::int64_t rows, std::int64_t cols, int channels,
                        double low_threshold, double high_threshold,
                        std::int64_t aperture_size) noexcept {
  return valid_grayscale_image(rows, cols, channels) &&
         valid_canny_threshold(low_threshold) &&
         valid_canny_threshold(high_threshold) &&
         low_threshold <= high_threshold &&
         valid_canny_aperture(aperture_size);
}

/// Checks six finite, bounded coefficients for one 2D affine transform.
inline bool valid_affine_transform(double m00, double m01, double m02,
                                   double m10, double m11,
                                   double m12) noexcept {
  return std::isfinite(m00) && std::abs(m00) <= kMaxTransformCoefficient &&
         std::isfinite(m01) && std::abs(m01) <= kMaxTransformCoefficient &&
         std::isfinite(m02) && std::abs(m02) <= kMaxTransformCoefficient &&
         std::isfinite(m10) && std::abs(m10) <= kMaxTransformCoefficient &&
         std::isfinite(m11) && std::abs(m11) <= kMaxTransformCoefficient &&
         std::isfinite(m12) && std::abs(m12) <= kMaxTransformCoefficient;
}

/// Checks nine finite, bounded coefficients for one 2D perspective transform.
inline bool valid_perspective_transform(
    double m00, double m01, double m02, double m10, double m11, double m12,
    double m20, double m21, double m22) noexcept {
  return valid_affine_transform(m00, m01, m02, m10, m11, m12) &&
         std::isfinite(m20) && std::abs(m20) <= kMaxTransformCoefficient &&
         std::isfinite(m21) && std::abs(m21) <= kMaxTransformCoefficient &&
         std::isfinite(m22) && std::abs(m22) <= kMaxTransformCoefficient;
}

/// Checks flattened finite coordinates against dimension and copied-value limits.
inline bool valid_flattened_coordinates(const double* points,
                                         std::size_t coordinate_count,
                                         std::size_t dimensions) noexcept {
  if ((points == nullptr && coordinate_count != 0) ||
      dimensions == 0 || coordinate_count % dimensions != 0 ||
      coordinate_count / dimensions > kMaxCopiedPointCount) {
    return false;
  }
  for (std::size_t index = 0; index < coordinate_count; ++index) {
    if (!std::isfinite(points[index]) ||
        std::abs(points[index]) > kMaxTransformCoefficient) {
      return false;
    }
  }
  return true;
}

/// Checks one flattened finite x/y point sequence against copied-value limits.
inline bool valid_flattened_points(const double* points,
                                   std::size_t coordinate_count) noexcept {
  return valid_flattened_coordinates(points, coordinate_count, 2);
}

/// Checks one flattened finite x/y/z point sequence against copied-value limits.
inline bool valid_flattened_object_points(
    const double* points, std::size_t coordinate_count) noexcept {
  return valid_flattened_coordinates(points, coordinate_count, 3);
}

/// Checks one copied three-dimensional vector against numeric limits.
inline bool valid_vector3(double x, double y, double z) noexcept {
  const double values[] = {x, y, z};
  return valid_flattened_object_points(values, 3);
}

/// Checks one Gaussian sigma without allowing NaN or infinity.
inline bool valid_sigma(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

/// Validates a public color-conversion discriminant and its input channels.
inline ColorConversionShape color_conversion_shape(
    std::int64_t rows, std::int64_t cols, int input_channels,
    std::int64_t conversion) noexcept {
  int required_input_channels = 0;
  int output_channels = 0;
  switch (conversion) {
    case 0:
      required_input_channels = 3;
      output_channels = 1;
      break;
    case 1:
      required_input_channels = 1;
      output_channels = 3;
      break;
    case 2:
      required_input_channels = 3;
      output_channels = 4;
      break;
    case 3:
      required_input_channels = 4;
      output_channels = 3;
      break;
    default:
      return {false, 0, 0};
  }
  if (input_channels != required_input_channels ||
      !image_shape(rows, cols, input_channels).valid) {
    return {false, 0, 0};
  }
  const auto output = image_shape(rows, cols, output_channels);
  return output.valid
             ? ColorConversionShape{true, output_channels, output.byte_count}
             : ColorConversionShape{false, 0, 0};
}

}  // namespace validation
}  // namespace terlan_opencv

#endif
