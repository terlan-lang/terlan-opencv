#include "opencv_contrast_validation.hpp"
#include "opencv_drawing_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_validation.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

using terlan_opencv::validation::color_conversion_shape;
using terlan_opencv::validation::image_shape;
using terlan_opencv::validation::image_shape_for_layout;
using terlan_opencv::validation::kMaxOwnedImageBytes;
using terlan_opencv::validation::kMaxEncodedImageBytes;
using terlan_opencv::validation::kMaxCannyThreshold;
using terlan_opencv::validation::kMaxAdaptiveEqualizationClipLimit;
using terlan_opencv::validation::kMaxAdaptiveEqualizationTiles;
using terlan_opencv::validation::kMaxDetectedCorners;
using terlan_opencv::validation::kMaxDetectedCircles;
using terlan_opencv::validation::kMaxDetectedLines;
using terlan_opencv::validation::kMaxDetectedLineSegments;
using terlan_opencv::validation::kMaxFeatureDistance;
using terlan_opencv::validation::kMaxCornerBlockSize;
using terlan_opencv::validation::kMaxHarrisResponseK;
using terlan_opencv::validation::kMaxFilterSigma;
using terlan_opencv::validation::kMaxImageOffset;
using terlan_opencv::validation::kMaxImageWeight;
using terlan_opencv::validation::kMaxKernelDimension;
using terlan_opencv::validation::kMaxTransformCoefficient;
using terlan_opencv::validation::kMaxCopiedPointCount;
using terlan_opencv::validation::kMaxMorphologyIterations;
using terlan_opencv::validation::layout_channels;
using terlan_opencv::validation::matrix_shape;
using terlan_opencv::validation::padded_image_shape;
using terlan_opencv::validation::rotated_image_shape;
using terlan_opencv::validation::valid_border_mode;
using terlan_opencv::validation::valid_bilateral_filter;
using terlan_opencv::validation::valid_box_blur;
using terlan_opencv::validation::valid_adaptive_threshold;
using terlan_opencv::validation::valid_adaptive_threshold_method;
using terlan_opencv::validation::valid_binary_threshold_mode;
using terlan_opencv::validation::valid_laplacian_edges;
using terlan_opencv::validation::valid_laplacian_kernel;
using terlan_opencv::validation::valid_compound_morphology;
using terlan_opencv::validation::valid_morphology_operation;
using terlan_opencv::validation::pyramid_down_shape;
using terlan_opencv::validation::pyramid_up_shape;
using terlan_opencv::validation::valid_buffer_size;
using terlan_opencv::validation::valid_canny;
using terlan_opencv::validation::valid_canny_aperture;
using terlan_opencv::validation::valid_canny_threshold;
using terlan_opencv::validation::valid_channel_index;
using terlan_opencv::validation::valid_channel_replacement;
using terlan_opencv::validation::valid_comparison_mode;
using terlan_opencv::validation::valid_divide_scale;
using terlan_opencv::validation::valid_multiply_scale;
using terlan_opencv::validation::valid_analysis_image;
using terlan_opencv::validation::valid_analysis_point;
using terlan_opencv::validation::valid_integral_image;
using terlan_opencv::validation::valid_template_match;
using terlan_opencv::validation::valid_corner_detection;
using terlan_opencv::validation::valid_threshold_keypoint_detection;
using terlan_opencv::validation::valid_circle_detection;
using terlan_opencv::validation::valid_corner_derivative_image;
using terlan_opencv::validation::valid_corner_response;
using terlan_opencv::validation::valid_harris_response;
using terlan_opencv::validation::valid_line_detection;
using terlan_opencv::validation::valid_line_segment_detection;
using terlan_opencv::validation::valid_line_drawing;
using terlan_opencv::validation::valid_rectangle_drawing;
using terlan_opencv::validation::valid_rectangle_fill;
using terlan_opencv::validation::valid_circle_drawing;
using terlan_opencv::validation::valid_circle_fill;
using terlan_opencv::validation::valid_cross_drawing;
using terlan_opencv::validation::valid_ellipse_drawing;
using terlan_opencv::validation::valid_ellipse_fill;
using terlan_opencv::validation::valid_connectivity;
using terlan_opencv::validation::valid_encoded_image_input;
using terlan_opencv::validation::valid_decode_mode;
using terlan_opencv::validation::valid_decoded_storage;
using terlan_opencv::validation::valid_encoded_output_size;
using terlan_opencv::validation::valid_image_format;
using terlan_opencv::validation::valid_kernel_dimension;
using terlan_opencv::validation::valid_morphology;
using terlan_opencv::validation::valid_morphology_iterations;
using terlan_opencv::validation::valid_automatic_threshold;
using terlan_opencv::validation::valid_affine_transform;
using terlan_opencv::validation::valid_perspective_transform;
using terlan_opencv::validation::valid_flattened_points;
using terlan_opencv::validation::valid_flattened_object_points;
using terlan_opencv::validation::valid_vector3;
using terlan_opencv::validation::valid_flip_mode;
using terlan_opencv::validation::valid_flood_fill;
using terlan_opencv::validation::valid_grayscale_image;
using terlan_opencv::validation::valid_histogram_image;
using terlan_opencv::validation::valid_adaptive_equalization;
using terlan_opencv::validation::horizontal_concat_shape;
using terlan_opencv::validation::merged_image_shape;
using terlan_opencv::validation::valid_image_pair;
using terlan_opencv::validation::valid_blend_mask;
using terlan_opencv::validation::valid_mask_pair;
using terlan_opencv::validation::valid_median_blur;
using terlan_opencv::validation::valid_sobel_edges;
using terlan_opencv::validation::valid_scharr_edges;
using terlan_opencv::validation::valid_stack_blur;
using terlan_opencv::validation::valid_mean_shift_filter;
using terlan_opencv::validation::valid_lookup_table;
using terlan_opencv::validation::valid_sharpen;
using terlan_opencv::validation::valid_weighted_addition;
using terlan_opencv::validation::valid_rotate_mode;
using terlan_opencv::validation::valid_threshold_mode;
using terlan_opencv::validation::valid_threshold_value;
using terlan_opencv::validation::valid_u8_image;
using terlan_opencv::validation::valid_u8_range;
using terlan_opencv::validation::vertical_concat_shape;
using terlan_opencv::validation::transposed_image_shape;
using terlan_opencv::validation::tiled_image_shape;
using terlan_opencv::validation::valid_crop_region;
using terlan_opencv::validation::valid_sigma;
using terlan_opencv::failure::NativeFailure;

static_assert(layout_channels(1) == 1, "gray layout");
static_assert(layout_channels(3) == 3, "BGR layout");
static_assert(layout_channels(4) == 4, "BGRA layout");
static_assert(layout_channels(0) == 0, "invalid layout");
static_assert(matrix_shape(2, 3, 4, 8).valid, "typed matrix shape");
static_assert(matrix_shape(2, 3, 4, 8).byte_count == 192,
              "typed matrix byte count");
static_assert(!matrix_shape(0, 3, 4, 8).valid, "zero matrix rows");
static_assert(!matrix_shape(2, 0, 4, 8).valid, "zero matrix columns");
static_assert(!matrix_shape(2, 3, 0, 8).valid, "zero matrix channels");
static_assert(!matrix_shape(2, 3, 513, 8).valid,
              "excessive matrix channels");
static_assert(!matrix_shape(2, 3, 4, 0).valid, "zero scalar width");
static_assert(!matrix_shape(std::numeric_limits<std::int64_t>::max(), 3, 4, 8)
                   .valid,
              "overflowing matrix rows");
static_assert(valid_kernel_dimension(1), "one-pixel kernel");
static_assert(valid_kernel_dimension(255), "maximum kernel");
static_assert(!valid_kernel_dimension(0), "zero kernel");
static_assert(!valid_kernel_dimension(2), "even kernel");
static_assert(!valid_kernel_dimension(257), "oversized kernel");
static_assert(valid_morphology_iterations(1), "one morphology iteration");
static_assert(valid_morphology_iterations(kMaxMorphologyIterations),
              "maximum morphology iterations");
static_assert(!valid_morphology_iterations(0), "zero morphology iterations");
static_assert(!valid_morphology_iterations(kMaxMorphologyIterations + 1),
              "oversized morphology iterations");
static_assert(valid_canny_aperture(3), "minimum Canny aperture");
static_assert(valid_canny_aperture(5), "middle Canny aperture");
static_assert(valid_canny_aperture(7), "maximum Canny aperture");
static_assert(!valid_canny_aperture(1), "undersized Canny aperture");
static_assert(!valid_canny_aperture(4), "even Canny aperture");
static_assert(!valid_canny_aperture(9), "oversized Canny aperture");
static_assert(valid_decode_mode(-1), "unchanged decode mode");
static_assert(valid_decode_mode(0), "grayscale decode mode");
static_assert(valid_decode_mode(1), "color decode mode");
static_assert(!valid_decode_mode(2), "invalid decode mode");
static_assert(valid_image_format(0), "BMP image format");
static_assert(valid_image_format(1), "PNG image format");
static_assert(valid_image_format(2), "JPEG image format");
static_assert(valid_image_format(3), "WebP image format");
static_assert(valid_image_format(4), "TIFF image format");
static_assert(!valid_image_format(-1), "negative image format");
static_assert(!valid_image_format(5), "invalid image format");
static_assert(valid_encoded_output_size(1), "small encoded output");
static_assert(valid_encoded_output_size(kMaxEncodedImageBytes),
              "maximum encoded output");
static_assert(!valid_encoded_output_size(0), "empty encoded output");
static_assert(!valid_encoded_output_size(kMaxEncodedImageBytes + 1),
              "oversized encoded output");
static_assert(valid_decoded_storage(kMaxOwnedImageBytes, 1),
              "maximum decoded storage");
static_assert(!valid_decoded_storage(kMaxOwnedImageBytes, 2),
              "oversized decoded storage");
static_assert(!valid_decoded_storage(0, 1), "empty decoded storage");
static_assert(valid_crop_region(4, 5, 1, 1, 4, 3), "bounded crop");
static_assert(!valid_crop_region(4, 5, -1, 0, 1, 1), "negative crop x");
static_assert(!valid_crop_region(4, 5, 0, -1, 1, 1), "negative crop y");
static_assert(!valid_crop_region(4, 5, 0, 0, 0, 1), "empty crop width");
static_assert(!valid_crop_region(4, 5, 4, 0, 2, 1), "crop beyond right edge");
static_assert(!valid_crop_region(4, 5, 0, 3, 1, 2), "crop beyond bottom edge");
static_assert(!valid_crop_region(4, 5, 1, 1,
                                 std::numeric_limits<std::int64_t>::max(), 1),
              "overflowing crop width");
static_assert(valid_border_mode(0), "constant border");
static_assert(valid_border_mode(1), "replicate border");
static_assert(valid_border_mode(2), "reflect border");
static_assert(!valid_border_mode(3), "invalid border");
static_assert(valid_flip_mode(-1), "both-axis flip");
static_assert(valid_flip_mode(0), "vertical flip");
static_assert(valid_flip_mode(1), "horizontal flip");
static_assert(!valid_flip_mode(2), "invalid flip");
static_assert(valid_rotate_mode(0), "clockwise rotation");
static_assert(valid_rotate_mode(1), "half rotation");
static_assert(valid_rotate_mode(2), "counter-clockwise rotation");
static_assert(!valid_rotate_mode(3), "invalid rotation");
static_assert(valid_threshold_mode(0), "binary threshold");
static_assert(valid_threshold_mode(1), "inverse binary threshold");
static_assert(valid_threshold_mode(2), "truncate threshold");
static_assert(valid_threshold_mode(3), "to-zero threshold");
static_assert(valid_threshold_mode(4), "inverse to-zero threshold");
static_assert(!valid_threshold_mode(-1), "negative threshold mode");
static_assert(!valid_threshold_mode(5), "invalid threshold mode");
static_assert(static_cast<std::int64_t>(NativeFailure::InvalidEncodedImage) ==
                  6,
              "invalid encoded image status");
static_assert(static_cast<std::int64_t>(NativeFailure::CodecUnavailable) == 7,
              "codec unavailable status");
static_assert(static_cast<std::int64_t>(NativeFailure::EncodedImageTooLarge) ==
                  8,
              "encoded image limit status");
static_assert(static_cast<std::int64_t>(NativeFailure::DecodedImageTooLarge) ==
                  9,
              "decoded image limit status");

int main() {
  const double points[] = {0.0, 0.0, 4.0, 4.0};
  const double odd_points[] = {0.0, 0.0, 1.0};
  const double non_finite_points[] = {
      std::numeric_limits<double>::quiet_NaN(), 0.0};
  const double excessive_points[] = {kMaxTransformCoefficient + 1.0, 0.0};
  const std::vector<double> oversized_points((kMaxCopiedPointCount + 1) * 2,
                                              0.0);
  assert(valid_flattened_points(nullptr, 0));
  assert(valid_flattened_points(points, 4));
  assert(!valid_flattened_points(nullptr, 2));
  assert(!valid_flattened_points(odd_points, 3));
  assert(!valid_flattened_points(non_finite_points, 2));
  assert(!valid_flattened_points(excessive_points, 2));
  assert(!valid_flattened_points(oversized_points.data(),
                                 oversized_points.size()));
  assert(valid_flattened_object_points(points, 3));
  assert(!valid_flattened_object_points(points, 4));
  assert(valid_vector3(0.0, 1.0, -1.0));
  assert(!valid_vector3(0.0, std::numeric_limits<double>::infinity(), 0.0));
  const std::uint8_t bmp[] = {'B', 'M'};
  const std::uint8_t png[] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
  const std::uint8_t jpeg[] = {0xff, 0xd8, 0xff};
  const std::uint8_t webp[] = {'R', 'I', 'F', 'F', 0, 0, 0, 0,
                               'W', 'E', 'B', 'P'};
  const std::uint8_t little_tiff[] = {'I', 'I', 42, 0};
  const std::uint8_t big_tiff[] = {'M', 'M', 0, 42};
  const std::uint8_t invalid_webp[] = {'R', 'I', 'F', 'F', 0, 0, 0, 0,
                                       'W', 'A', 'V', 'E'};
  assert(valid_encoded_image_input(bmp, sizeof(bmp)));
  assert(valid_encoded_image_input(png, sizeof(png)));
  assert(valid_encoded_image_input(jpeg, sizeof(jpeg)));
  assert(valid_encoded_image_input(webp, sizeof(webp)));
  assert(valid_encoded_image_input(little_tiff, sizeof(little_tiff)));
  assert(valid_encoded_image_input(big_tiff, sizeof(big_tiff)));
  assert(!valid_encoded_image_input(nullptr, sizeof(bmp)));
  assert(!valid_encoded_image_input(bmp, 1));
  assert(!valid_encoded_image_input(invalid_webp, sizeof(invalid_webp)));
  assert(!valid_encoded_image_input(bmp, kMaxEncodedImageBytes + 1));
  assert(valid_threshold_value(0.0));
  assert(valid_threshold_value(127.5));
  assert(valid_threshold_value(255.0));
  assert(!valid_threshold_value(-0.1));
  assert(!valid_threshold_value(255.1));
  assert(!valid_threshold_value(std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_threshold_value(std::numeric_limits<double>::infinity()));
  assert(valid_morphology(2, 2, 1, 3, 3, 1));
  assert(!valid_morphology(0, 2, 1, 3, 3, 1));
  assert(!valid_morphology(2, 2, 1, 2, 3, 1));
  assert(!valid_morphology(2, 2, 1, 3, 2, 1));
  assert(!valid_morphology(2, 2, 1, 3, 3, 0));
  assert(valid_canny_threshold(0.0));
  assert(valid_canny_threshold(kMaxCannyThreshold));
  assert(!valid_canny_threshold(-0.1));
  assert(!valid_canny_threshold(kMaxCannyThreshold + 1.0));
  assert(!valid_canny_threshold(std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_canny_threshold(std::numeric_limits<double>::infinity()));
  assert(valid_canny(3, 3, 1, 10.0, 20.0, 3));
  assert(!valid_canny(3, 3, 3, 10.0, 20.0, 3));
  assert(!valid_canny(3, 3, 1, 20.0, 10.0, 3));
  assert(!valid_canny(3, 3, 1, 10.0, 20.0, 9));
  assert(valid_affine_transform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0));
  assert(valid_affine_transform(kMaxTransformCoefficient,
                                -kMaxTransformCoefficient, 0.0, 0.0, 1.0,
                                0.0));
  assert(!valid_affine_transform(kMaxTransformCoefficient + 1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0));
  assert(!valid_affine_transform(
      1.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0, 0.0));
  assert(!valid_affine_transform(
      1.0, 0.0, 0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0));
  assert(valid_perspective_transform(
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0));
  assert(valid_perspective_transform(
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, kMaxTransformCoefficient,
      -kMaxTransformCoefficient, 1.0));
  assert(!valid_perspective_transform(
      1.0, 0.0, 0.0, 1.0, 0.0, 0.0, kMaxTransformCoefficient + 1.0, 0.0,
      1.0));
  assert(!valid_perspective_transform(
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0));
  assert(!valid_perspective_transform(
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      std::numeric_limits<double>::infinity(), 1.0));
  assert(valid_grayscale_image(2, 2, 1));
  assert(!valid_grayscale_image(2, 2, 3));
  assert(!valid_grayscale_image(0, 2, 1));
  assert(!valid_grayscale_image(2, 0, 1));
  assert(!valid_grayscale_image(
      std::numeric_limits<std::int64_t>::max(), 2, 1));
  assert(valid_histogram_image(4096, 4096, 1));
  assert(!valid_histogram_image(4097, 4096, 1));
  assert(!valid_histogram_image(2, 2, 3));
  assert(valid_adaptive_equalization(8, 8, 1, 2.0, 1, 1));
  assert(valid_adaptive_equalization(
      64, 64, 1, kMaxAdaptiveEqualizationClipLimit,
      kMaxAdaptiveEqualizationTiles, kMaxAdaptiveEqualizationTiles));
  assert(!valid_adaptive_equalization(0, 8, 1, 2.0, 1, 1));
  assert(!valid_adaptive_equalization(8, 8, 3, 2.0, 1, 1));
  assert(!valid_adaptive_equalization(1025, 1024, 1, 2.0, 1, 1));
  assert(!valid_adaptive_equalization(8, 8, 1, 0.0, 1, 1));
  assert(!valid_adaptive_equalization(8, 8, 1, -1.0, 1, 1));
  assert(!valid_adaptive_equalization(
      8, 8, 1, kMaxAdaptiveEqualizationClipLimit + 1.0, 1, 1));
  assert(!valid_adaptive_equalization(
      8, 8, 1, std::numeric_limits<double>::quiet_NaN(), 1, 1));
  assert(!valid_adaptive_equalization(
      8, 8, 1, std::numeric_limits<double>::infinity(), 1, 1));
  assert(!valid_adaptive_equalization(8, 8, 1, 2.0, 0, 1));
  assert(!valid_adaptive_equalization(8, 8, 1, 2.0, 1, 0));
  assert(!valid_adaptive_equalization(8, 8, 1, 2.0, 9, 1));
  assert(!valid_adaptive_equalization(8, 8, 1, 2.0, 1, 9));
  assert(!valid_adaptive_equalization(
      65, 65, 1, 2.0, kMaxAdaptiveEqualizationTiles + 1, 1));
  assert(valid_connectivity(4));
  assert(valid_connectivity(8));
  assert(!valid_connectivity(0));
  assert(!valid_connectivity(6));
  assert(valid_analysis_image(1024, 1024, 1));
  assert(!valid_analysis_image(1025, 1024, 1));
  assert(!valid_analysis_image(2, 2, 3));
  assert(valid_integral_image(1024, 1024, 1));
  assert(!valid_integral_image(1025, 1024, 1));
  assert(!valid_integral_image(2, 2, 3));
  assert(!valid_integral_image(0, 2, 1));
  assert(valid_template_match(3, 3, 1, 2, 2, 1));
  assert(valid_template_match(2, 2, 3, 2, 2, 3));
  assert(!valid_template_match(2, 2, 1, 3, 2, 1));
  assert(!valid_template_match(2, 2, 1, 2, 2, 3));
  assert(!valid_template_match(1025, 1024, 1, 1, 1, 1));
  assert(!valid_template_match(512, 512, 1, 256, 256, 1));
  assert(!valid_template_match(0, 2, 1, 1, 1, 1));
  assert(valid_image_pair(2, 3, 1, 2, 3, 1));
  assert(valid_image_pair(2, 3, 4, 2, 3, 4));
  assert(!valid_image_pair(2, 3, 1, 3, 2, 1));
  assert(!valid_image_pair(2, 3, 1, 2, 3, 3));
  assert(!valid_image_pair(0, 3, 1, 0, 3, 1));
  assert(!valid_image_pair(
      std::numeric_limits<std::int64_t>::max(), 1, 1,
      std::numeric_limits<std::int64_t>::max(), 1, 1));
  const auto horizontal = horizontal_concat_shape(2, 2, 3, 2, 1, 3);
  assert(horizontal.valid);
  assert(horizontal.byte_count == 18);
  assert(!horizontal_concat_shape(2, 2, 3, 1, 1, 3).valid);
  assert(!horizontal_concat_shape(2, 2, 3, 2, 1, 1).valid);
  assert(horizontal_concat_shape(1, 134217728, 1, 1, 134217728, 1).valid);
  assert(!horizontal_concat_shape(1, 134217729, 1, 1, 134217729, 1).valid);
  const auto vertical = vertical_concat_shape(1, 2, 4, 2, 2, 4);
  assert(vertical.valid);
  assert(vertical.byte_count == 24);
  assert(!vertical_concat_shape(1, 2, 4, 2, 1, 4).valid);
  assert(!vertical_concat_shape(1, 2, 4, 2, 2, 3).valid);
  assert(vertical_concat_shape(134217728, 1, 1, 134217728, 1, 1).valid);
  assert(!vertical_concat_shape(134217729, 1, 1, 134217729, 1, 1).valid);
  const auto transposed = transposed_image_shape(2, 3, 4);
  assert(transposed.valid);
  assert(transposed.byte_count == 24);
  assert(transposed_image_shape(1, 268435456, 1).valid);
  assert(!transposed_image_shape(1, 268435457, 1).valid);
  assert(!transposed_image_shape(2, 3, 2).valid);
  assert(!transposed_image_shape(0, 3, 1).valid);
  const auto tiled = tiled_image_shape(2, 3, 3, 2, 2);
  assert(tiled.valid);
  assert(tiled.byte_count == 72);
  assert(tiled_image_shape(1, 1, 1, 16384, 16384).valid);
  assert(!tiled_image_shape(1, 1, 1, 16384, 16385).valid);
  assert(!tiled_image_shape(2, 3, 3, 0, 1).valid);
  assert(!tiled_image_shape(2, 3, 3, 1, 0).valid);
  assert(!tiled_image_shape(2, 3, 3, -1, 1).valid);
  assert(!tiled_image_shape(2, 3, 3, 1, -1).valid);
  assert(!tiled_image_shape(1, 1, 1,
                            std::numeric_limits<std::int64_t>::max(), 1)
              .valid);
  assert(valid_channel_index(2, 3, 1, 0));
  assert(valid_channel_index(2, 3, 3, 2));
  assert(valid_channel_index(2, 3, 4, 3));
  assert(!valid_channel_index(2, 3, 1, 1));
  assert(!valid_channel_index(2, 3, 3, -1));
  assert(!valid_channel_index(2, 3, 3, 3));
  assert(!valid_channel_index(2, 3, 2, 0));
  assert(!valid_channel_index(0, 3, 1, 0));
  assert(merged_image_shape(2, 3, 3).valid);
  assert(merged_image_shape(2, 3, 3).byte_count == 18);
  assert(merged_image_shape(2, 3, 4).valid);
  assert(merged_image_shape(2, 3, 4).byte_count == 24);
  assert(merged_image_shape(8192, 8192, 4).valid);
  assert(!merged_image_shape(8192, 8193, 4).valid);
  assert(!merged_image_shape(2, 3, 1).valid);
  assert(!merged_image_shape(2, 3, 2).valid);
  assert(!merged_image_shape(2, 3, 5).valid);
  assert(!merged_image_shape(0, 3, 3).valid);
  assert(valid_channel_replacement(2, 3, 3, 2, 3, 1, 2));
  assert(valid_channel_replacement(2, 3, 4, 2, 3, 1, 3));
  assert(valid_channel_replacement(2, 3, 1, 2, 3, 1, 0));
  assert(!valid_channel_replacement(2, 3, 3, 2, 3, 1, -1));
  assert(!valid_channel_replacement(2, 3, 3, 2, 3, 1, 3));
  assert(!valid_channel_replacement(2, 3, 3, 3, 2, 1, 0));
  assert(!valid_channel_replacement(2, 3, 3, 2, 3, 3, 0));
  assert(!valid_channel_replacement(0, 3, 3, 0, 3, 1, 0));
  assert(valid_comparison_mode(0));
  assert(valid_comparison_mode(5));
  assert(!valid_comparison_mode(-1));
  assert(!valid_comparison_mode(6));
  assert(!valid_comparison_mode(std::numeric_limits<std::int64_t>::max()));
  assert(valid_multiply_scale(0.0));
  assert(valid_multiply_scale(1.0));
  assert(!valid_multiply_scale(-0.0001));
  assert(!valid_multiply_scale(1.0001));
  assert(!valid_multiply_scale(std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_multiply_scale(std::numeric_limits<double>::infinity()));
  assert(valid_divide_scale(0.0));
  assert(valid_divide_scale(255.0));
  assert(!valid_divide_scale(-0.0001));
  assert(!valid_divide_scale(255.0001));
  assert(!valid_divide_scale(std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_divide_scale(std::numeric_limits<double>::infinity()));
  assert(valid_flood_fill(3, 4, 1, 0, 0, 0, 4));
  assert(valid_flood_fill(3, 4, 3, 3, 2, 255, 8));
  assert(!valid_flood_fill(3, 4, 4, 1, 1, 128, 4));
  assert(!valid_flood_fill(3, 4, 1, -1, 0, 0, 4));
  assert(!valid_flood_fill(3, 4, 1, 4, 0, 0, 4));
  assert(!valid_flood_fill(3, 4, 1, 0, -1, 0, 4));
  assert(!valid_flood_fill(3, 4, 1, 0, 3, 0, 4));
  assert(!valid_flood_fill(3, 4, 1, 0, 0, -1, 4));
  assert(!valid_flood_fill(3, 4, 1, 0, 0, 256, 4));
  assert(!valid_flood_fill(3, 4, 1, 0, 0, 0, 5));
  assert(!valid_flood_fill(3, 4, 2, 0, 0, 0, 4));
  assert(valid_u8_image(2, 3, 1));
  assert(valid_u8_image(2, 3, 3));
  assert(valid_u8_image(2, 3, 4));
  assert(!valid_u8_image(2, 3, 2));
  assert(!valid_u8_image(0, 3, 3));
  assert(!valid_u8_image(2, 0, 3));
  assert(!valid_u8_image(16384, 16384, 3));
  assert(!valid_u8_image(std::numeric_limits<std::int64_t>::max(), 1, 1));
  assert(valid_median_blur(3, 3, 1, 3));
  assert(valid_median_blur(3, 3, 3, 255));
  assert(valid_median_blur(3, 3, 4, 5));
  assert(!valid_median_blur(3, 3, 1, 1));
  assert(!valid_median_blur(3, 3, 1, 2));
  assert(!valid_median_blur(3, 3, 1, 257));
  assert(!valid_median_blur(3, 3, 2, 3));
  assert(!valid_median_blur(0, 3, 1, 3));
  assert(!valid_median_blur(16384, 16384, 3, 3));
  assert(valid_bilateral_filter(3, 3, 1, 1, 1.0, 1.0));
  assert(valid_bilateral_filter(3, 3, 3, 255, kMaxFilterSigma,
                                kMaxFilterSigma));
  assert(!valid_bilateral_filter(3, 3, 4, 3, 1.0, 1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 0, 1.0, 1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 2, 1.0, 1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 257, 1.0, 1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 3, 0.0, 1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 3, 1.0, -1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 3, kMaxFilterSigma + 1.0, 1.0));
  assert(!valid_bilateral_filter(3, 3, 3, 3, 1.0,
                                 kMaxFilterSigma + 1.0));
  assert(!valid_bilateral_filter(
      3, 3, 3, 3, std::numeric_limits<double>::quiet_NaN(), 1.0));
  assert(!valid_bilateral_filter(
      3, 3, 3, 3, 1.0, std::numeric_limits<double>::infinity()));
  assert(!valid_bilateral_filter(0, 3, 3, 3, 1.0, 1.0));
  assert(valid_box_blur(3, 3, 1, 1, 255));
  assert(valid_box_blur(3, 3, 3, 3, 5));
  assert(valid_box_blur(3, 3, 4, 255, 1));
  assert(!valid_box_blur(0, 3, 1, 3, 3));
  assert(!valid_box_blur(3, 3, 2, 3, 3));
  assert(!valid_box_blur(3, 3, 1, 0, 3));
  assert(!valid_box_blur(3, 3, 1, 2, 3));
  assert(!valid_box_blur(3, 3, 1, 3, 257));
  assert(valid_sobel_edges(3, 3, 1, 1, 0, 1));
  assert(valid_sobel_edges(3, 3, 1, 0, 1, 7));
  assert(!valid_sobel_edges(3, 3, 3, 1, 0, 3));
  assert(!valid_sobel_edges(3, 3, 1, 0, 0, 3));
  assert(!valid_sobel_edges(3, 3, 1, 1, 1, 3));
  assert(!valid_sobel_edges(3, 3, 1, 2, 0, 3));
  assert(!valid_sobel_edges(3, 3, 1, 1, 0, 2));
  assert(valid_scharr_edges(3, 3, 1, 1, 0));
  assert(valid_scharr_edges(3, 3, 1, 0, 1));
  assert(!valid_scharr_edges(3, 3, 3, 1, 0));
  assert(!valid_scharr_edges(0, 3, 1, 1, 0));
  assert(!valid_scharr_edges(3, 3, 1, 0, 0));
  assert(!valid_scharr_edges(3, 3, 1, 1, 1));
  assert(!valid_scharr_edges(3, 3, 1, -1, 0));
  assert(valid_stack_blur(3, 3, 1, 1, 1));
  assert(valid_stack_blur(3, 3, 3, 3, 255));
  assert(valid_stack_blur(3, 3, 4, 255, 3));
  assert(!valid_stack_blur(0, 3, 1, 3, 3));
  assert(!valid_stack_blur(3, 3, 2, 3, 3));
  assert(!valid_stack_blur(3, 3, 1, 0, 3));
  assert(!valid_stack_blur(3, 3, 1, 2, 3));
  assert(!valid_stack_blur(3, 3, 1, 3, 257));
  assert(valid_mean_shift_filter(3, 3, 3, 1.0, 1.0, 0));
  assert(valid_mean_shift_filter(3, 3, 3, 65535.0, 65535.0, 8));
  assert(!valid_mean_shift_filter(3, 3, 1, 1.0, 1.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 4, 1.0, 1.0, 0));
  assert(!valid_mean_shift_filter(0, 3, 3, 1.0, 1.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 3, 0.0, 1.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 3, 1.0, -1.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 3, 65536.0, 1.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 3, 1.0, 65536.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 3,
                                  std::numeric_limits<double>::quiet_NaN(),
                                  1.0, 0));
  assert(!valid_mean_shift_filter(3, 3, 3, 1.0,
                                  std::numeric_limits<double>::infinity(),
                                  0));
  assert(!valid_mean_shift_filter(3, 3, 3, 1.0, 1.0, -1));
  assert(!valid_mean_shift_filter(3, 3, 3, 1.0, 1.0, 9));
  assert(valid_lookup_table(3, 3, 1, 256));
  assert(valid_lookup_table(3, 3, 3, 256));
  assert(valid_lookup_table(3, 3, 4, 256));
  assert(!valid_lookup_table(0, 3, 1, 256));
  assert(!valid_lookup_table(3, 3, 2, 256));
  assert(!valid_lookup_table(3, 3, 1, 0));
  assert(!valid_lookup_table(3, 3, 1, 255));
  assert(!valid_lookup_table(3, 3, 1, 257));
  assert(valid_sharpen(3, 3, 1, 0.0));
  assert(valid_sharpen(3, 3, 3, 1.0));
  assert(valid_sharpen(3, 3, 4, 16.0));
  assert(!valid_sharpen(0, 3, 1, 1.0));
  assert(!valid_sharpen(3, 3, 2, 1.0));
  assert(!valid_sharpen(3, 3, 1, -0.1));
  assert(!valid_sharpen(3, 3, 1, 16.1));
  assert(!valid_sharpen(3, 3, 1,
                        std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_sharpen(3, 3, 1,
                        std::numeric_limits<double>::infinity()));
  assert(valid_adaptive_threshold_method(0));
  assert(valid_adaptive_threshold_method(1));
  assert(!valid_adaptive_threshold_method(-1));
  assert(!valid_adaptive_threshold_method(2));
  assert(valid_binary_threshold_mode(0));
  assert(valid_binary_threshold_mode(1));
  assert(!valid_binary_threshold_mode(-1));
  assert(!valid_binary_threshold_mode(2));
  assert(valid_adaptive_threshold(3, 3, 1, 0.0, 0, 0, 3, -255.0));
  assert(valid_adaptive_threshold(3, 3, 1, 255.0, 1, 1, 255, 255.0));
  assert(!valid_adaptive_threshold(3, 3, 3, 200.0, 0, 0, 3, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, -0.1, 0, 0, 3, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 255.1, 0, 0, 3, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 2, 0, 3, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 0, 2, 3, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 0, 0, 1, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 0, 0, 2, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 0, 0, 257, 0.0));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 0, 0, 3, -255.1));
  assert(!valid_adaptive_threshold(3, 3, 1, 200.0, 0, 0, 3, 255.1));
  assert(!valid_adaptive_threshold(
      3, 3, 1, 200.0, 0, 0, 3,
      std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_adaptive_threshold(
      3, 3, 1, std::numeric_limits<double>::infinity(), 0, 0, 3, 0.0));
  assert(valid_automatic_threshold(2, 4, 1, 0.0, 0));
  assert(valid_automatic_threshold(2, 4, 1, 255.0, 1));
  assert(!valid_automatic_threshold(2, 4, 3, 200.0, 0));
  assert(!valid_automatic_threshold(0, 4, 1, 200.0, 0));
  assert(!valid_automatic_threshold(2, 4, 1, -0.1, 0));
  assert(!valid_automatic_threshold(2, 4, 1, 255.1, 0));
  assert(!valid_automatic_threshold(2, 4, 1, 200.0, 2));
  assert(!valid_automatic_threshold(
      2, 4, 1, std::numeric_limits<double>::quiet_NaN(), 0));
  assert(valid_laplacian_kernel(1));
  assert(valid_laplacian_kernel(3));
  assert(valid_laplacian_kernel(5));
  assert(valid_laplacian_kernel(7));
  assert(!valid_laplacian_kernel(-1));
  assert(!valid_laplacian_kernel(0));
  assert(!valid_laplacian_kernel(2));
  assert(!valid_laplacian_kernel(9));
  assert(valid_laplacian_edges(3, 3, 1, 1));
  assert(valid_laplacian_edges(3, 3, 1, 7));
  assert(!valid_laplacian_edges(3, 3, 3, 3));
  assert(!valid_laplacian_edges(0, 3, 1, 3));
  assert(!valid_laplacian_edges(3, 3, 1, 9));
  assert(valid_morphology_operation(0));
  assert(valid_morphology_operation(1));
  assert(valid_morphology_operation(2));
  assert(valid_morphology_operation(3));
  assert(valid_morphology_operation(4));
  assert(!valid_morphology_operation(-1));
  assert(!valid_morphology_operation(5));
  assert(valid_compound_morphology(3, 3, 1, 0, 1, 1, 1));
  assert(valid_compound_morphology(3, 3, 4, 4, 255, 255, 64));
  assert(!valid_compound_morphology(3, 3, 1, 5, 3, 3, 1));
  assert(!valid_compound_morphology(3, 3, 1, 0, 2, 3, 1));
  assert(!valid_compound_morphology(3, 3, 1, 0, 3, 2, 1));
  assert(!valid_compound_morphology(3, 3, 1, 0, 3, 3, 0));
  assert(!valid_compound_morphology(0, 3, 1, 0, 3, 3, 1));
  const auto down_even = pyramid_down_shape(4, 6, 3);
  assert(down_even.valid && down_even.byte_count == 18);
  const auto down_odd = pyramid_down_shape(3, 5, 1);
  assert(down_odd.valid && down_odd.byte_count == 6);
  assert(!pyramid_down_shape(0, 5, 1).valid);
  const auto up = pyramid_up_shape(2, 3, 4);
  assert(up.valid && up.byte_count == 96);
  assert(!pyramid_up_shape(8193, 8192, 1).valid);
  assert(!pyramid_up_shape(std::numeric_limits<std::int64_t>::max(), 1, 1)
              .valid);
  assert(valid_mask_pair(2, 3, 1, 2, 3, 1));
  assert(valid_mask_pair(2, 3, 3, 2, 3, 1));
  assert(valid_mask_pair(2, 3, 4, 2, 3, 1));
  assert(!valid_mask_pair(2, 3, 2, 2, 3, 1));
  assert(!valid_mask_pair(2, 3, 3, 3, 2, 1));
  assert(!valid_mask_pair(2, 3, 3, 2, 3, 3));
  assert(!valid_mask_pair(0, 3, 3, 0, 3, 1));
  assert(!valid_mask_pair(
      std::numeric_limits<std::int64_t>::max(), 1, 3,
      std::numeric_limits<std::int64_t>::max(), 1, 1));
  assert(valid_blend_mask(2, 3, 1, 2, 3, 1, 2, 3, 1));
  assert(valid_blend_mask(2, 3, 3, 2, 3, 3, 2, 3, 1));
  assert(valid_blend_mask(2, 3, 4, 2, 3, 4, 2, 3, 1));
  assert(!valid_blend_mask(2, 3, 3, 2, 3, 4, 2, 3, 1));
  assert(!valid_blend_mask(2, 3, 3, 3, 2, 3, 2, 3, 1));
  assert(!valid_blend_mask(2, 3, 3, 2, 3, 3, 2, 3, 3));
  assert(!valid_blend_mask(1025, 1024, 1, 1025, 1024, 1, 1025, 1024, 1));
  assert(valid_u8_range(2, 3, 1, 0, 255));
  assert(valid_u8_range(2, 3, 1, 20, 20));
  assert(!valid_u8_range(2, 3, 3, 0, 255));
  assert(!valid_u8_range(2, 3, 1, -1, 255));
  assert(!valid_u8_range(2, 3, 1, 0, 256));
  assert(!valid_u8_range(2, 3, 1, 21, 20));
  assert(!valid_u8_range(0, 3, 1, 0, 255));
  assert(valid_weighted_addition(0.25, 0.75, 10.0));
  assert(valid_weighted_addition(-kMaxImageWeight, kMaxImageWeight,
                                 -kMaxImageOffset));
  assert(!valid_weighted_addition(kMaxImageWeight + 0.01, 0.0, 0.0));
  assert(!valid_weighted_addition(0.0, -kMaxImageWeight - 0.01, 0.0));
  assert(!valid_weighted_addition(0.0, 0.0, kMaxImageOffset + 0.01));
  assert(!valid_weighted_addition(
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
  assert(!valid_weighted_addition(
      0.0, std::numeric_limits<double>::infinity(), 0.0));
  assert(!valid_weighted_addition(
      0.0, 0.0, -std::numeric_limits<double>::infinity()));
  assert(valid_corner_detection(1024, 1024, 1, kMaxDetectedCorners, 0.01,
                                kMaxFeatureDistance));
  assert(!valid_corner_detection(1025, 1024, 1, 1, 0.01, 1.0));
  assert(!valid_corner_detection(2, 2, 3, 1, 0.01, 1.0));
  assert(!valid_corner_detection(2, 2, 1, 0, 0.01, 1.0));
  assert(!valid_corner_detection(2, 2, 1, kMaxDetectedCorners + 1, 0.01,
                                 1.0));
  assert(!valid_corner_detection(2, 2, 1, 1, 0.0, 1.0));
  assert(!valid_corner_detection(2, 2, 1, 1, 1.01, 1.0));
  assert(!valid_corner_detection(
      2, 2, 1, 1, std::numeric_limits<double>::quiet_NaN(), 1.0));
  assert(!valid_corner_detection(
      2, 2, 1, 1, std::numeric_limits<double>::infinity(), 1.0));
  assert(!valid_corner_detection(2, 2, 1, 1, 0.01, -0.1));
  assert(!valid_corner_detection(2, 2, 1, 1, 0.01,
                                 kMaxFeatureDistance + 1.0));
  assert(!valid_corner_detection(
      2, 2, 1, 1, 0.01, std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_corner_detection(
      2, 2, 1, 1, 0.01, std::numeric_limits<double>::infinity()));
  assert(valid_threshold_keypoint_detection(1024, 1024, 1, 255,
                                             kMaxDetectedCorners));
  assert(!valid_threshold_keypoint_detection(1025, 1024, 1, 10, 1));
  assert(!valid_threshold_keypoint_detection(2, 2, 3, 10, 1));
  assert(!valid_threshold_keypoint_detection(2, 2, 1, 0, 1));
  assert(!valid_threshold_keypoint_detection(2, 2, 1, 256, 1));
  assert(!valid_threshold_keypoint_detection(2, 2, 1, 10, 0));
  assert(!valid_threshold_keypoint_detection(
      2, 2, 1, 10, kMaxDetectedCorners + 1));
  assert(valid_corner_derivative_image(5, 7, 1, 3));
  assert(valid_corner_derivative_image(1024, 1024, 1, 7));
  assert(!valid_corner_derivative_image(0, 7, 1, 3));
  assert(!valid_corner_derivative_image(1025, 1024, 1, 3));
  assert(!valid_corner_derivative_image(5, 7, 3, 3));
  assert(!valid_corner_derivative_image(5, 7, 1, 1));
  assert(!valid_corner_derivative_image(5, 7, 1, 9));
  assert(valid_corner_response(5, 7, 1, 3, 3));
  assert(valid_corner_response(31, 31, 1, kMaxCornerBlockSize, 7));
  assert(!valid_corner_response(0, 7, 1, 3, 3));
  assert(!valid_corner_response(5, 7, 3, 3, 3));
  assert(!valid_corner_response(5, 7, 1, 2, 3));
  assert(!valid_corner_response(5, 7, 1, 7, 3));
  assert(!valid_corner_response(5, 7, 1, 3, 9));
  assert(valid_harris_response(5, 7, 1, 3, 3, 0.04));
  assert(valid_harris_response(31, 31, 1, kMaxCornerBlockSize, 7,
                               kMaxHarrisResponseK));
  assert(!valid_harris_response(0, 7, 1, 3, 3, 0.04));
  assert(!valid_harris_response(1025, 1024, 1, 3, 3, 0.04));
  assert(!valid_harris_response(5, 7, 3, 3, 3, 0.04));
  assert(!valid_harris_response(5, 7, 1, 2, 3, 0.04));
  assert(!valid_harris_response(5, 7, 1, 7, 3, 0.04));
  assert(!valid_harris_response(33, 33, 1, 33, 3, 0.04));
  assert(!valid_harris_response(5, 7, 1, 3, 1, 0.04));
  assert(!valid_harris_response(5, 7, 1, 3, 9, 0.04));
  assert(!valid_harris_response(5, 7, 1, 3, 3, 0.0));
  assert(!valid_harris_response(5, 7, 1, 3, 3, 0.26));
  assert(!valid_harris_response(
      5, 7, 1, 3, 3, std::numeric_limits<double>::quiet_NaN()));
  assert(!valid_harris_response(
      5, 7, 1, 3, 3, std::numeric_limits<double>::infinity()));
  static_assert(kMaxDetectedLineSegments == 4096U,
                "line-segment copy cap must remain explicit");
  static_assert(kMaxDetectedLines == 4096U,
                "line copy cap must remain explicit");
  assert(valid_line_detection(7, 7, 1, 3));
  assert(valid_line_detection(1, 7, 1, 7));
  assert(!valid_line_detection(1025, 1024, 1, 1));
  assert(!valid_line_detection(7, 7, 3, 1));
  assert(!valid_line_detection(7, 7, 1, 0));
  assert(!valid_line_detection(7, 7, 1, 8));
  assert(valid_line_segment_detection(7, 7, 1, 3, 4, 0));
  assert(valid_line_segment_detection(1, 7, 1, 7, 6, 6));
  assert(!valid_line_segment_detection(1025, 1024, 1, 1, 0, 0));
  assert(!valid_line_segment_detection(7, 7, 3, 1, 0, 0));
  assert(!valid_line_segment_detection(7, 7, 1, 0, 0, 0));
  assert(!valid_line_segment_detection(7, 7, 1, 8, 0, 0));
  assert(!valid_line_segment_detection(7, 7, 1, 1, -1, 0));
  assert(!valid_line_segment_detection(7, 7, 1, 1, 9, 0));
  assert(!valid_line_segment_detection(7, 7, 1, 1, 0, -1));
  static_assert(kMaxDetectedCircles == 4096U,
                "circle copy cap must remain explicit");
  assert(valid_circle_detection(64, 64, 1, 10.0, 100.0, 10.0, 8, 12));
  assert(!valid_circle_detection(0, 64, 1, 10.0, 100.0, 10.0, 8, 12));
  assert(!valid_circle_detection(1025, 1024, 1, 10.0, 100.0, 10.0, 8,
                                 12));
  assert(!valid_circle_detection(64, 64, 3, 10.0, 100.0, 10.0, 8, 12));
  assert(!valid_circle_detection(64, 64, 1, 0.0, 100.0, 10.0, 8, 12));
  assert(!valid_circle_detection(64, 64, 1, 100.0, 100.0, 10.0, 8, 12));
  assert(!valid_circle_detection(
      64, 64, 1, std::numeric_limits<double>::infinity(), 100.0, 10.0, 8,
      12));
  assert(!valid_circle_detection(64, 64, 1, 10.0, 0.0, 10.0, 8, 12));
  assert(!valid_circle_detection(
      64, 64, 1, 10.0, std::numeric_limits<double>::quiet_NaN(), 10.0, 8,
      12));
  assert(!valid_circle_detection(64, 64, 1, 10.0, 100.0, 0.0, 8, 12));
  assert(!valid_circle_detection(64, 64, 1, 10.0, 100.0, 10.0, 0, 12));
  assert(!valid_circle_detection(64, 64, 1, 10.0, 100.0, 10.0, 12, 8));
  assert(!valid_circle_detection(64, 64, 1, 10.0, 100.0, 10.0, 8, 90));
  assert(!valid_line_segment_detection(7, 7, 1, 1, 0, 9));
  assert(valid_line_drawing(3, 5, 1, 0, 1, 4, 1, 255, 1));
  assert(valid_line_drawing(3, 5, 1, 4, 2, 4, 2, 0,
                            kMaxKernelDimension));
  assert(!valid_line_drawing(1025, 1024, 1, 0, 0, 0, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 3, 0, 0, 0, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, -1, 0, 0, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 5, 0, 0, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, -1, 0, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 3, 0, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, -1, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 5, 0, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 0, -1, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 0, 3, 0, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 0, 0, -1, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 0, 0, 256, 1));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 0, 0, 0, 0));
  assert(!valid_line_drawing(3, 5, 1, 0, 0, 0, 0, 0,
                             kMaxKernelDimension + 1));
  assert(valid_rectangle_drawing(5, 6, 1, 1, 1, 4, 3, 255, 1));
  assert(valid_rectangle_drawing(5, 6, 1, 5, 4, 1, 1, 0,
                                 kMaxKernelDimension));
  assert(!valid_rectangle_drawing(1025, 1024, 1, 0, 0, 1, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 3, 0, 0, 1, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, -1, 0, 1, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, -1, 1, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 0, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 1, 0, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 7, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 1, 6, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 5, 0, 2, 1, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 4, 1, 2, 0, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 1, 1, -1, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 1, 1, 256, 1));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 1, 1, 0, 0));
  assert(!valid_rectangle_drawing(5, 6, 1, 0, 0, 1, 1, 0,
                                  kMaxKernelDimension + 1));
  assert(valid_rectangle_fill(5, 6, 1, 1, 1, 4, 3, 255));
  assert(valid_rectangle_fill(5, 6, 1, 5, 4, 1, 1, 0));
  assert(!valid_rectangle_fill(1025, 1024, 1, 0, 0, 1, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 3, 0, 0, 1, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 1, -1, 0, 1, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, -1, 1, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 0, 0, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 0, 1, 0, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 0, 7, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 0, 1, 6, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 5, 0, 2, 1, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 4, 1, 2, 0));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 0, 1, 1, -1));
  assert(!valid_rectangle_fill(5, 6, 1, 0, 0, 1, 1, 256));
  assert(valid_circle_drawing(5, 5, 1, 2, 2, 1, 255, 1));
  assert(valid_circle_drawing(1, 1, 1, 0, 0, 0, 0, 1));
  assert(valid_circle_drawing(7, 7, 1, 3, 3, 1, 0, 3));
  assert(!valid_circle_drawing(1025, 1024, 1, 1, 1, 0, 0, 1));
  assert(!valid_circle_drawing(5, 5, 3, 2, 2, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, -1, 2, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 5, 2, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, -1, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 5, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, -1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, 5, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 0, 2, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 0, 1, 0, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, 0, 0, 2));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, 1, -1, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, 1, 256, 1));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, 1, 0, 0));
  assert(!valid_circle_drawing(5, 5, 1, 2, 2, 1, 0,
                               kMaxKernelDimension + 1));
  assert(valid_circle_fill(5, 5, 1, 2, 2, 1, 255));
  assert(valid_circle_fill(1, 1, 1, 0, 0, 0, 0));
  assert(!valid_circle_fill(1025, 1024, 1, 1, 1, 0, 0));
  assert(!valid_circle_fill(5, 5, 3, 2, 2, 1, 0));
  assert(!valid_circle_fill(5, 5, 1, -1, 2, 0, 0));
  assert(!valid_circle_fill(5, 5, 1, 5, 2, 0, 0));
  assert(!valid_circle_fill(5, 5, 1, 2, -1, 0, 0));
  assert(!valid_circle_fill(5, 5, 1, 2, 5, 0, 0));
  assert(!valid_circle_fill(5, 5, 1, 2, 2, -1, 0));
  assert(!valid_circle_fill(5, 5, 1, 2, 2, 5, 0));
  assert(!valid_circle_fill(5, 5, 1, 0, 2, 1, 0));
  assert(!valid_circle_fill(5, 5, 1, 2, 0, 1, 0));
  assert(!valid_circle_fill(5, 5, 1, 2, 2, 1, -1));
  assert(!valid_circle_fill(5, 5, 1, 2, 2, 1, 256));
  assert(valid_cross_drawing(5, 5, 1, 2, 2, 3, 255, 1));
  assert(valid_cross_drawing(1, 1, 1, 0, 0, 1, 0, 1));
  assert(!valid_cross_drawing(1025, 1024, 1, 1, 1, 1, 0, 1));
  assert(!valid_cross_drawing(5, 5, 3, 2, 2, 3, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, -1, 2, 1, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 5, 2, 1, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, -1, 1, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 5, 1, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2, 0, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2, 2, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2,
                              kMaxKernelDimension + 2, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 0, 2, 3, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 0, 3, 0, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2, 1, -1, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2, 1, 256, 1));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2, 1, 0, 0));
  assert(!valid_cross_drawing(5, 5, 1, 2, 2, 1, 0,
                              kMaxKernelDimension + 1));
  assert(valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 1, 255, 1));
  assert(valid_ellipse_drawing(7, 9, 1, 4, 3, 2, 1, 0, 3));
  assert(!valid_ellipse_drawing(1025, 1024, 1, 3, 2, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 3, 3, 2, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, -1, 2, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 7, 2, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, -1, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 5, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 0, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 7, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 0, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 5, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 1, 2, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 0, 2, 1, 0, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 1, -1, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 1, 256, 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 1, 0, 0));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 1, 0,
                                kMaxKernelDimension + 1));
  assert(!valid_ellipse_drawing(5, 7, 1, 3, 2, 2, 1, 0, 5));
  assert(valid_ellipse_fill(5, 7, 1, 3, 2, 2, 1, 255));
  assert(!valid_ellipse_fill(1025, 1024, 1, 3, 2, 2, 1, 0));
  assert(!valid_ellipse_fill(5, 7, 3, 3, 2, 2, 1, 0));
  assert(!valid_ellipse_fill(5, 7, 1, 1, 2, 2, 1, 0));
  assert(!valid_ellipse_fill(5, 7, 1, 3, 0, 2, 1, 0));
  assert(!valid_ellipse_fill(5, 7, 1, 3, 2, 0, 1, 0));
  assert(!valid_ellipse_fill(5, 7, 1, 3, 2, 2, 0, 0));
  assert(!valid_ellipse_fill(5, 7, 1, 3, 2, 2, 1, -1));
  assert(!valid_ellipse_fill(5, 7, 1, 3, 2, 2, 1, 256));

  const auto bgr = image_shape_for_layout(2, 3, 3);
  assert(bgr.valid);
  assert(bgr.channels == 3);
  assert(bgr.byte_count == 18);
  assert(valid_buffer_size(bgr, 18));
  assert(!valid_buffer_size(bgr, 17));
  assert(!valid_buffer_size(bgr, 19));

  assert(!image_shape_for_layout(0, 1, 1).valid);
  assert(!image_shape_for_layout(-1, 1, 1).valid);
  assert(!image_shape_for_layout(1, -1, 1).valid);
  assert(!image_shape_for_layout(1, 1, 99).valid);
  assert(!image_shape(std::numeric_limits<std::int64_t>::max(), 1, 1).valid);
  assert(!image_shape(1, std::numeric_limits<std::int64_t>::max(), 1).valid);
  assert(!image_shape(1, 1, 0).valid);

  const auto padded = padded_image_shape(2, 3, 3, 1, 2, 3, 4);
  assert(padded.valid);
  assert(padded.byte_count == 5 * 10 * 3);
  assert(!padded_image_shape(2, 3, 3, -1, 0, 0, 0).valid);
  assert(!padded_image_shape(2, 3, 3, 0, -1, 0, 0).valid);
  assert(!padded_image_shape(2, 3, 3, 0, 0, -1, 0).valid);
  assert(!padded_image_shape(2, 3, 3, 0, 0, 0, -1).valid);
  assert(!padded_image_shape(2, 3, 3,
                             std::numeric_limits<std::int64_t>::max(), 0, 0,
                             0)
              .valid);
  assert(!padded_image_shape(16384, 16384, 1, 1, 0, 0, 0).valid);

  const auto clockwise = rotated_image_shape(2, 3, 4, 0);
  assert(clockwise.valid && clockwise.byte_count == 3 * 2 * 4);
  const auto half_turn = rotated_image_shape(2, 3, 4, 1);
  assert(half_turn.valid && half_turn.byte_count == 2 * 3 * 4);
  assert(!rotated_image_shape(2, 3, 4, 99).valid);

  const auto exact_limit = image_shape(16384, 16384, 1);
  assert(exact_limit.valid);
  assert(exact_limit.byte_count == kMaxOwnedImageBytes);
  assert(!image_shape(16384, 16385, 1).valid);
  assert(!image_shape(16384, 16384, 3).valid);

  const auto bgr_to_gray = color_conversion_shape(2, 3, 3, 0);
  assert(bgr_to_gray.valid && bgr_to_gray.output_channels == 1);
  assert(bgr_to_gray.output_byte_count == 6);
  assert(color_conversion_shape(2, 3, 1, 1).valid);
  assert(color_conversion_shape(2, 3, 3, 2).valid);
  assert(color_conversion_shape(2, 3, 4, 3).valid);
  assert(!color_conversion_shape(2, 3, 1, 0).valid);
  assert(!color_conversion_shape(2, 3, 3, 1).valid);
  assert(!color_conversion_shape(2, 3, 4, 2).valid);
  assert(!color_conversion_shape(2, 3, 3, 3).valid);
  assert(!color_conversion_shape(2, 3, 3, 99).valid);
  assert(!color_conversion_shape(0, 3, 3, 0).valid);

  assert(valid_analysis_point(5, 7, 0.0, 0.0));
  assert(valid_analysis_point(5, 7, 6.0, 4.0));
  assert(valid_analysis_point(1, 1, 0.0, 0.0));
  assert(!valid_analysis_point(0, 7, 0.0, 0.0));
  assert(!valid_analysis_point(5, 0, 0.0, 0.0));
  assert(!valid_analysis_point(5, 7, -0.1, 0.0));
  assert(!valid_analysis_point(5, 7, 6.1, 0.0));
  assert(!valid_analysis_point(5, 7, 0.0, -0.1));
  assert(!valid_analysis_point(5, 7, 0.0, 4.1));
  assert(!valid_analysis_point(
      5, 7, std::numeric_limits<double>::quiet_NaN(), 0.0));
  assert(!valid_analysis_point(
      5, 7, 0.0, std::numeric_limits<double>::infinity()));

  assert(valid_sigma(0.0));
  assert(valid_sigma(1.25));
  assert(!valid_sigma(-0.01));
  assert(!valid_sigma(std::numeric_limits<double>::infinity()));
  assert(!valid_sigma(-std::numeric_limits<double>::infinity()));
  assert(!valid_sigma(std::numeric_limits<double>::quiet_NaN()));

  using terlan_opencv::failure::clear;
  using terlan_opencv::failure::set;
  using terlan_opencv::failure::take;
  clear();
  assert(take() == static_cast<std::int64_t>(NativeFailure::None));
  set(NativeFailure::InvalidArgument);
  assert(take() == static_cast<std::int64_t>(NativeFailure::InvalidArgument));
  assert(take() == static_cast<std::int64_t>(NativeFailure::None));

  set(NativeFailure::AllocationFailed);
  std::int64_t child_initial = -1;
  std::int64_t child_failure = -1;
  std::thread child([&] {
    child_initial = take();
    set(NativeFailure::NativeException);
    child_failure = take();
  });
  child.join();
  assert(child_initial == static_cast<std::int64_t>(NativeFailure::None));
  assert(child_failure ==
         static_cast<std::int64_t>(NativeFailure::NativeException));
  assert(take() == static_cast<std::int64_t>(NativeFailure::AllocationFailed));

  clear();
  auto success = terlan_opencv::failure::contain<int>(
      [] { return std::make_unique<int>(42); });
  assert(success != nullptr && *success == 42);
  assert(take() == static_cast<std::int64_t>(NativeFailure::None));

  auto allocation_failure = terlan_opencv::failure::contain<int>(
      []() -> std::unique_ptr<int> { throw std::bad_alloc(); });
  assert(allocation_failure == nullptr);
  assert(take() == static_cast<std::int64_t>(NativeFailure::AllocationFailed));

  auto standard_failure = terlan_opencv::failure::contain<int>(
      []() -> std::unique_ptr<int> { throw std::runtime_error("private"); });
  assert(standard_failure == nullptr);
  assert(take() == static_cast<std::int64_t>(NativeFailure::NativeException));

  auto unknown_failure = terlan_opencv::failure::contain<int>(
      []() -> std::unique_ptr<int> { throw 42; });
  assert(unknown_failure == nullptr);
  assert(take() == static_cast<std::int64_t>(NativeFailure::NativeFailure));
}
