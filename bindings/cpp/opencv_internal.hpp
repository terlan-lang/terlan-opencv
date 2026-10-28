#pragma once

#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_validation.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <utility>

namespace terlan_opencv {

/// Private matrix representation shared by package-owned adapter units.
struct OpenCvMat::Impl final {
  /// Takes ownership of one OpenCV matrix value.
  explicit Impl(cv::Mat value) : value(std::move(value)) {}

  /// Owned OpenCV matrix storage.
  cv::Mat value;
};

/// Returns the checked package storage shape for one two-dimensional matrix.
inline validation::ImageShape matrix_storage_shape(
    const cv::Mat& matrix) noexcept {
  if (matrix.dims != 2) {
    return {false, 0, 0};
  }
  return validation::matrix_shape(matrix.rows, matrix.cols, matrix.channels(),
                                  matrix.elemSize1());
}

/// Reports whether one matrix fits the complete package-owned storage contract.
inline bool valid_owned_matrix(const cv::Mat& matrix) noexcept {
  const auto shape = matrix_storage_shape(matrix);
  return shape.valid && matrix.total() * matrix.elemSize() == shape.byte_count;
}

/// Reports whether two matrices have identical valid shape and scalar type.
inline bool compatible_matrices(const cv::Mat& first,
                                const cv::Mat& second) noexcept {
  return valid_owned_matrix(first) && valid_owned_matrix(second) &&
         first.rows == second.rows && first.cols == second.cols &&
         first.type() == second.type();
}

/// Reports whether two valid matrices share one scalar and channel type.
inline bool compatible_matrix_types(const cv::Mat& first,
                                    const cv::Mat& second) noexcept {
  return valid_owned_matrix(first) && valid_owned_matrix(second) &&
         first.type() == second.type();
}

/// Reports whether one matrix has valid single-channel byte mask storage.
inline bool valid_mask_storage(const cv::Mat& mask) noexcept {
  return valid_owned_matrix(mask) && mask.type() == CV_8UC1;
}

/// Reports whether one byte mask covers the complete two-dimensional matrix.
inline bool compatible_mask(const cv::Mat& matrix,
                            const cv::Mat& mask) noexcept {
  return valid_owned_matrix(matrix) && valid_mask_storage(mask) &&
         matrix.rows == mask.rows && matrix.cols == mask.cols;
}

/// Reports whether two byte masks have identical valid shape.
inline bool compatible_masks(const cv::Mat& first,
                             const cv::Mat& second) noexcept {
  return valid_mask_storage(first) && valid_mask_storage(second) &&
         first.rows == second.rows && first.cols == second.cols;
}

/// Computes a checked output allocation preserving one matrix type.
inline validation::ImageShape resized_matrix_shape(
    const cv::Mat& source, std::int64_t rows, std::int64_t cols) noexcept {
  return valid_owned_matrix(source)
             ? validation::matrix_shape(rows, cols, source.channels(),
                                        source.elemSize1())
             : validation::ImageShape{false, 0, 0};
}

/// Reports whether two matrices share one valid public unsigned-byte image type.
inline bool compatible_u8_images(const cv::Mat& first,
                                 const cv::Mat& second) noexcept {
  return validation::valid_image_pair(
             first.rows, first.cols, first.channels(), second.rows,
             second.cols, second.channels()) &&
         first.depth() == CV_8U && first.type() == second.type();
}

/// Reports whether one matrix has valid unsigned-byte grayscale storage.
inline bool valid_u8_grayscale_image(const cv::Mat& image) noexcept {
  return validation::valid_grayscale_image(
             image.rows, image.cols, image.channels()) &&
         image.type() == CV_8UC1;
}

/// Reports whether two matrices share valid unsigned-byte grayscale storage.
inline bool compatible_u8_grayscale_images(const cv::Mat& first,
                                           const cv::Mat& second) noexcept {
  return valid_u8_grayscale_image(first) &&
         valid_u8_grayscale_image(second) && first.rows == second.rows &&
         first.cols == second.cols;
}

/// Converts a validated expected-shape result into fresh adapter ownership.
inline std::unique_ptr<OpenCvMat> own_resized_mat(
    cv::Mat result, const cv::Mat& source, std::int64_t rows,
    std::int64_t cols) {
  if (result.rows != rows || result.cols != cols ||
      result.type() != source.type()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(result)));
}

/// Converts a shape- and type-preserving result into fresh adapter ownership.
inline std::unique_ptr<OpenCvMat> own_matching_mat(
    cv::Mat result, const cv::Mat& source) {
  return own_resized_mat(std::move(result), source, source.rows, source.cols);
}

/// Converts a same-shape public channel layout into fresh adapter ownership.
inline std::unique_ptr<OpenCvMat> own_channel_count_mat(
    cv::Mat result, const cv::Mat& source, int channels) {
  if ((channels != 1 && channels != 3 && channels != 4) ||
      result.rows != source.rows || result.cols != source.cols ||
      result.type() != CV_MAKETYPE(CV_8U, channels)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(result)));
}

/// Converts a same-shape typed channel result into fresh adapter ownership.
inline std::unique_ptr<OpenCvMat> own_typed_channel_count_mat(
    cv::Mat result, const cv::Mat& source, int channels) {
  if (channels <= 0 || channels > validation::kMaxMatrixChannels ||
      result.rows != source.rows || result.cols != source.cols ||
      result.type() != CV_MAKETYPE(source.depth(), channels) ||
      !valid_owned_matrix(result)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(result)));
}

/// Converts one comparison result into a fresh same-channel byte mask owner.
inline std::unique_ptr<OpenCvMat> own_comparison_mask(
    cv::Mat result, const cv::Mat& source) {
  if (result.rows != source.rows || result.cols != source.cols ||
      result.type() != CV_MAKETYPE(CV_8U, source.channels())) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(result)));
}

/// Converts one same-shape single-channel result into fresh adapter ownership.
inline std::unique_ptr<OpenCvMat> own_grayscale_mat(cv::Mat result,
                                                   const cv::Mat& source) {
  return own_channel_count_mat(std::move(result), source, 1);
}

/// Converts one expected-size grayscale result into fresh adapter ownership.
inline std::unique_ptr<OpenCvMat> own_grayscale_mat(
    cv::Mat result, std::int64_t rows, std::int64_t cols) {
  if (!validation::valid_grayscale_image(rows, cols, 1) ||
      result.rows != rows || result.cols != cols || result.type() != CV_8UC1) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(result)));
}

/// Copies finite scalar values constrained to the unsigned-byte value domain.
inline std::unique_ptr<std::vector<double>> copy_u8_domain_scalar_values(
    const cv::Scalar& values, int count) {
  auto result = std::make_unique<std::vector<double>>();
  result->reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const double value = values[index];
    if (!std::isfinite(value) || value < 0.0 || value > 255.0) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    result->push_back(value);
  }
  return result;
}

/// Copies one finite row-major interleaved floating-point matrix.
inline std::unique_ptr<std::vector<double>> copy_finite_f32_channels(
    const cv::Mat& values, int expected_rows, int expected_cols,
    int expected_channels) {
  if (expected_channels <= 0 || expected_channels > CV_CN_MAX ||
      values.rows != expected_rows || values.cols != expected_cols ||
      values.type() != CV_MAKETYPE(CV_32F, expected_channels)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  const auto scalar_count =
      values.total() * static_cast<std::size_t>(expected_channels);
  auto result = std::make_unique<std::vector<double>>();
  result->reserve(scalar_count);
  const int row_scalars = values.cols * expected_channels;
  for (int row = 0; row < values.rows; ++row) {
    const auto* row_values = values.ptr<float>(row);
    for (int index = 0; index < row_scalars; ++index) {
      const double value = row_values[index];
      if (!std::isfinite(value)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return nullptr;
      }
      result->push_back(value);
    }
  }
  return result;
}

/// Copies one finite row-major single-channel floating-point matrix.
inline std::unique_ptr<std::vector<double>> copy_finite_f32_values(
    const cv::Mat& values, int expected_rows, int expected_cols) {
  return copy_finite_f32_channels(values, expected_rows, expected_cols, 1);
}

}  // namespace terlan_opencv
