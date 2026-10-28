#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <limits>
#include <utility>

namespace terlan_opencv {
namespace {

/// Identifies one package-owned binary bitwise operation.
enum class BinaryBitwiseOperation {
  And,
  Or,
  Xor,
};

/// Applies one validated binary bitwise operation into fresh matching storage.
std::unique_ptr<OpenCvMat> apply_binary_bitwise(
    const cv::Mat& first, const cv::Mat& second,
    BinaryBitwiseOperation operation, const cv::Mat* mask = nullptr) {
  cv::Mat result;
  switch (operation) {
    case BinaryBitwiseOperation::And:
      if (mask == nullptr) {
        cv::bitwise_and(first, second, result);
      } else {
        cv::bitwise_and(first, second, result, *mask);
      }
      break;
    case BinaryBitwiseOperation::Or:
      if (mask == nullptr) {
        cv::bitwise_or(first, second, result);
      } else {
        cv::bitwise_or(first, second, result, *mask);
      }
      break;
    case BinaryBitwiseOperation::Xor:
      if (mask == nullptr) {
        cv::bitwise_xor(first, second, result);
      } else {
        cv::bitwise_xor(first, second, result, *mask);
      }
      break;
  }
  return own_matching_mat(std::move(result), first);
}

/// Lowers one validated semantic comparison mode to its private OpenCV code.
int comparison_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(ComparisonMode::Equal):
      return cv::CMP_EQ;
    case static_cast<std::int64_t>(ComparisonMode::NotEqual):
      return cv::CMP_NE;
    case static_cast<std::int64_t>(ComparisonMode::Less):
      return cv::CMP_LT;
    case static_cast<std::int64_t>(ComparisonMode::LessOrEqual):
      return cv::CMP_LE;
    case static_cast<std::int64_t>(ComparisonMode::Greater):
      return cv::CMP_GT;
    case static_cast<std::int64_t>(ComparisonMode::GreaterOrEqual):
      return cv::CMP_GE;
    default:
      return -1;
  }
}

}  // namespace

std::unique_ptr<OpenCvMat> OpenCvMat::abs_diff_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::absdiff(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::add_saturated_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::add(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::subtract_saturated_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::subtract(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::minimum_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::min(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::maximum_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::max(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::compare_mask_owned(
    const OpenCvMat& other, std::int64_t mode) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !validation::valid_comparison_mode(mode)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, mode] {
    cv::Mat result;
    cv::compare(impl_->value, other.impl_->value, result,
                comparison_code(mode));
    return own_comparison_mask(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::multiply_scaled_owned(
    const OpenCvMat& other, double scale) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !validation::valid_multiply_scale(scale)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, scale] {
    cv::Mat result;
    cv::multiply(impl_->value, other.impl_->value, result, scale);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::divide_scaled_owned(
    const OpenCvMat& other, double scale) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !validation::valid_divide_scale(scale)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, scale] {
    cv::Mat result;
    cv::divide(impl_->value, other.impl_->value, result, scale);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::gradient_magnitude_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_u8_grayscale_images(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat first_float;
    cv::Mat second_float;
    cv::Mat magnitude_float;
    cv::Mat result;
    impl_->value.convertTo(first_float, CV_32F);
    other.impl_->value.convertTo(second_float, CV_32F);
    cv::magnitude(first_float, second_float, magnitude_float);
    magnitude_float.convertTo(result, CV_8U);
    return own_grayscale_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::add_weighted_owned(
    const OpenCvMat& other, double first_weight, double second_weight,
    double offset) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !validation::valid_weighted_addition(first_weight, second_weight,
                                           offset)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, first_weight,
                                     second_weight, offset] {
    cv::Mat result;
    cv::addWeighted(impl_->value, first_weight, other.impl_->value,
                    second_weight, offset, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::mean_absolute_error_channels_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_u8_images(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, &other] {
    cv::Mat difference;
    cv::absdiff(impl_->value, other.impl_->value, difference);
    return copy_u8_domain_scalar_values(cv::mean(difference),
                                        impl_->value.channels());
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::concat_horizontal_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  const bool compatible =
      impl_ != nullptr && other.impl_ != nullptr &&
      compatible_matrix_types(impl_->value, other.impl_->value) &&
      impl_->value.rows == other.impl_->value.rows &&
      impl_->value.cols <= max_dimension - other.impl_->value.cols;
  const auto shape =
      compatible
          ? resized_matrix_shape(
                impl_->value, impl_->value.rows,
                static_cast<std::int64_t>(impl_->value.cols) +
                    other.impl_->value.cols)
          : validation::ImageShape{false, 0, 0};
  if (!shape.valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const std::int64_t output_cols =
      static_cast<std::int64_t>(impl_->value.cols) + other.impl_->value.cols;
  return failure::contain<OpenCvMat>([this, &other, output_cols] {
    cv::Mat result;
    cv::hconcat(impl_->value, other.impl_->value, result);
    return own_resized_mat(std::move(result), impl_->value, impl_->value.rows,
                           output_cols);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::concat_vertical_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  const bool compatible =
      impl_ != nullptr && other.impl_ != nullptr &&
      compatible_matrix_types(impl_->value, other.impl_->value) &&
      impl_->value.cols == other.impl_->value.cols &&
      impl_->value.rows <= max_dimension - other.impl_->value.rows;
  const auto shape =
      compatible
          ? resized_matrix_shape(
                impl_->value,
                static_cast<std::int64_t>(impl_->value.rows) +
                    other.impl_->value.rows,
                impl_->value.cols)
          : validation::ImageShape{false, 0, 0};
  if (!shape.valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const std::int64_t output_rows =
      static_cast<std::int64_t>(impl_->value.rows) + other.impl_->value.rows;
  return failure::contain<OpenCvMat>([this, &other, output_rows] {
    cv::Mat result;
    cv::vconcat(impl_->value, other.impl_->value, result);
    return own_resized_mat(std::move(result), impl_->value, output_rows,
                           impl_->value.cols);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::transpose_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    cv::Mat result;
    cv::transpose(impl_->value, result);
    return own_resized_mat(std::move(result), impl_->value, impl_->value.cols,
                           impl_->value.rows);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::tile_owned(
    std::int64_t vertical_count,
    std::int64_t horizontal_count) const noexcept {
  failure::clear();
  const auto max_dimension =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  const bool counts_valid =
      impl_ != nullptr && valid_owned_matrix(impl_->value) &&
      vertical_count > 0 && horizontal_count > 0 &&
      vertical_count <= max_dimension / impl_->value.rows &&
      horizontal_count <= max_dimension / impl_->value.cols;
  const auto shape =
      counts_valid
          ? resized_matrix_shape(impl_->value,
                                 impl_->value.rows * vertical_count,
                                 impl_->value.cols * horizontal_count)
          : validation::ImageShape{false, 0, 0};
  if (!shape.valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const std::int64_t output_rows =
      static_cast<std::int64_t>(impl_->value.rows) * vertical_count;
  const std::int64_t output_cols =
      static_cast<std::int64_t>(impl_->value.cols) * horizontal_count;
  return failure::contain<OpenCvMat>(
      [this, vertical_count, horizontal_count, output_rows, output_cols] {
        cv::Mat result;
        cv::repeat(impl_->value, static_cast<int>(vertical_count),
                   static_cast<int>(horizontal_count), result);
        return own_resized_mat(std::move(result), impl_->value, output_rows,
                               output_cols);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::scale_abs_owned(
    double scale, double offset) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      !std::isfinite(scale) || !std::isfinite(offset)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, scale, offset] {
    cv::Mat result;
    cv::convertScaleAbs(impl_->value, result, scale, offset);
    return own_comparison_mask(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::invert_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    cv::Mat result;
    cv::bitwise_not(impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::invert_masked_owned(
    const OpenCvMat& mask) const noexcept {
  failure::clear();
  if (impl_ == nullptr || mask.impl_ == nullptr ||
      !compatible_mask(impl_->value, mask.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &mask] {
    cv::Mat inverted;
    cv::bitwise_not(impl_->value, inverted);
    cv::Mat result = cv::Mat::zeros(impl_->value.size(), impl_->value.type());
    inverted.copyTo(result, mask.impl_->value);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bitwise_and_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    return apply_binary_bitwise(impl_->value, other.impl_->value,
                                BinaryBitwiseOperation::And);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bitwise_and_masked_owned(
    const OpenCvMat& other, const OpenCvMat& mask) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || mask.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !compatible_mask(impl_->value, mask.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, &mask] {
    return apply_binary_bitwise(impl_->value, other.impl_->value,
                                BinaryBitwiseOperation::And,
                                &mask.impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bitwise_or_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    return apply_binary_bitwise(impl_->value, other.impl_->value,
                                BinaryBitwiseOperation::Or);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bitwise_or_masked_owned(
    const OpenCvMat& other, const OpenCvMat& mask) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || mask.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !compatible_mask(impl_->value, mask.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, &mask] {
    return apply_binary_bitwise(impl_->value, other.impl_->value,
                                BinaryBitwiseOperation::Or,
                                &mask.impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bitwise_xor_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    return apply_binary_bitwise(impl_->value, other.impl_->value,
                                BinaryBitwiseOperation::Xor);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bitwise_xor_masked_owned(
    const OpenCvMat& other, const OpenCvMat& mask) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || mask.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !compatible_mask(impl_->value, mask.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, &mask] {
    return apply_binary_bitwise(impl_->value, other.impl_->value,
                                BinaryBitwiseOperation::Xor,
                                &mask.impl_->value);
  });
}

}  // namespace terlan_opencv
