#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_mask_analysis.hpp"
#include "opencv_validation.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <utility>

namespace terlan_opencv {
std::unique_ptr<OpenCvMat> OpenCvMat::apply_mask_owned(
    const OpenCvMat& mask) const noexcept {
  failure::clear();
  if (impl_ == nullptr || mask.impl_ == nullptr ||
      !compatible_mask(impl_->value, mask.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &mask] {
    cv::Mat result;
    cv::bitwise_and(impl_->value, impl_->value, result, mask.impl_->value);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::select_mask_owned(
    const OpenCvMat& other, const OpenCvMat& mask) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || mask.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value) ||
      !compatible_mask(impl_->value, mask.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, &mask] {
    cv::Mat result = other.impl_->value.clone();
    impl_->value.copyTo(result, mask.impl_->value);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::blend_mask_owned(
    const OpenCvMat& other, const OpenCvMat& weights) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || weights.impl_ == nullptr ||
      !compatible_u8_images(impl_->value, other.impl_->value) ||
      weights.impl_->value.type() != CV_8UC1 ||
      !validation::valid_blend_mask(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          other.impl_->value.rows, other.impl_->value.cols,
          other.impl_->value.channels(), weights.impl_->value.rows,
          weights.impl_->value.cols, weights.impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other, &weights] {
    cv::Mat first_weights;
    weights.impl_->value.convertTo(first_weights, CV_32FC1, 1.0 / 255.0);
    cv::Mat second_weights;
    cv::subtract(cv::Scalar::all(1.0), first_weights, second_weights);
    cv::Mat result;
    cv::blendLinear(impl_->value, other.impl_->value, first_weights,
                    second_weights, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::in_range_owned(
    std::int64_t lower, std::int64_t upper) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.type() != CV_8UC1 ||
      !validation::valid_u8_range(impl_->value.rows, impl_->value.cols,
                                 impl_->value.channels(), lower, upper)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, lower, upper] {
    cv::Mat result;
    cv::inRange(impl_->value, cv::Scalar(lower), cv::Scalar(upper), result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::mask_and_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_masks(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::bitwise_and(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::mask_or_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_masks(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::bitwise_or(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::mask_xor_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_masks(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::bitwise_xor(impl_->value, other.impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::mask_not_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_mask_storage(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    cv::Mat result;
    cv::bitwise_not(impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::non_zero_points_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.type() != CV_8UC1 ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    std::vector<cv::Point> native_points;
    if (!collect_non_zero_points(impl_->value, native_points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto points = std::make_unique<std::vector<std::int64_t>>();
    points->reserve(native_points.size() * 2U);
    for (const auto& point : native_points) {
      points->push_back(point.x);
      points->push_back(point.y);
    }
    return points;
  });
}

}  // namespace terlan_opencv
