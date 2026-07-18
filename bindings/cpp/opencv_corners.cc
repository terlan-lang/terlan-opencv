#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace terlan_opencv {

namespace {

/// Orders FAST candidates by strength before deterministic truncation.
bool stronger_keypoint(const cv::KeyPoint& left,
                       const cv::KeyPoint& right) noexcept {
  if (left.response != right.response) {
    return left.response > right.response;
  }
  if (left.pt.y != right.pt.y) {
    return left.pt.y < right.pt.y;
  }
  return left.pt.x < right.pt.x;
}

/// Orders retained keypoints canonically in row-major coordinate order.
bool earlier_keypoint(const cv::KeyPoint& left,
                      const cv::KeyPoint& right) noexcept {
  return left.pt.y < right.pt.y ||
         (left.pt.y == right.pt.y && left.pt.x < right.pt.x);
}

/// Validates, bounds, orders, and copies one detector's keypoints.
std::unique_ptr<std::vector<double>> copy_detected_keypoints(
    std::vector<cv::KeyPoint> keypoints, int rows, int cols,
    std::int64_t max_keypoints) {
  const auto total = static_cast<std::size_t>(rows) *
                     static_cast<std::size_t>(cols);
  if (keypoints.size() > total) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  for (const auto& keypoint : keypoints) {
    if (!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y) ||
        !std::isfinite(keypoint.response) || keypoint.response < 0.0F ||
        keypoint.pt.x < 0.0F || keypoint.pt.y < 0.0F ||
        keypoint.pt.x >= static_cast<float>(cols) ||
        keypoint.pt.y >= static_cast<float>(rows)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
  }
  std::sort(keypoints.begin(), keypoints.end(), stronger_keypoint);
  if (keypoints.size() > static_cast<std::size_t>(max_keypoints)) {
    keypoints.resize(static_cast<std::size_t>(max_keypoints));
  }
  std::sort(keypoints.begin(), keypoints.end(), earlier_keypoint);
  for (std::size_t index = 1; index < keypoints.size(); ++index) {
    if (keypoints[index - 1].pt == keypoints[index].pt) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
  }
  auto result = std::make_unique<std::vector<double>>();
  result->reserve(keypoints.size() * 2);
  for (const auto& keypoint : keypoints) {
    result->push_back(keypoint.pt.x);
    result->push_back(keypoint.pt.y);
  }
  return result;
}

}  // namespace

std::unique_ptr<std::vector<double>>
OpenCvMat::good_features_to_track_owned(
    std::int64_t max_corners, double quality_level,
    double minimum_distance) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_corner_detection(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          max_corners, quality_level, minimum_distance)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, max_corners,
                                                 quality_level,
                                                 minimum_distance] {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(impl_->value, corners,
                            static_cast<int>(max_corners), quality_level,
                            minimum_distance);
    if (corners.size() > static_cast<std::size_t>(max_corners)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }

    auto coordinates = std::make_unique<std::vector<double>>();
    coordinates->reserve(corners.size() * 2);
    for (const auto& corner : corners) {
      const auto x = static_cast<double>(corner.x);
      const auto y = static_cast<double>(corner.y);
      if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || y < 0.0 ||
          x >= impl_->value.cols || y >= impl_->value.rows) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      coordinates->push_back(x);
      coordinates->push_back(y);
    }
    return coordinates;
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::fast_keypoints_owned(
    std::int64_t threshold, bool nonmax_suppression,
    std::int64_t max_keypoints) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_threshold_keypoint_detection(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          threshold, max_keypoints)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, threshold, nonmax_suppression, max_keypoints] {
        std::vector<cv::KeyPoint> keypoints;
        cv::FAST(impl_->value, keypoints, static_cast<int>(threshold),
                 nonmax_suppression, cv::FastFeatureDetector::TYPE_9_16);
        return copy_detected_keypoints(std::move(keypoints), impl_->value.rows,
                                       impl_->value.cols, max_keypoints);
      });
}

std::unique_ptr<std::vector<double>> OpenCvMat::agast_keypoints_owned(
    std::int64_t threshold, bool nonmax_suppression,
    std::int64_t max_keypoints) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_threshold_keypoint_detection(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          threshold, max_keypoints)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, threshold, nonmax_suppression, max_keypoints] {
        std::vector<cv::KeyPoint> keypoints;
        cv::AGAST(impl_->value, keypoints, static_cast<int>(threshold),
                  nonmax_suppression,
                  cv::AgastFeatureDetector::OAST_9_16);
        return copy_detected_keypoints(std::move(keypoints), impl_->value.rows,
                                       impl_->value.cols, max_keypoints);
      });
}

std::unique_ptr<std::vector<double>> OpenCvMat::harris_response_owned(
    std::int64_t block_size, std::int64_t aperture_size,
    double response_k) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_harris_response(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          block_size, aperture_size, response_k)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, block_size, aperture_size, response_k] {
        cv::Mat response;
        cv::cornerHarris(impl_->value, response, static_cast<int>(block_size),
                         static_cast<int>(aperture_size), response_k,
                         cv::BORDER_REFLECT_101);
        return copy_finite_f32_values(response, impl_->value.rows,
                                      impl_->value.cols);
      });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::corner_min_eigenvalue_owned(
    std::int64_t block_size, std::int64_t aperture_size) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_corner_response(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          block_size, aperture_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, block_size, aperture_size] {
        cv::Mat response;
        cv::cornerMinEigenVal(impl_->value, response,
                              static_cast<int>(block_size),
                              static_cast<int>(aperture_size),
                              cv::BORDER_REFLECT_101);
        auto values = copy_finite_f32_values(response, impl_->value.rows,
                                              impl_->value.cols);
        if (values == nullptr) {
          return std::unique_ptr<std::vector<double>>{};
        }
        for (double& value : *values) {
          if (value < -1.0e-6) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<std::vector<double>>{};
          }
          value = value < 0.0 ? 0.0 : value;
        }
        return values;
      });
}

std::unique_ptr<std::vector<double>> OpenCvMat::pre_corner_response_owned(
    std::int64_t aperture_size) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_corner_derivative_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          aperture_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, aperture_size] {
    cv::Mat response;
    cv::preCornerDetect(impl_->value, response,
                        static_cast<int>(aperture_size),
                        cv::BORDER_REFLECT_101);
    return copy_finite_f32_values(response, impl_->value.rows,
                                  impl_->value.cols);
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::corner_eigensystem_owned(
    std::int64_t block_size, std::int64_t aperture_size) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_corner_response(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          block_size, aperture_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, block_size, aperture_size] {
        cv::Mat response;
        cv::cornerEigenValsAndVecs(impl_->value, response,
                                   static_cast<int>(block_size),
                                   static_cast<int>(aperture_size),
                                   cv::BORDER_REFLECT_101);
        auto values = copy_finite_f32_channels(
            response, impl_->value.rows, impl_->value.cols, 6);
        if (values == nullptr) {
          return std::unique_ptr<std::vector<double>>{};
        }
        for (std::size_t index = 0; index < values->size(); index += 6) {
          double& maximum = (*values)[index];
          double& minimum = (*values)[index + 1];
          if (maximum < -1.0e-6 || minimum < -1.0e-6 ||
              maximum + 1.0e-6 < minimum) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<std::vector<double>>{};
          }
          maximum = maximum < 0.0 ? 0.0 : maximum;
          minimum = minimum < 0.0 ? 0.0 : minimum;
        }
        return values;
      });
}

}  // namespace terlan_opencv
