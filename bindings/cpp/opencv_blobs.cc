#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>

namespace terlan_opencv {
namespace {

constexpr float kBlobMinimumThreshold = 0.0F;
constexpr float kBlobMaximumThreshold = 256.0F;
constexpr float kBlobThresholdStep = 10.0F;
constexpr std::size_t kBlobMinimumRepeatability = 2;

/// Orders stronger blob candidates before deterministic truncation.
bool stronger_blob(const cv::KeyPoint& left,
                   const cv::KeyPoint& right) noexcept {
  if (left.response != right.response) {
    return left.response > right.response;
  }
  if (left.size != right.size) {
    return left.size > right.size;
  }
  if (left.pt.y != right.pt.y) {
    return left.pt.y < right.pt.y;
  }
  return left.pt.x < right.pt.x;
}

/// Orders retained blobs canonically in row-major center order.
bool earlier_blob(const cv::KeyPoint& left,
                  const cv::KeyPoint& right) noexcept {
  if (left.pt.y != right.pt.y) {
    return left.pt.y < right.pt.y;
  }
  if (left.pt.x != right.pt.x) {
    return left.pt.x < right.pt.x;
  }
  if (left.size != right.size) {
    return left.size > right.size;
  }
  return left.response > right.response;
}

/// Checks that OpenCV retained the package-owned blob policy exactly.
bool valid_blob_policy(const cv::SimpleBlobDetector::Params& params,
                       std::int64_t blob_value,
                       std::int64_t minimum_area,
                       std::int64_t maximum_area,
                       double minimum_distance) noexcept {
  return params.minThreshold == kBlobMinimumThreshold &&
         params.maxThreshold == kBlobMaximumThreshold &&
         params.thresholdStep == kBlobThresholdStep &&
         params.minRepeatability == kBlobMinimumRepeatability &&
         params.minDistBetweenBlobs == static_cast<float>(minimum_distance) &&
         params.filterByColor && params.blobColor == blob_value &&
         params.filterByArea &&
         params.minArea == static_cast<float>(minimum_area) &&
         params.maxArea == static_cast<float>(maximum_area) &&
         !params.filterByCircularity && !params.filterByInertia &&
         !params.filterByConvexity && !params.collectContours;
}

}  // namespace

std::unique_ptr<std::vector<double>> OpenCvMat::simple_blob_keypoints_owned(
    std::int64_t blob_value, std::int64_t minimum_area,
    std::int64_t maximum_area, double minimum_distance,
    std::int64_t max_blobs) const noexcept {
  failure::clear();
  const std::int64_t image_area = impl_ == nullptr
                                      ? 0
                                      : static_cast<std::int64_t>(
                                            impl_->value.rows) *
                                            impl_->value.cols;
  const double diagonal = impl_ == nullptr
                              ? 0.0
                              : std::hypot(impl_->value.rows,
                                           impl_->value.cols);
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      impl_->value.rows < 3 || impl_->value.cols < 3 ||
      (blob_value != 0 && blob_value != 255) || minimum_area < 1 ||
      maximum_area <= minimum_area || maximum_area > image_area + 1 ||
      !std::isfinite(minimum_distance) || minimum_distance <= 0.0 ||
      minimum_distance > diagonal || max_blobs < 1 ||
      max_blobs > validation::kMaxDetectedCorners) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, blob_value, minimum_area, maximum_area, minimum_distance,
       max_blobs] {
        cv::SimpleBlobDetector::Params params;
        params.minThreshold = kBlobMinimumThreshold;
        params.maxThreshold = kBlobMaximumThreshold;
        params.thresholdStep = kBlobThresholdStep;
        params.minRepeatability = kBlobMinimumRepeatability;
        params.minDistBetweenBlobs = static_cast<float>(minimum_distance);
        params.filterByColor = true;
        params.blobColor = static_cast<unsigned char>(blob_value);
        params.filterByArea = true;
        params.minArea = static_cast<float>(minimum_area);
        params.maxArea = static_cast<float>(maximum_area);
        params.filterByCircularity = false;
        params.filterByInertia = false;
        params.filterByConvexity = false;
        params.collectContours = false;
        auto detector = cv::SimpleBlobDetector::create(params);
        if (detector.empty() || !valid_blob_policy(
                                    detector->getParams(), blob_value,
                                    minimum_area, maximum_area,
                                    minimum_distance)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<double>>{};
        }
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(impl_->value, keypoints, cv::noArray());
        const float maximum_size = static_cast<float>(
            2.0 * std::hypot(impl_->value.rows, impl_->value.cols));
        for (const auto& keypoint : keypoints) {
          if (!std::isfinite(keypoint.pt.x) ||
              !std::isfinite(keypoint.pt.y) ||
              !std::isfinite(keypoint.size) ||
              !std::isfinite(keypoint.response) || keypoint.pt.x < 0.0F ||
              keypoint.pt.y < 0.0F ||
              keypoint.pt.x >= static_cast<float>(impl_->value.cols) ||
              keypoint.pt.y >= static_cast<float>(impl_->value.rows) ||
              keypoint.size <= 0.0F || keypoint.size > maximum_size ||
              keypoint.response < 0.0F) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<std::vector<double>>{};
          }
        }
        std::stable_sort(keypoints.begin(), keypoints.end(), stronger_blob);
        if (keypoints.size() > static_cast<std::size_t>(max_blobs)) {
          keypoints.resize(static_cast<std::size_t>(max_blobs));
        }
        std::sort(keypoints.begin(), keypoints.end(), earlier_blob);
        auto result = std::make_unique<std::vector<double>>();
        result->reserve(keypoints.size() * 4);
        for (std::size_t index = 0; index < keypoints.size(); ++index) {
          if (index > 0 && keypoints[index - 1].pt == keypoints[index].pt) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<std::vector<double>>{};
          }
          result->push_back(keypoints[index].pt.x);
          result->push_back(keypoints[index].pt.y);
          result->push_back(keypoints[index].size);
          result->push_back(keypoints[index].response);
        }
        return result;
      });
}

}  // namespace terlan_opencv
