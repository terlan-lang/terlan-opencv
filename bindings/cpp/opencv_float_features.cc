#include "opencv_adapter.hpp"
#include "opencv_feature_internal.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <utility>

namespace terlan_opencv {
namespace {

constexpr std::int64_t kMaximumFloatFeatures = 4096;
constexpr int kSiftDescriptorWidth = 128;
constexpr int kSiftOctaveLayers = 3;
constexpr double kSiftEdgeThreshold = 10.0;
constexpr double kSiftSigma = 1.6;
constexpr int kAffineSiftMaximumTilt = 2;
constexpr int kAffineSiftMinimumTilt = 0;
constexpr float kAffineSiftTiltStep = 1.4142135623730951F;
constexpr float kAffineSiftRotateStepBase = 72.0F;
constexpr int kKazeDescriptorWidth = 64;
constexpr int kKazeOctaves = 4;
constexpr int kKazeOctaveLayers = 4;
std::atomic<std::int64_t> live_sift_features{0};
std::atomic<std::int64_t> live_sift_matches{0};
std::atomic<std::int64_t> live_kaze_features{0};
std::atomic<std::int64_t> live_kaze_matches{0};

}  // namespace

struct OpenCvSiftFeatures::Impl final {
  /// Stores canonical SIFT keypoints and row-aligned descriptors.
  Impl(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors)
      : data(std::move(keypoints), std::move(descriptors),
             kSiftDescriptorWidth, CV_32FC1) {}

  feature::Data data;
};

struct OpenCvSiftMatches::Impl final {
  /// Stores canonical copied SIFT match records.
  explicit Impl(std::vector<feature::Match> matches)
      : matches(std::move(matches)) {}

  std::vector<feature::Match> matches;
};

struct OpenCvKazeFeatures::Impl final {
  /// Stores canonical KAZE keypoints and row-aligned descriptors.
  Impl(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors)
      : data(std::move(keypoints), std::move(descriptors),
             kKazeDescriptorWidth, CV_32FC1) {}

  feature::Data data;
};

struct OpenCvKazeMatches::Impl final {
  /// Stores canonical copied KAZE match records.
  explicit Impl(std::vector<feature::Match> matches)
      : matches(std::move(matches)) {}

  std::vector<feature::Match> matches;
};

std::unique_ptr<OpenCvSiftFeatures> OpenCvMat::sift_features_owned(
    double contrast_threshold, std::int64_t max_features) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      !std::isfinite(contrast_threshold) || contrast_threshold <= 0.0 ||
      contrast_threshold > 1.0 || max_features < 1 ||
      max_features > kMaximumFloatFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvSiftFeatures>(
      [this, contrast_threshold, max_features] {
        auto detector = cv::SIFT::create(
            static_cast<int>(max_features), kSiftOctaveLayers,
            contrast_threshold, kSiftEdgeThreshold, kSiftSigma, CV_32F, false);
        if (detector.empty() ||
            detector->getNFeatures() != static_cast<int>(max_features) ||
            detector->getNOctaveLayers() != kSiftOctaveLayers ||
            detector->getContrastThreshold() != contrast_threshold ||
            detector->getEdgeThreshold() != kSiftEdgeThreshold ||
            detector->getSigma() != kSiftSigma ||
            detector->descriptorSize() != kSiftDescriptorWidth ||
            detector->descriptorType() != CV_32F ||
            detector->defaultNorm() != cv::NORM_L2) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSiftFeatures>{};
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detectAndCompute(impl_->value, cv::noArray(), keypoints,
                                   descriptors, false);
        if (!feature::canonicalize(
                keypoints, descriptors, max_features, impl_->value.rows,
                impl_->value.cols, kSiftDescriptorWidth, CV_32FC1, -1, false)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSiftFeatures>{};
        }
        return std::make_unique<OpenCvSiftFeatures>(
            std::make_unique<OpenCvSiftFeatures::Impl>(
                std::move(keypoints), std::move(descriptors)));
      });
}

std::unique_ptr<OpenCvSiftFeatures> OpenCvMat::affine_sift_features_owned(
    double contrast_threshold, std::int64_t max_features) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      !std::isfinite(contrast_threshold) || contrast_threshold <= 0.0 ||
      contrast_threshold > 1.0 || max_features < 1 ||
      max_features > kMaximumFloatFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvSiftFeatures>(
      [this, contrast_threshold, max_features] {
        auto backend = cv::SIFT::create(
            static_cast<int>(max_features), kSiftOctaveLayers,
            contrast_threshold, kSiftEdgeThreshold, kSiftSigma, CV_32F, false);
        auto detector = cv::AffineFeature::create(
            backend, kAffineSiftMaximumTilt, kAffineSiftMinimumTilt,
            kAffineSiftTiltStep, kAffineSiftRotateStepBase);
        if (backend.empty() || detector.empty() ||
            backend->getNFeatures() != static_cast<int>(max_features) ||
            backend->getNOctaveLayers() != kSiftOctaveLayers ||
            backend->getContrastThreshold() != contrast_threshold ||
            backend->getEdgeThreshold() != kSiftEdgeThreshold ||
            backend->getSigma() != kSiftSigma ||
            detector->descriptorSize() != kSiftDescriptorWidth ||
            detector->descriptorType() != CV_32F ||
            detector->defaultNorm() != cv::NORM_L2) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSiftFeatures>{};
        }
        std::vector<float> tilts;
        std::vector<float> rolls;
        detector->getViewParams(tilts, rolls);
        if (tilts.empty() || tilts.size() != rolls.size() ||
            tilts.size() > 256) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSiftFeatures>{};
        }
        for (std::size_t index = 0; index < tilts.size(); ++index) {
          if (!std::isfinite(tilts[index]) || !std::isfinite(rolls[index]) ||
              tilts[index] < 1.0F || rolls[index] < 0.0F ||
              rolls[index] >= 360.0F) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<OpenCvSiftFeatures>{};
          }
        }
        detector->setViewParams(tilts, rolls);
        std::vector<float> configured_tilts;
        std::vector<float> configured_rolls;
        detector->getViewParams(configured_tilts, configured_rolls);
        if (configured_tilts != tilts || configured_rolls != rolls) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSiftFeatures>{};
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector->detectAndCompute(impl_->value, cv::noArray(), keypoints,
                                   descriptors, false);
        if (!feature::truncate_by_response(keypoints, descriptors,
                                           max_features) ||
            !feature::canonicalize(
                keypoints, descriptors, max_features, impl_->value.rows,
                impl_->value.cols, kSiftDescriptorWidth, CV_32FC1, -1, false)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSiftFeatures>{};
        }
        return std::make_unique<OpenCvSiftFeatures>(
            std::make_unique<OpenCvSiftFeatures::Impl>(
                std::move(keypoints), std::move(descriptors)));
      });
}

OpenCvSiftFeatures::OpenCvSiftFeatures(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_sift_features.fetch_add(1, std::memory_order_relaxed);
}

OpenCvSiftFeatures::~OpenCvSiftFeatures() noexcept {
  live_sift_features.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvSiftFeatures::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->data.keypoints.size());
}

std::unique_ptr<std::vector<double>> OpenCvSiftFeatures::keypoints_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_keypoints(impl_->data); });
}

std::int64_t OpenCvSiftFeatures::descriptor_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.descriptor_width;
}

std::unique_ptr<std::vector<double>> OpenCvSiftFeatures::descriptors_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_f32_descriptors(impl_->data); });
}

std::unique_ptr<OpenCvSiftMatches> OpenCvSiftFeatures::match_owned(
    const OpenCvSiftFeatures& other,
    std::int64_t max_matches) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || max_matches < 1 ||
      max_matches > kMaximumFloatFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvSiftMatches>([this, &other, max_matches] {
    std::vector<feature::Match> result;
    if (!feature::match(impl_->data, other.impl_->data, max_matches,
                        cv::NORM_L2,
                        std::numeric_limits<float>::max(), false, result)) {
      return std::unique_ptr<OpenCvSiftMatches>{};
    }
    return std::make_unique<OpenCvSiftMatches>(
        std::make_unique<OpenCvSiftMatches::Impl>(std::move(result)));
  });
}

std::int64_t live_sift_feature_count() noexcept {
  return live_sift_features.load(std::memory_order_relaxed);
}

OpenCvSiftMatches::OpenCvSiftMatches(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_sift_matches.fetch_add(1, std::memory_order_relaxed);
}

OpenCvSiftMatches::~OpenCvSiftMatches() noexcept {
  live_sift_matches.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvSiftMatches::count() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<std::int64_t>(impl_->matches.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvSiftMatches::indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_match_indices(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvSiftMatches::distances_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_distances(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvSiftMatches::coordinates_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_coordinates(impl_->matches); });
}

std::int64_t live_sift_match_count() noexcept {
  return live_sift_matches.load(std::memory_order_relaxed);
}

std::unique_ptr<OpenCvKazeFeatures> OpenCvMat::kaze_features_owned(
    double threshold, std::int64_t max_features) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      !std::isfinite(threshold) || threshold <= 0.0 || threshold > 1.0 ||
      max_features < 1 || max_features > kMaximumFloatFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvKazeFeatures>(
      [this, threshold, max_features] {
        auto detector = cv::KAZE::create(
            false, false, static_cast<float>(threshold), kKazeOctaves,
            kKazeOctaveLayers, cv::KAZE::DIFF_PM_G2);
        if (detector.empty() || detector->getExtended() ||
            detector->getUpright() ||
            detector->getThreshold() != static_cast<float>(threshold) ||
            detector->getNOctaves() != kKazeOctaves ||
            detector->getNOctaveLayers() != kKazeOctaveLayers ||
            detector->getDiffusivity() != cv::KAZE::DIFF_PM_G2 ||
            detector->descriptorSize() != kKazeDescriptorWidth ||
            detector->descriptorType() != CV_32F ||
            detector->defaultNorm() != cv::NORM_L2) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvKazeFeatures>{};
        }
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(impl_->value, keypoints, cv::noArray());
        for (const auto& keypoint : keypoints) {
          if (!feature::valid_keypoint(keypoint, impl_->value.rows,
                                       impl_->value.cols, kKazeOctaves,
                                       false)) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<OpenCvKazeFeatures>{};
          }
        }
        std::stable_sort(keypoints.begin(), keypoints.end(),
                         [](const cv::KeyPoint& left,
                            const cv::KeyPoint& right) {
                           if (left.response != right.response) {
                             return left.response > right.response;
                           }
                           return feature::keypoint_less(left, right);
                         });
        if (keypoints.size() > static_cast<std::size_t>(max_features)) {
          keypoints.resize(static_cast<std::size_t>(max_features));
        }
        cv::Mat descriptors;
        detector->compute(impl_->value, keypoints, descriptors);
        if (!feature::canonicalize(
                keypoints, descriptors, max_features, impl_->value.rows,
                impl_->value.cols, kKazeDescriptorWidth, CV_32FC1,
                kKazeOctaves, false)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvKazeFeatures>{};
        }
        return std::make_unique<OpenCvKazeFeatures>(
            std::make_unique<OpenCvKazeFeatures::Impl>(
                std::move(keypoints), std::move(descriptors)));
      });
}

OpenCvKazeFeatures::OpenCvKazeFeatures(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_kaze_features.fetch_add(1, std::memory_order_relaxed);
}

OpenCvKazeFeatures::~OpenCvKazeFeatures() noexcept {
  live_kaze_features.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvKazeFeatures::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->data.keypoints.size());
}

std::unique_ptr<std::vector<double>> OpenCvKazeFeatures::keypoints_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_keypoints(impl_->data); });
}

std::int64_t OpenCvKazeFeatures::descriptor_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.descriptor_width;
}

std::unique_ptr<std::vector<double>> OpenCvKazeFeatures::descriptors_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_f32_descriptors(impl_->data); });
}

std::unique_ptr<OpenCvKazeMatches> OpenCvKazeFeatures::match_owned(
    const OpenCvKazeFeatures& other,
    std::int64_t max_matches) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || max_matches < 1 ||
      max_matches > kMaximumFloatFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvKazeMatches>([this, &other, max_matches] {
    std::vector<feature::Match> result;
    if (!feature::match(impl_->data, other.impl_->data, max_matches,
                        cv::NORM_L2, std::numeric_limits<float>::max(), false,
                        result)) {
      return std::unique_ptr<OpenCvKazeMatches>{};
    }
    return std::make_unique<OpenCvKazeMatches>(
        std::make_unique<OpenCvKazeMatches::Impl>(std::move(result)));
  });
}

std::int64_t live_kaze_feature_count() noexcept {
  return live_kaze_features.load(std::memory_order_relaxed);
}

OpenCvKazeMatches::OpenCvKazeMatches(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_kaze_matches.fetch_add(1, std::memory_order_relaxed);
}

OpenCvKazeMatches::~OpenCvKazeMatches() noexcept {
  live_kaze_matches.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvKazeMatches::count() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<std::int64_t>(impl_->matches.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvKazeMatches::indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_match_indices(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvKazeMatches::distances_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_distances(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvKazeMatches::coordinates_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_coordinates(impl_->matches); });
}

std::int64_t live_kaze_match_count() noexcept {
  return live_kaze_matches.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
