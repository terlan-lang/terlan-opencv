#include "opencv_adapter.hpp"
#include "opencv_feature_internal.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <utility>

namespace terlan_opencv {
namespace {

constexpr std::int64_t kMaximumBinaryFeatures = 4096;
constexpr int kOrbDescriptorWidth = 32;
constexpr int kAkazeDescriptorWidth = 61;
constexpr int kAkazeOctaves = 4;
constexpr int kAkazeOctaveLayers = 4;
std::atomic<std::int64_t> live_orb_features{0};
std::atomic<std::int64_t> live_orb_matches{0};
std::atomic<std::int64_t> live_brisk_features{0};
std::atomic<std::int64_t> live_brisk_matches{0};
std::atomic<std::int64_t> live_akaze_features{0};
std::atomic<std::int64_t> live_akaze_matches{0};

using BinaryFeatureData = feature::Data;
using BinaryMatch = feature::Match;

bool valid_detected_keypoint(const cv::KeyPoint& keypoint, int rows,
                             int cols, int maximum_octaves) noexcept {
  return feature::valid_keypoint(keypoint, rows, cols, maximum_octaves, true);
}

bool keypoint_less(const cv::KeyPoint& left,
                   const cv::KeyPoint& right) noexcept {
  return feature::keypoint_less(left, right);
}

bool canonicalize_binary(std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors, std::int64_t maximum, int rows,
                         int cols, int descriptor_width, int maximum_octaves) {
  return feature::canonicalize(keypoints, descriptors, maximum, rows, cols,
                               descriptor_width, CV_8UC1, maximum_octaves,
                               false);
}

std::unique_ptr<std::vector<std::uint8_t>> copy_descriptors(
    const BinaryFeatureData& data) {
  return feature::copy_u8_descriptors(data);
}

bool match_binary_features(const BinaryFeatureData& query,
                           const BinaryFeatureData& train,
                           std::int64_t max_matches,
                           std::vector<BinaryMatch>& result) {
  return feature::match(query, train, max_matches, cv::NORM_HAMMING,
                        static_cast<double>(query.descriptor_width * 8), true,
                        result);
}

}  // namespace

struct OpenCvOrbFeatures::Impl final {
  /// Stores canonical ORB keypoints and their row-aligned descriptors.
  Impl(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors)
      : data(std::move(keypoints), std::move(descriptors), kOrbDescriptorWidth,
             CV_8UC1) {}

  BinaryFeatureData data;
};

struct OpenCvBriskFeatures::Impl final {
  /// Stores canonical BRISK keypoints and their row-aligned descriptors.
  Impl(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors,
       int descriptor_width)
      : data(std::move(keypoints), std::move(descriptors), descriptor_width,
             CV_8UC1) {}

  BinaryFeatureData data;
};

struct OpenCvAkazeFeatures::Impl final {
  /// Stores canonical AKAZE keypoints and their row-aligned descriptors.
  Impl(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors)
      : data(std::move(keypoints), std::move(descriptors),
             kAkazeDescriptorWidth, CV_8UC1) {}

  BinaryFeatureData data;
};

struct OpenCvOrbMatches::Impl final {
  /// Stores canonical copied match records.
  explicit Impl(std::vector<BinaryMatch> matches)
      : matches(std::move(matches)) {}

  std::vector<BinaryMatch> matches;
};

struct OpenCvBriskMatches::Impl final {
  /// Stores canonical copied match records.
  explicit Impl(std::vector<BinaryMatch> matches)
      : matches(std::move(matches)) {}

  std::vector<BinaryMatch> matches;
};

struct OpenCvAkazeMatches::Impl final {
  /// Stores canonical copied match records.
  explicit Impl(std::vector<BinaryMatch> matches)
      : matches(std::move(matches)) {}

  std::vector<BinaryMatch> matches;
};

std::unique_ptr<OpenCvOrbFeatures> OpenCvMat::orb_features_owned(
    std::int64_t max_features) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      max_features < 1 || max_features > kMaximumBinaryFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvOrbFeatures>([this, max_features] {
    auto detector = cv::ORB::create(static_cast<int>(max_features), 1.2F, 8,
                                    31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    if (detector.empty() || detector->descriptorSize() != kOrbDescriptorWidth ||
        detector->descriptorType() != CV_8U) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvOrbFeatures>{};
    }
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(impl_->value, cv::noArray(), keypoints,
                               descriptors, false);
    if (!canonicalize_binary(keypoints, descriptors, max_features,
                             impl_->value.rows, impl_->value.cols,
                             kOrbDescriptorWidth, 8)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvOrbFeatures>{};
    }
    return std::make_unique<OpenCvOrbFeatures>(
        std::make_unique<OpenCvOrbFeatures::Impl>(
            std::move(keypoints), std::move(descriptors)));
  });
}

std::unique_ptr<OpenCvBriskFeatures> OpenCvMat::brisk_features_owned(
    std::int64_t threshold, std::int64_t octaves,
    std::int64_t max_features) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      threshold < 1 || threshold > 255 || octaves < 0 || octaves > 8 ||
      max_features < 1 || max_features > kMaximumBinaryFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvBriskFeatures>(
      [this, threshold, octaves, max_features] {
        auto detector = cv::BRISK::create(static_cast<int>(threshold),
                                          static_cast<int>(octaves), 1.0F);
        const int descriptor_width = detector.empty() ? 0 : detector->descriptorSize();
        if (detector.empty() || descriptor_width < 1 || descriptor_width > 256 ||
            detector->descriptorType() != CV_8U ||
            detector->defaultNorm() != cv::NORM_HAMMING) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvBriskFeatures>{};
        }
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(impl_->value, keypoints, cv::noArray());
        const int maximum_octaves = static_cast<int>(octaves) + 1;
        for (const auto& keypoint : keypoints) {
          if (!valid_detected_keypoint(keypoint, impl_->value.rows,
                                       impl_->value.cols, maximum_octaves)) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<OpenCvBriskFeatures>{};
          }
        }
        std::stable_sort(keypoints.begin(), keypoints.end(),
                         [](const cv::KeyPoint& left,
                            const cv::KeyPoint& right) {
                           if (left.response != right.response) {
                             return left.response > right.response;
                           }
                           return keypoint_less(left, right);
                         });
        if (keypoints.size() > static_cast<std::size_t>(max_features)) {
          keypoints.resize(static_cast<std::size_t>(max_features));
        }
        cv::Mat descriptors;
        detector->compute(impl_->value, keypoints, descriptors);
        if (!canonicalize_binary(keypoints, descriptors, max_features,
                                 impl_->value.rows, impl_->value.cols,
                                 descriptor_width, maximum_octaves)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvBriskFeatures>{};
        }
        return std::make_unique<OpenCvBriskFeatures>(
            std::make_unique<OpenCvBriskFeatures::Impl>(
                std::move(keypoints), std::move(descriptors), descriptor_width));
      });
}

std::unique_ptr<OpenCvAkazeFeatures> OpenCvMat::akaze_features_owned(
    double threshold, std::int64_t max_features) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      !std::isfinite(threshold) || threshold <= 0.0 || threshold > 1.0 ||
      max_features < 1 || max_features > kMaximumBinaryFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvAkazeFeatures>([this, threshold, max_features] {
    auto detector = cv::AKAZE::create(
        cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, static_cast<float>(threshold),
        kAkazeOctaves, kAkazeOctaveLayers, cv::KAZE::DIFF_PM_G2,
        static_cast<int>(max_features));
    if (detector.empty() ||
        detector->getDescriptorType() != cv::AKAZE::DESCRIPTOR_MLDB ||
        detector->getDescriptorSize() != 0 ||
        detector->getDescriptorChannels() != 3 ||
        detector->getThreshold() != static_cast<float>(threshold) ||
        detector->getNOctaves() != kAkazeOctaves ||
        detector->getNOctaveLayers() != kAkazeOctaveLayers ||
        detector->getDiffusivity() != cv::KAZE::DIFF_PM_G2 ||
        detector->getMaxPoints() != static_cast<int>(max_features) ||
        detector->descriptorSize() != kAkazeDescriptorWidth ||
        detector->descriptorType() != CV_8U ||
        detector->defaultNorm() != cv::NORM_HAMMING) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvAkazeFeatures>{};
    }
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(impl_->value, cv::noArray(), keypoints,
                               descriptors, false);
    if (!canonicalize_binary(keypoints, descriptors, max_features,
                             impl_->value.rows, impl_->value.cols,
                             kAkazeDescriptorWidth,
                             kAkazeOctaves * kAkazeOctaveLayers)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvAkazeFeatures>{};
    }
    return std::make_unique<OpenCvAkazeFeatures>(
        std::make_unique<OpenCvAkazeFeatures::Impl>(
            std::move(keypoints), std::move(descriptors)));
  });
}

OpenCvOrbFeatures::OpenCvOrbFeatures(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_orb_features.fetch_add(1, std::memory_order_relaxed);
}

OpenCvOrbFeatures::~OpenCvOrbFeatures() noexcept {
  live_orb_features.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvOrbFeatures::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->data.keypoints.size());
}

std::unique_ptr<std::vector<double>> OpenCvOrbFeatures::keypoints_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_keypoints(impl_->data); });
}

std::int64_t OpenCvOrbFeatures::descriptor_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.descriptor_width;
}

std::unique_ptr<std::vector<std::uint8_t>>
OpenCvOrbFeatures::descriptors_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::uint8_t>>(
      [this] { return copy_descriptors(impl_->data); });
}

std::unique_ptr<OpenCvOrbMatches> OpenCvOrbFeatures::match_owned(
    const OpenCvOrbFeatures& other, std::int64_t max_matches) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || max_matches < 1 ||
      max_matches > kMaximumBinaryFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvOrbMatches>([this, &other, max_matches] {
    std::vector<BinaryMatch> result;
    if (!match_binary_features(impl_->data, other.impl_->data, max_matches,
                               result)) {
      return std::unique_ptr<OpenCvOrbMatches>{};
    }
    return std::make_unique<OpenCvOrbMatches>(
        std::make_unique<OpenCvOrbMatches::Impl>(std::move(result)));
  });
}

std::int64_t live_orb_feature_count() noexcept {
  return live_orb_features.load(std::memory_order_relaxed);
}

OpenCvOrbMatches::OpenCvOrbMatches(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_orb_matches.fetch_add(1, std::memory_order_relaxed);
}

OpenCvOrbMatches::~OpenCvOrbMatches() noexcept {
  live_orb_matches.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvOrbMatches::count() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<std::int64_t>(impl_->matches.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvOrbMatches::indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_match_indices(impl_->matches); });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvOrbMatches::distances_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_integral_match_distances(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvOrbMatches::coordinates_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_coordinates(impl_->matches); });
}

std::int64_t live_orb_match_count() noexcept {
  return live_orb_matches.load(std::memory_order_relaxed);
}

OpenCvBriskFeatures::OpenCvBriskFeatures(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_brisk_features.fetch_add(1, std::memory_order_relaxed);
}

OpenCvBriskFeatures::~OpenCvBriskFeatures() noexcept {
  live_brisk_features.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvBriskFeatures::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->data.keypoints.size());
}

std::unique_ptr<std::vector<double>> OpenCvBriskFeatures::keypoints_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_keypoints(impl_->data); });
}

std::int64_t OpenCvBriskFeatures::descriptor_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.descriptor_width;
}

std::unique_ptr<std::vector<std::uint8_t>>
OpenCvBriskFeatures::descriptors_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::uint8_t>>(
      [this] { return copy_descriptors(impl_->data); });
}

std::unique_ptr<OpenCvBriskMatches> OpenCvBriskFeatures::match_owned(
    const OpenCvBriskFeatures& other,
    std::int64_t max_matches) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || max_matches < 1 ||
      max_matches > kMaximumBinaryFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvBriskMatches>([this, &other, max_matches] {
    std::vector<BinaryMatch> result;
    if (!match_binary_features(impl_->data, other.impl_->data, max_matches,
                               result)) {
      return std::unique_ptr<OpenCvBriskMatches>{};
    }
    return std::make_unique<OpenCvBriskMatches>(
        std::make_unique<OpenCvBriskMatches::Impl>(std::move(result)));
  });
}

std::int64_t live_brisk_feature_count() noexcept {
  return live_brisk_features.load(std::memory_order_relaxed);
}

OpenCvBriskMatches::OpenCvBriskMatches(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_brisk_matches.fetch_add(1, std::memory_order_relaxed);
}

OpenCvBriskMatches::~OpenCvBriskMatches() noexcept {
  live_brisk_matches.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvBriskMatches::count() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<std::int64_t>(impl_->matches.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvBriskMatches::indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_match_indices(impl_->matches); });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvBriskMatches::distances_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_integral_match_distances(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvBriskMatches::coordinates_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_coordinates(impl_->matches); });
}

std::int64_t live_brisk_match_count() noexcept {
  return live_brisk_matches.load(std::memory_order_relaxed);
}

OpenCvAkazeFeatures::OpenCvAkazeFeatures(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_akaze_features.fetch_add(1, std::memory_order_relaxed);
}

OpenCvAkazeFeatures::~OpenCvAkazeFeatures() noexcept {
  live_akaze_features.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvAkazeFeatures::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->data.keypoints.size());
}

std::unique_ptr<std::vector<double>> OpenCvAkazeFeatures::keypoints_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_keypoints(impl_->data); });
}

std::int64_t OpenCvAkazeFeatures::descriptor_width() const noexcept {
  return impl_ == nullptr ? 0 : impl_->data.descriptor_width;
}

std::unique_ptr<std::vector<std::uint8_t>>
OpenCvAkazeFeatures::descriptors_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::uint8_t>>(
      [this] { return copy_descriptors(impl_->data); });
}

std::unique_ptr<OpenCvAkazeMatches> OpenCvAkazeFeatures::match_owned(
    const OpenCvAkazeFeatures& other,
    std::int64_t max_matches) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr || max_matches < 1 ||
      max_matches > kMaximumBinaryFeatures) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvAkazeMatches>([this, &other, max_matches] {
    std::vector<BinaryMatch> result;
    if (!match_binary_features(impl_->data, other.impl_->data, max_matches,
                               result)) {
      return std::unique_ptr<OpenCvAkazeMatches>{};
    }
    return std::make_unique<OpenCvAkazeMatches>(
        std::make_unique<OpenCvAkazeMatches::Impl>(std::move(result)));
  });
}

std::int64_t live_akaze_feature_count() noexcept {
  return live_akaze_features.load(std::memory_order_relaxed);
}

OpenCvAkazeMatches::OpenCvAkazeMatches(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_akaze_matches.fetch_add(1, std::memory_order_relaxed);
}

OpenCvAkazeMatches::~OpenCvAkazeMatches() noexcept {
  live_akaze_matches.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvAkazeMatches::count() const noexcept {
  return impl_ == nullptr ? 0 : static_cast<std::int64_t>(impl_->matches.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvAkazeMatches::indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_match_indices(impl_->matches); });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvAkazeMatches::distances_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return feature::copy_integral_match_distances(impl_->matches); });
}

std::unique_ptr<std::vector<double>> OpenCvAkazeMatches::coordinates_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return feature::copy_match_coordinates(impl_->matches); });
}

std::int64_t live_akaze_match_count() noexcept {
  return live_akaze_matches.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
