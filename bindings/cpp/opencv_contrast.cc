#include "opencv_adapter.hpp"
#include "opencv_contrast_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <utility>

namespace terlan_opencv {

std::unique_ptr<OpenCvMat> OpenCvMat::equalize_hist_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_grayscale_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    cv::Mat result;
    cv::equalizeHist(impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::histogram_u8_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_histogram_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    constexpr int kChannel = 0;
    constexpr int kBinCount = 256;
    constexpr float kRange[] = {0.0F, 256.0F};
    const float* ranges[] = {kRange};
    cv::Mat histogram;
    cv::calcHist(&impl_->value, 1, &kChannel, cv::Mat(), histogram, 1,
                 &kBinCount, ranges, true, false);
    if (histogram.type() != CV_32FC1 || histogram.total() != kBinCount) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto bins = std::make_unique<std::vector<std::int64_t>>();
    bins->reserve(kBinCount);
    std::int64_t total = 0;
    for (int index = 0; index < kBinCount; ++index) {
      const float count = histogram.at<float>(index);
      const auto rounded = static_cast<std::int64_t>(std::llround(count));
      if (!std::isfinite(count) || count < 0.0F ||
          static_cast<float>(rounded) != count) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      bins->push_back(rounded);
      total += rounded;
    }
    const auto expected = static_cast<std::int64_t>(impl_->value.rows) *
                          static_cast<std::int64_t>(impl_->value.cols);
    if (total != expected) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    return bins;
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::adaptive_equalize_hist_owned(
    double clip_limit, std::int64_t tiles_x,
    std::int64_t tiles_y) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_adaptive_equalization(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          clip_limit, tiles_x, tiles_y)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, clip_limit, tiles_x, tiles_y] {
    cv::Mat result;
    const auto equalizer = cv::createCLAHE(
        clip_limit,
        cv::Size(static_cast<int>(tiles_x), static_cast<int>(tiles_y)));
    equalizer->apply(impl_->value, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

}  // namespace terlan_opencv
