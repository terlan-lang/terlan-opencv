#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>

namespace terlan_opencv {
namespace {

/// Copies exact nonnegative channel sums after validating their finite bounds.
std::unique_ptr<std::vector<std::int64_t>> copy_channel_sums(
    const cv::Scalar& values, int count, std::int64_t maximum) {
  auto result = std::make_unique<std::vector<std::int64_t>>();
  result->reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const double value = values[index];
    if (!std::isfinite(value) || value < 0.0 || value > maximum ||
        std::trunc(value) != value) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    result->push_back(static_cast<std::int64_t>(value));
  }
  return result;
}

/// Checks that two matrices are equal-size unsigned-byte grayscale channels.
bool compatible_grayscale_channels(const cv::Mat& first,
                                   const cv::Mat& second) noexcept {
  return first.type() == CV_8UC1 && second.type() == CV_8UC1 &&
         validation::valid_image_pair(
             first.rows, first.cols, first.channels(), second.rows,
             second.cols, second.channels());
}

}  // namespace

std::unique_ptr<std::vector<double>> OpenCvMat::mean_channels_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_u8_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    return copy_u8_domain_scalar_values(cv::mean(impl_->value),
                                        impl_->value.channels());
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::stddev_channels_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_u8_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    cv::Scalar means;
    cv::Scalar standard_deviations;
    cv::meanStdDev(impl_->value, means, standard_deviations);
    return copy_u8_domain_scalar_values(standard_deviations,
                                        impl_->value.channels());
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::min_max_values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_grayscale_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(impl_->value, &minimum, &maximum);
    if (minimum > maximum) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    return copy_u8_domain_scalar_values(cv::Scalar(minimum, maximum), 2);
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::integral_values_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_integral_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    cv::Mat integral;
    cv::integral(impl_->value, integral, CV_32S);
    if (integral.rows != impl_->value.rows + 1 ||
        integral.cols != impl_->value.cols + 1 ||
        integral.type() != CV_32SC1) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }

    auto values = std::make_unique<std::vector<std::int64_t>>();
    values->reserve(integral.total());
    const std::int64_t maximum =
        static_cast<std::int64_t>(impl_->value.rows) * impl_->value.cols * 255;
    for (int row = 0; row < integral.rows; ++row) {
      const auto* row_values = integral.ptr<int>(row);
      for (int col = 0; col < integral.cols; ++col) {
        if (row_values[col] < 0 || row_values[col] > maximum) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<std::int64_t>>{};
        }
        values->push_back(static_cast<std::int64_t>(row_values[col]));
      }
    }
    return values;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::non_zero_counts_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_u8_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto counts = std::make_unique<std::vector<std::int64_t>>();
    counts->reserve(static_cast<std::size_t>(impl_->value.channels()));
    const std::int64_t pixel_count =
        static_cast<std::int64_t>(impl_->value.rows) * impl_->value.cols;
    for (int index = 0; index < impl_->value.channels(); ++index) {
      cv::Mat channel;
      cv::extractChannel(impl_->value, channel, index);
      const int count = cv::countNonZero(channel);
      if (count < 0 || count > pixel_count) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      counts->push_back(count);
    }
    return counts;
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::extract_channel_owned(
    std::int64_t channel_index) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_channel_index(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          channel_index)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, channel_index] {
    cv::Mat result;
    cv::extractChannel(impl_->value, result, static_cast<int>(channel_index));
    return own_grayscale_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::merge_bgr_owned(
    const OpenCvMat& green, const OpenCvMat& red) const noexcept {
  failure::clear();
  const auto shape =
      impl_ == nullptr
          ? validation::ImageShape{false, 0, 0}
          : validation::merged_image_shape(impl_->value.rows,
                                           impl_->value.cols, 3);
  if (!shape.valid || green.impl_ == nullptr || red.impl_ == nullptr ||
      !compatible_grayscale_channels(impl_->value, green.impl_->value) ||
      !compatible_grayscale_channels(impl_->value, red.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &green, &red] {
    const std::array<cv::Mat, 3> channels = {
        impl_->value, green.impl_->value, red.impl_->value};
    cv::Mat result;
    cv::merge(channels.data(), channels.size(), result);
    return own_channel_count_mat(std::move(result), impl_->value, 3);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::merge_bgra_owned(
    const OpenCvMat& green, const OpenCvMat& red,
    const OpenCvMat& alpha) const noexcept {
  failure::clear();
  const auto shape =
      impl_ == nullptr
          ? validation::ImageShape{false, 0, 0}
          : validation::merged_image_shape(impl_->value.rows,
                                           impl_->value.cols, 4);
  if (!shape.valid || green.impl_ == nullptr || red.impl_ == nullptr ||
      alpha.impl_ == nullptr ||
      !compatible_grayscale_channels(impl_->value, green.impl_->value) ||
      !compatible_grayscale_channels(impl_->value, red.impl_->value) ||
      !compatible_grayscale_channels(impl_->value, alpha.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &green, &red, &alpha] {
    const std::array<cv::Mat, 4> channels = {
        impl_->value, green.impl_->value, red.impl_->value, alpha.impl_->value};
    cv::Mat result;
    cv::merge(channels.data(), channels.size(), result);
    return own_channel_count_mat(std::move(result), impl_->value, 4);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::replace_channel_owned(
    const OpenCvMat& replacement,
    std::int64_t channel_index) const noexcept {
  failure::clear();
  if (impl_ == nullptr || replacement.impl_ == nullptr ||
      impl_->value.depth() != CV_8U ||
      replacement.impl_->value.type() != CV_8UC1 ||
      !validation::valid_channel_replacement(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          replacement.impl_->value.rows, replacement.impl_->value.cols,
          replacement.impl_->value.channels(), channel_index)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &replacement, channel_index] {
    cv::Mat result = impl_->value.clone();
    cv::insertChannel(replacement.impl_->value, result,
                      static_cast<int>(channel_index));
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::sum_channels_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_u8_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    const std::int64_t maximum =
        static_cast<std::int64_t>(impl_->value.rows) * impl_->value.cols * 255;
    return copy_channel_sums(cv::sum(impl_->value), impl_->value.channels(),
                             maximum);
  });
}

}  // namespace terlan_opencv
