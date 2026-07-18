#include "opencv_adapter.hpp"
#include "opencv_dnn_internal.hpp"
#include "opencv_dnn_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cmath>
#include <utility>

namespace terlan_opencv {
namespace {

std::atomic<std::int64_t> live_dnn_blobs{0};

/// Reports whether a fixed blob shape fits package-owned storage policy.
bool valid_blob_shape(std::int64_t width, std::int64_t height,
                      std::int64_t channels) noexcept {
  const auto shape = validation::image_shape(height, width, channels);
  return shape.valid &&
         shape.byte_count <=
             validation::kMaxOwnedImageBytes / sizeof(float);
}

/// Reports whether one image and requested blob shape match a fixed layout.
bool valid_blob_input(const cv::Mat& image, std::int64_t width,
                      std::int64_t height, std::int64_t channels) noexcept {
  const int expected_type = channels == 1 ? CV_8UC1 : CV_8UC3;
  return (channels == 1 || channels == 3) && image.type() == expected_type &&
         validation::image_shape(image.rows, image.cols, channels).valid &&
         valid_blob_shape(width, height, channels);
}

/// Validates and transfers one fixed-shape NCHW blob into package ownership.
std::unique_ptr<OpenCvDnnBlob> own_blob(cv::Mat blob, std::int64_t width,
                                       std::int64_t height,
                                       std::int64_t channels) {
  if (blob.dims != 4 || blob.type() != CV_32FC1 || !blob.isContinuous() ||
      blob.size[0] != 1 || blob.size[1] != static_cast<int>(channels) ||
      blob.size[2] != static_cast<int>(height) ||
      blob.size[3] != static_cast<int>(width)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvDnnBlob>(
      std::make_unique<OpenCvDnnBlob::Impl>(std::move(blob)));
}

/// Creates one validated fixed-policy NCHW blob from an unsigned-byte image.
std::unique_ptr<OpenCvDnnBlob> create_blob(const cv::Mat& image,
                                          std::int64_t width,
                                          std::int64_t height,
                                          std::int64_t channels,
                                          bool swap_red_blue,
                                          bool center_crop) {
  cv::Mat blob = cv::dnn::blobFromImage(
      image, 1.0 / 255.0,
      cv::Size(static_cast<int>(width), static_cast<int>(height)), cv::Scalar(),
      swap_red_blue, center_crop, CV_32F);
  return own_blob(std::move(blob), width, height, channels);
}

/// Creates one maintained parameterized NCHW blob.
std::unique_ptr<OpenCvDnnBlob> create_parameterized_blob(
    const cv::Mat& image, std::int64_t width, std::int64_t height,
    const dnn_validation::DnnNormalization& normalization,
    std::int64_t channels, bool convert_bgr_to_rgb,
    cv::dnn::ImagePaddingMode padding_mode) {
  cv::Mat converted;
  const cv::Mat* source = &image;
  if (convert_bgr_to_rgb) {
    cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
    source = &converted;
  }
  const cv::dnn::Image2BlobParams parameters(
      cv::Scalar(normalization.byte_scales[0], normalization.byte_scales[1],
                 normalization.byte_scales[2]),
      cv::Size(static_cast<int>(width), static_cast<int>(height)),
      cv::Scalar(normalization.byte_means[0], normalization.byte_means[1],
                 normalization.byte_means[2]),
      false, CV_32F, cv::dnn::DNN_LAYOUT_NCHW, padding_mode);
  cv::Mat blob = cv::dnn::blobFromImageWithParams(*source, parameters);
  return own_blob(std::move(blob), width, height, channels);
}

/// Validates and contains one configurable normalized preprocessing call.
std::unique_ptr<OpenCvDnnBlob> checked_normalized_blob(
    const cv::Mat& image, std::int64_t width, std::int64_t height,
    rust::Slice<const double> means, rust::Slice<const double> scales,
    std::int64_t channels, bool convert_bgr_to_rgb,
    cv::dnn::ImagePaddingMode padding_mode) noexcept {
  dnn_validation::DnnNormalization normalization{};
  const bool valid_normalization =
      channels == 3
          ? dnn_validation::rgb_normalization_parameters(
                means.data(), means.size(), scales.data(), scales.size(),
                normalization)
          : dnn_validation::grayscale_normalization_parameters(
                means.data(), means.size(), scales.data(), scales.size(),
                normalization);
  if (!valid_blob_input(image, width, height, channels) ||
      !valid_normalization) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnBlob>(
      [&image, width, height, normalization, channels, convert_bgr_to_rgb,
       padding_mode] {
        return create_parameterized_blob(
            image, width, height, normalization, channels, convert_bgr_to_rgb,
            padding_mode);
      });
}

}  // namespace

OpenCvDnnBlob::OpenCvDnnBlob(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_dnn_blobs.fetch_add(1, std::memory_order_relaxed);
}

OpenCvDnnBlob::~OpenCvDnnBlob() noexcept {
  live_dnn_blobs.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvDnnBlob::batch_count() const noexcept {
  return impl_ == nullptr || impl_->value.dims != 4 ? 0 : impl_->value.size[0];
}

std::int64_t OpenCvDnnBlob::channel_count() const noexcept {
  return impl_ == nullptr || impl_->value.dims != 4 ? 0 : impl_->value.size[1];
}

std::int64_t OpenCvDnnBlob::height() const noexcept {
  return impl_ == nullptr || impl_->value.dims != 4 ? 0 : impl_->value.size[2];
}

std::int64_t OpenCvDnnBlob::width() const noexcept {
  return impl_ == nullptr || impl_->value.dims != 4 ? 0 : impl_->value.size[3];
}

std::unique_ptr<std::vector<double>> OpenCvDnnBlob::values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.dims != 4 ||
      impl_->value.type() != CV_32FC1 || !impl_->value.isContinuous() ||
      batch_count() != 1 ||
      (channel_count() != 1 && channel_count() != 3) || height() <= 0 ||
      width() <= 0 || !valid_blob_shape(width(), height(), channel_count())) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    const std::size_t count = impl_->value.total();
    const float* source = impl_->value.ptr<float>();
    auto values = std::make_unique<std::vector<double>>();
    values->reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      if (!std::isfinite(source[index])) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      values->push_back(source[index]);
    }
    return values;
  });
}

std::unique_ptr<OpenCvDnnBlob> OpenCvMat::blob_from_bgr_rgb_owned(
    std::int64_t width, std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_blob_input(impl_->value, width, height, 3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnBlob>([this, width, height] {
    return create_blob(impl_->value, width, height, 3, true, false);
  });
}

std::unique_ptr<OpenCvDnnBlob> OpenCvMat::blob_from_grayscale_owned(
    std::int64_t width, std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_blob_input(impl_->value, width, height, 1)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnBlob>([this, width, height] {
    return create_blob(impl_->value, width, height, 1, false, false);
  });
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_bgr_rgb_center_crop_owned(
    std::int64_t width, std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_blob_input(impl_->value, width, height, 3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnBlob>([this, width, height] {
    return create_blob(impl_->value, width, height, 3, true, true);
  });
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_grayscale_center_crop_owned(
    std::int64_t width, std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_blob_input(impl_->value, width, height, 1)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnBlob>([this, width, height] {
    return create_blob(impl_->value, width, height, 1, false, true);
  });
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_bgr_rgb_normalized_owned(
    std::int64_t width, std::int64_t height,
    rust::Slice<const double> means,
    rust::Slice<const double> scales) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return checked_normalized_blob(impl_->value, width, height, means, scales, 3,
                                 true, cv::dnn::DNN_PMODE_NULL);
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_bgr_rgb_normalized_center_crop_owned(
    std::int64_t width, std::int64_t height,
    rust::Slice<const double> means,
    rust::Slice<const double> scales) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return checked_normalized_blob(impl_->value, width, height, means, scales, 3,
                                 true, cv::dnn::DNN_PMODE_CROP_CENTER);
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_bgr_rgb_normalized_letterbox_owned(
    std::int64_t width, std::int64_t height,
    rust::Slice<const double> means,
    rust::Slice<const double> scales) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return checked_normalized_blob(impl_->value, width, height, means, scales, 3,
                                 true, cv::dnn::DNN_PMODE_LETTERBOX);
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_grayscale_normalized_owned(
    std::int64_t width, std::int64_t height,
    rust::Slice<const double> means,
    rust::Slice<const double> scales) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return checked_normalized_blob(impl_->value, width, height, means, scales, 1,
                                 false, cv::dnn::DNN_PMODE_NULL);
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_grayscale_normalized_center_crop_owned(
    std::int64_t width, std::int64_t height,
    rust::Slice<const double> means,
    rust::Slice<const double> scales) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return checked_normalized_blob(impl_->value, width, height, means, scales, 1,
                                 false,
                                 cv::dnn::DNN_PMODE_CROP_CENTER);
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_grayscale_normalized_letterbox_owned(
    std::int64_t width, std::int64_t height,
    rust::Slice<const double> means,
    rust::Slice<const double> scales) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return checked_normalized_blob(impl_->value, width, height, means, scales, 1,
                                 false, cv::dnn::DNN_PMODE_LETTERBOX);
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_bgr_rgb_letterbox_owned(
    std::int64_t width, std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_blob_input(impl_->value, width, height, 3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const dnn_validation::DnnNormalization normalization{
      {0.0, 0.0, 0.0},
      {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}};
  return failure::contain<OpenCvDnnBlob>(
      [this, width, height, normalization] {
        return create_parameterized_blob(
            impl_->value, width, height, normalization, 3, true,
            cv::dnn::DNN_PMODE_LETTERBOX);
      });
}

std::unique_ptr<OpenCvDnnBlob>
OpenCvMat::blob_from_grayscale_letterbox_owned(
    std::int64_t width, std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_blob_input(impl_->value, width, height, 1)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const dnn_validation::DnnNormalization normalization{
      {0.0, 0.0, 0.0},
      {1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0}};
  return failure::contain<OpenCvDnnBlob>(
      [this, width, height, normalization] {
        return create_parameterized_blob(
            impl_->value, width, height, normalization, 1, false,
            cv::dnn::DNN_PMODE_LETTERBOX);
      });
}

std::int64_t live_dnn_blob_count() noexcept {
  return live_dnn_blobs.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
