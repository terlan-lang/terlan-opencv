#include "opencv_dnn_internal.hpp"
#include "opencv_dnn_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <utility>

namespace terlan_opencv {
namespace {

/// Reports whether one rank-four output supports a bounded channel projection.
bool valid_segmentation_output(const cv::Mat& value,
                               std::size_t minimum_channels) noexcept {
  return valid_dnn_output(value) && value.dims == 4 && value.size[0] == 1 &&
         value.size[1] >= static_cast<int>(minimum_channels) &&
         value.size[1] <=
             static_cast<int>(dnn_validation::kMaxClassMaskCount) &&
         value.size[2] > 0 && value.size[3] > 0 &&
         validation::image_shape(value.size[2], value.size[3], 1).valid;
}

/// Allocates one grayscale mask and publishes it only after projection succeeds.
template <typename Projection>
std::unique_ptr<OpenCvMat> project_mask(const cv::Mat& value,
                                        Projection projection) noexcept {
  return failure::contain<OpenCvMat>([&value, &projection] {
    cv::Mat mask(value.size[2], value.size[3], CV_8UC1);
    if (!projection(mask.ptr<std::uint8_t>())) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvMat>{};
    }
    return std::make_unique<OpenCvMat>(
        std::make_unique<OpenCvMat::Impl>(std::move(mask)));
  });
}

}  // namespace

std::unique_ptr<OpenCvMat>
OpenCvDnnOutput::sigmoid_class_probability_mask_owned(
    std::int64_t class_index) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_segmentation_output(impl_->value, 1) ||
      class_index < 0 || class_index >= impl_->value.size[1]) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(impl_->value, [this, class_index](std::uint8_t* mask) {
    const cv::Mat& value = impl_->value;
    return dnn_validation::sigmoid_class_probability_mask(
        value.ptr<float>(), value.total(),
        static_cast<std::size_t>(value.size[1]),
        static_cast<std::size_t>(value.size[2]),
        static_cast<std::size_t>(value.size[3]),
        static_cast<std::size_t>(class_index), mask);
  });
}

std::unique_ptr<OpenCvMat> OpenCvDnnOutput::sigmoid_class_mask_owned(
    std::int64_t class_index, double minimum_probability) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_segmentation_output(impl_->value, 1) ||
      class_index < 0 || class_index >= impl_->value.size[1] ||
      !dnn_validation::valid_probability(minimum_probability)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(
      impl_->value,
      [this, class_index, minimum_probability](std::uint8_t* mask) {
        const cv::Mat& value = impl_->value;
        return dnn_validation::sigmoid_class_mask(
            value.ptr<float>(), value.total(),
            static_cast<std::size_t>(value.size[1]),
            static_cast<std::size_t>(value.size[2]),
            static_cast<std::size_t>(value.size[3]),
            static_cast<std::size_t>(class_index), minimum_probability, mask);
      });
}

std::unique_ptr<OpenCvMat> OpenCvDnnOutput::sigmoid_any_class_mask_owned(
    double minimum_probability) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_segmentation_output(impl_->value, 1) ||
      !dnn_validation::valid_probability(minimum_probability)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(
      impl_->value, [this, minimum_probability](std::uint8_t* mask) {
        const cv::Mat& value = impl_->value;
        return dnn_validation::sigmoid_any_class_mask(
            value.ptr<float>(), value.total(),
            static_cast<std::size_t>(value.size[1]),
            static_cast<std::size_t>(value.size[2]),
            static_cast<std::size_t>(value.size[3]), minimum_probability,
            mask);
      });
}

std::unique_ptr<OpenCvMat>
OpenCvDnnOutput::sigmoid_class_confidence_mask_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_segmentation_output(impl_->value, 1)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(impl_->value, [this](std::uint8_t* mask) {
    const cv::Mat& value = impl_->value;
    return dnn_validation::sigmoid_class_confidence_mask(
        value.ptr<float>(), value.total(),
        static_cast<std::size_t>(value.size[1]),
        static_cast<std::size_t>(value.size[2]),
        static_cast<std::size_t>(value.size[3]), mask);
  });
}

std::unique_ptr<OpenCvMat>
OpenCvDnnOutput::softmax_class_confidence_mask_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_segmentation_output(
                              impl_->value,
                              dnn_validation::kMinClassMaskCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(impl_->value, [this](std::uint8_t* mask) {
    const cv::Mat& value = impl_->value;
    return dnn_validation::softmax_class_confidence_mask(
        value.ptr<float>(), value.total(),
        static_cast<std::size_t>(value.size[1]),
        static_cast<std::size_t>(value.size[2]),
        static_cast<std::size_t>(value.size[3]), mask);
  });
}

std::unique_ptr<OpenCvMat>
OpenCvDnnOutput::softmax_class_probability_mask_owned(
    std::int64_t class_index) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !valid_segmentation_output(impl_->value,
                                 dnn_validation::kMinClassMaskCount) ||
      class_index < 0 || class_index >= impl_->value.size[1]) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(impl_->value, [this, class_index](std::uint8_t* mask) {
    const cv::Mat& value = impl_->value;
    return dnn_validation::softmax_class_probability_mask(
        value.ptr<float>(), value.total(),
        static_cast<std::size_t>(value.size[1]),
        static_cast<std::size_t>(value.size[2]),
        static_cast<std::size_t>(value.size[3]),
        static_cast<std::size_t>(class_index), mask);
  });
}

std::unique_ptr<OpenCvMat> OpenCvDnnOutput::softmax_class_mask_owned(
    std::int64_t class_index, double minimum_probability) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !valid_segmentation_output(impl_->value,
                                 dnn_validation::kMinClassMaskCount) ||
      class_index < 0 || class_index >= impl_->value.size[1] ||
      !dnn_validation::valid_probability(minimum_probability)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return project_mask(
      impl_->value,
      [this, class_index, minimum_probability](std::uint8_t* mask) {
        const cv::Mat& value = impl_->value;
        return dnn_validation::softmax_class_mask(
            value.ptr<float>(), value.total(),
            static_cast<std::size_t>(value.size[1]),
            static_cast<std::size_t>(value.size[2]),
            static_cast<std::size_t>(value.size[3]),
            static_cast<std::size_t>(class_index), minimum_probability, mask);
      });
}

}  // namespace terlan_opencv
