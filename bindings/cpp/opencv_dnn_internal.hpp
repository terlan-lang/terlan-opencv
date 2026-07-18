#ifndef TERLAN_OPENCV_DNN_INTERNAL_HPP
#define TERLAN_OPENCV_DNN_INTERNAL_HPP

#include "opencv_dnn.hpp"
#include "opencv_validation.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace terlan_opencv {

/// Maximum rank accepted for one copied inference output.
constexpr int kMaxDnnOutputRank = 32;

/// Private native storage shared by preprocessing and inference translation units.
struct OpenCvDnnBlob::Impl final {
  /// Takes ownership of one validated NCHW floating-point blob.
  explicit Impl(cv::Mat value) : value(std::move(value)) {}

  /// Private four-dimensional OpenCV tensor storage.
  cv::Mat value;
};

/// Private native storage shared by inference projection translation units.
struct OpenCvDnnOutput::Impl final {
  /// Takes ownership of one validated finite contiguous output tensor.
  explicit Impl(cv::Mat value) : value(std::move(value)) {}

  /// Private contiguous OpenCV output storage.
  cv::Mat value;
};

/// Reports whether one output can be copied into package-owned storage.
inline bool valid_dnn_output(const cv::Mat& output) noexcept {
  return !output.empty() && output.dims > 0 &&
         output.dims <= kMaxDnnOutputRank && output.type() == CV_32FC1 &&
         output.isContinuous() && output.total() > 0 &&
         validation::valid_decoded_storage(output.total(), sizeof(float));
}

}  // namespace terlan_opencv

#endif
