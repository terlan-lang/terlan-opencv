#pragma once

#include "opencv_failure.hpp"
#include "opencv_validation.hpp"

#include <opencv2/ml.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace terlan_opencv {
namespace ml_internal {

/// Maximum number of rows accepted by one model prediction call.
constexpr std::int64_t kMaxQuerySamples = 4096;
/// Maximum number of copied features accepted per sample.
constexpr std::int64_t kMaxFeatures = 4096;
/// Maximum number of copied scalars accepted by one model operation.
constexpr std::int64_t kMaxModelElements = 4194304;

/// Serializes and restores temporary use of OpenCV's process-global RNG.
class ScopedRandomSeed final {
 public:
  /// Acquires the package RNG lane and installs one deterministic seed.
  explicit ScopedRandomSeed(std::uint64_t seed)
      : lock_(mutex()), previous_state_(cv::theRNG().state) {
    cv::theRNG().state = seed;
  }

  /// Restores the exact caller-visible RNG state before releasing the lane.
  ~ScopedRandomSeed() noexcept { cv::theRNG().state = previous_state_; }

  ScopedRandomSeed(const ScopedRandomSeed&) = delete;
  ScopedRandomSeed& operator=(const ScopedRandomSeed&) = delete;

 private:
  /// Returns the package-owned lock guarding temporary global RNG mutation.
  static std::mutex& mutex() {
    static std::mutex value;
    return value;
  }

  std::unique_lock<std::mutex> lock_;
  std::uint64_t previous_state_;
};

/// Holds one trained OpenCV classifier and its immutable copied dimensions.
template <typename Model>
struct ClassifierState {
  /// Constructs state around one private trained model.
  ClassifierState(cv::Ptr<Model> model, std::int64_t sample_count,
                  std::int64_t feature_count)
      : model(std::move(model)),
        sample_count(sample_count),
        feature_count(feature_count) {}

  /// Private OpenCV statistical model.
  cv::Ptr<Model> model;
  /// Number of copied training rows.
  std::int64_t sample_count;
  /// Number of copied features per row.
  std::int64_t feature_count;
};

/// Reports whether one copied feature scalar can be narrowed safely.
inline bool valid_feature(double value) noexcept {
  return std::isfinite(value) &&
         std::abs(value) <= validation::kMaxTransformCoefficient;
}

/// Reports whether one copied class label has an exact OpenCV representation.
inline bool valid_label(std::int64_t value) noexcept {
  constexpr std::int64_t kMaximumLabel = 1000000;
  return value >= -kMaximumLabel && value <= kMaximumLabel;
}

/// Copies bounded row-major samples into one private float matrix.
inline cv::Mat copy_samples(rust::Slice<const double> samples,
                            std::int64_t rows, std::int64_t columns) {
  cv::Mat result(static_cast<int>(rows), static_cast<int>(columns), CV_32FC1);
  for (std::int64_t row = 0; row < rows; ++row) {
    float* output = result.ptr<float>(static_cast<int>(row));
    for (std::int64_t column = 0; column < columns; ++column) {
      output[column] = static_cast<float>(
          samples[static_cast<std::size_t>(row * columns + column)]);
    }
  }
  return result;
}

/// Copies bounded class labels into one private categorical matrix type.
template <typename Label, int MatrixType>
cv::Mat copy_labels(rust::Slice<const std::int64_t> labels,
                    std::int64_t rows) {
  cv::Mat result(static_cast<int>(rows), 1, MatrixType);
  for (std::int64_t row = 0; row < rows; ++row) {
    result.at<Label>(static_cast<int>(row), 0) =
        static_cast<Label>(labels[static_cast<std::size_t>(row)]);
  }
  return result;
}

/// Reports whether copied labels are bounded and optionally span two classes.
inline bool valid_labels(rust::Slice<const std::int64_t> labels,
                         std::int64_t rows,
                         bool require_multiple_classes) noexcept {
  if (labels.size() != static_cast<std::size_t>(rows)) {
    return false;
  }
  bool multiple_classes = false;
  const std::int64_t first = labels.empty() ? 0 : labels[0];
  for (std::int64_t label : labels) {
    if (!valid_label(label)) {
      return false;
    }
    multiple_classes = multiple_classes || label != first;
  }
  return !require_multiple_classes || multiple_classes;
}

/// Returns two sorted bounded classes or an equal sentinel for invalid input.
inline std::pair<std::int64_t, std::int64_t> binary_label_pair(
    rust::Slice<const std::int64_t> labels, std::int64_t rows) noexcept {
  if (!valid_labels(labels, rows, true)) {
    return {0, 0};
  }
  const std::int64_t first = labels[0];
  std::int64_t second = first;
  for (std::int64_t label : labels) {
    if (label != first) {
      second = label;
      break;
    }
  }
  for (std::int64_t label : labels) {
    if (label != first && label != second) {
      return {0, 0};
    }
  }
  return first < second ? std::make_pair(first, second)
                        : std::make_pair(second, first);
}

/// Returns sorted unique labels or an empty vector for invalid class input.
inline std::vector<std::int64_t> sorted_unique_labels(
    rust::Slice<const std::int64_t> labels, std::int64_t rows,
    std::int64_t maximum_classes) {
  if (!valid_labels(labels, rows, true) || maximum_classes < 2) {
    return {};
  }
  std::vector<std::int64_t> classes(labels.begin(), labels.end());
  std::sort(classes.begin(), classes.end());
  classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
  if (classes.size() > static_cast<std::size_t>(maximum_classes)) {
    return {};
  }
  return classes;
}

/// Copies binary categorical labels into OpenCV's required signed responses.
inline cv::Mat copy_binary_sign_labels(
    rust::Slice<const std::int64_t> labels, std::int64_t rows,
    const std::pair<std::int64_t, std::int64_t>& classes) {
  cv::Mat result(static_cast<int>(rows), 1, CV_32FC1);
  for (std::int64_t row = 0; row < rows; ++row) {
    result.at<float>(static_cast<int>(row), 0) =
        labels[static_cast<std::size_t>(row)] == classes.second ? 1.0F : -1.0F;
  }
  return result;
}

/// Copies binary categorical labels into OpenCV zero/one class indexes.
inline cv::Mat copy_binary_index_labels(
    rust::Slice<const std::int64_t> labels, std::int64_t rows,
    const std::pair<std::int64_t, std::int64_t>& classes) {
  cv::Mat result(static_cast<int>(rows), 1, CV_32FC1);
  for (std::int64_t row = 0; row < rows; ++row) {
    result.at<float>(static_cast<int>(row), 0) =
        labels[static_cast<std::size_t>(row)] == classes.second ? 1.0F : 0.0F;
  }
  return result;
}

/// Validates one bounded flattened sample matrix and returns its row count.
inline std::int64_t valid_sample_rows(rust::Slice<const double> samples,
                                      std::int64_t feature_count,
                                      std::int64_t maximum_rows) noexcept {
  if (feature_count < 1 || feature_count > kMaxFeatures || samples.empty() ||
      samples.size() % static_cast<std::size_t>(feature_count) != 0) {
    return 0;
  }
  const auto rows =
      static_cast<std::int64_t>(samples.size() / feature_count);
  if (rows < 1 || rows > maximum_rows ||
      rows > kMaxModelElements / feature_count) {
    return 0;
  }
  for (double sample : samples) {
    if (!valid_feature(sample)) {
      return 0;
    }
  }
  return rows;
}

/// Reports whether one OpenCV prediction matrix has the reviewed shape.
inline bool valid_float_matrix(const cv::Mat& matrix, std::int64_t rows,
                               std::int64_t columns) noexcept {
  return matrix.type() == CV_32FC1 &&
         matrix.rows == static_cast<int>(rows) &&
         matrix.cols == static_cast<int>(columns) && matrix.isContinuous();
}

/// Copies one validated OpenCV prediction column into integer labels.
inline std::unique_ptr<std::vector<std::int64_t>> copy_prediction_labels(
    const cv::Mat& predictions, std::int64_t rows) {
  if (predictions.rows != static_cast<int>(rows) || predictions.cols != 1 ||
      !predictions.isContinuous() ||
      (predictions.type() != CV_32FC1 && predictions.type() != CV_32SC1)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  auto result = std::make_unique<std::vector<std::int64_t>>();
  result->reserve(static_cast<std::size_t>(rows));
  for (std::int64_t row = 0; row < rows; ++row) {
    const double prediction =
        predictions.type() == CV_32FC1
            ? static_cast<double>(
                  predictions.at<float>(static_cast<int>(row), 0))
            : static_cast<double>(
                  predictions.at<int>(static_cast<int>(row), 0));
    if (!std::isfinite(prediction) || std::trunc(prediction) != prediction ||
        !valid_label(static_cast<std::int64_t>(prediction))) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    result->push_back(static_cast<std::int64_t>(prediction));
  }
  return result;
}

/// Runs the shared copied prediction path for one trained statistical model.
template <typename Model>
std::unique_ptr<std::vector<std::int64_t>> predict_stat_model(
    const ClassifierState<Model>* state,
    rust::Slice<const double> samples) noexcept {
  failure::clear();
  const std::int64_t rows =
      state == nullptr
          ? 0
          : valid_sample_rows(samples, state->feature_count, kMaxQuerySamples);
  if (rows == 0 || state->model.empty() || !state->model->isTrained() ||
      !state->model->isClassifier()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([state, samples, rows] {
    const cv::Mat input = copy_samples(samples, rows, state->feature_count);
    cv::Mat predictions;
    const float first = state->model->predict(input, predictions, 0);
    if (!std::isfinite(first)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    return copy_prediction_labels(predictions, rows);
  });
}

}  // namespace ml_internal
}  // namespace terlan_opencv
