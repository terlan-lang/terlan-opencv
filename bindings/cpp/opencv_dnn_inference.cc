#include "opencv_dnn_internal.hpp"
#include "opencv_dnn_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <utility>

namespace terlan_opencv {
namespace {

constexpr std::size_t kMaxOnnxModelBytes = 128U * 1024U * 1024U;
std::atomic<std::int64_t> live_onnx_models{0};
std::atomic<std::int64_t> live_dnn_outputs{0};
std::atomic<std::int64_t> live_dnn_detections{0};
std::atomic<std::int64_t> live_dnn_classifications{0};

/// Reports whether copied model bytes fit the package's bounded memory policy.
bool valid_model_bytes(rust::Slice<const std::uint8_t> bytes) noexcept {
  return bytes.data() != nullptr && !bytes.empty() &&
         bytes.size() <= kMaxOnnxModelBytes;
}

/// Reports whether one output is a bounded single-batch classification row.
bool valid_classification_output(const cv::Mat& output) noexcept {
  return valid_dnn_output(output) && output.dims == 2 &&
         output.size[0] == 1 &&
         output.size[1] >=
             static_cast<int>(dnn_validation::kMinClassificationCount) &&
         output.size[1] <=
             static_cast<int>(dnn_validation::kMaxClassificationCount);
}

}  // namespace

struct OpenCvDnnDetections::Impl final {
  /// Takes ownership of one validated canonically ordered detection collection.
  explicit Impl(std::vector<dnn_validation::DnnDetection> values)
      : values(std::move(values)) {}

  /// Private copied normalized detection rows.
  std::vector<dnn_validation::DnnDetection> values;
};

struct OpenCvDnnClassifications::Impl final {
  /// Takes ownership of one validated canonically ranked class collection.
  explicit Impl(std::vector<OpenCvDnnClassification> values)
      : values(std::move(values)) {}

  /// Private copied ranked classification records.
  std::vector<OpenCvDnnClassification> values;
};

struct OpenCvOnnxModel::Impl final {
  /// Takes ownership of one configured OpenCV CPU network.
  explicit Impl(cv::dnn::Net network) : network(std::move(network)) {}

  /// OpenCV inference state, privately mutable within this thread-confined owner.
  mutable cv::dnn::Net network;
};

namespace {

/// Copies selected score indexes into one independent canonical owner.
std::unique_ptr<OpenCvDnnClassifications> copy_classifications(
    const float* scores, std::size_t count,
    const std::vector<std::size_t>& indices) {
  std::vector<OpenCvDnnClassification> classifications;
  classifications.reserve(indices.size());
  for (const std::size_t index : indices) {
    if (index >= count) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    classifications.push_back(OpenCvDnnClassification{
        static_cast<std::int64_t>(index), static_cast<double>(scores[index])});
  }
  return std::make_unique<OpenCvDnnClassifications>(
      std::make_unique<OpenCvDnnClassifications::Impl>(
          std::move(classifications)));
}

/// Copies aligned ranked probabilities into one independent canonical owner.
std::unique_ptr<OpenCvDnnClassifications> copy_probability_classifications(
    const std::vector<std::size_t>& indices,
    const std::vector<double>& probabilities) {
  if (indices.size() != probabilities.size()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  std::vector<OpenCvDnnClassification> classifications;
  classifications.reserve(indices.size());
  for (std::size_t position = 0; position < indices.size(); ++position) {
    const double probability = probabilities[position];
    if (!std::isfinite(probability) || probability < 0.0 ||
        probability > 1.0) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    classifications.push_back(OpenCvDnnClassification{
        static_cast<std::int64_t>(indices[position]), probability});
  }
  return std::make_unique<OpenCvDnnClassifications>(
      std::make_unique<OpenCvDnnClassifications::Impl>(
          std::move(classifications)));
}

/// Copies one already validated half-open output range into independent storage.
std::unique_ptr<std::vector<double>> copy_output_values(
    const cv::Mat& output, std::size_t begin, std::size_t end) {
  if (begin > end || end > output.total()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  const float* source = output.ptr<float>();
  auto values = std::make_unique<std::vector<double>>();
  values->reserve(end - begin);
  for (std::size_t index = begin; index < end; ++index) {
    if (!std::isfinite(source[index])) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    values->push_back(source[index]);
  }
  return values;
}

}  // namespace

OpenCvDnnOutput::OpenCvDnnOutput(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_dnn_outputs.fetch_add(1, std::memory_order_relaxed);
}

OpenCvDnnOutput::~OpenCvDnnOutput() noexcept {
  live_dnn_outputs.fetch_sub(1, std::memory_order_relaxed);
}

OpenCvDnnDetections::OpenCvDnnDetections(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_dnn_detections.fetch_add(1, std::memory_order_relaxed);
}

OpenCvDnnDetections::~OpenCvDnnDetections() noexcept {
  live_dnn_detections.fetch_sub(1, std::memory_order_relaxed);
}

OpenCvDnnClassifications::OpenCvDnnClassifications(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_dnn_classifications.fetch_add(1, std::memory_order_relaxed);
}

OpenCvDnnClassifications::~OpenCvDnnClassifications() noexcept {
  live_dnn_classifications.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvDnnClassifications::count() const noexcept {
  if (impl_ == nullptr ||
      impl_->values.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return 0;
  }
  return static_cast<std::int64_t>(impl_->values.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvDnnClassifications::class_indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto values = std::make_unique<std::vector<std::int64_t>>();
    values->reserve(impl_->values.size());
    for (const auto& classification : impl_->values) {
      values->push_back(classification.class_index);
    }
    return values;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvDnnClassifications::scores_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto values = std::make_unique<std::vector<double>>();
    values->reserve(impl_->values.size());
    for (const auto& classification : impl_->values) {
      values->push_back(classification.score);
    }
    return values;
  });
}

std::int64_t OpenCvDnnDetections::count() const noexcept {
  if (impl_ == nullptr ||
      impl_->values.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return 0;
  }
  return static_cast<std::int64_t>(impl_->values.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvDnnDetections::class_indices_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto values = std::make_unique<std::vector<std::int64_t>>();
    values->reserve(impl_->values.size());
    for (const auto& detection : impl_->values) {
      values->push_back(detection.class_index);
    }
    return values;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvDnnDetections::confidences_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto values = std::make_unique<std::vector<double>>();
    values->reserve(impl_->values.size());
    for (const auto& detection : impl_->values) {
      values->push_back(detection.confidence);
    }
    return values;
  });
}

std::unique_ptr<std::vector<double>> OpenCvDnnDetections::boxes_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto values = std::make_unique<std::vector<double>>();
    values->reserve(impl_->values.size() * 4);
    for (const auto& detection : impl_->values) {
      values->insert(values->end(), {detection.left, detection.top,
                                     detection.right, detection.bottom});
    }
    return values;
  });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnDetections::non_max_suppression_owned(
    double overlap_threshold, std::int64_t max_detections) const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  if (max_detections <= 0 ||
      !dnn_validation::nms_parameters(
          overlap_threshold, static_cast<std::size_t>(max_detections))) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, overlap_threshold, max_detections] {
        if (impl_->values.empty()) {
          return std::make_unique<OpenCvDnnDetections>(
              std::make_unique<OpenCvDnnDetections::Impl>(
                  std::vector<dnn_validation::DnnDetection>{}));
        }
        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> class_indices;
        boxes.reserve(impl_->values.size());
        scores.reserve(impl_->values.size());
        class_indices.reserve(impl_->values.size());
        for (const auto& detection : impl_->values) {
          boxes.emplace_back(detection.left, detection.top,
                             detection.right - detection.left,
                             detection.bottom - detection.top);
          scores.push_back(static_cast<float>(detection.confidence));
          class_indices.push_back(static_cast<int>(detection.class_index));
        }
        std::vector<int> selected_indices;
        cv::dnn::NMSBoxesBatched(
            boxes, scores, class_indices, std::nextafter(0.0F, -1.0F),
            static_cast<float>(overlap_threshold), selected_indices, 1.0F, 0);
        std::sort(selected_indices.begin(), selected_indices.end());
        std::vector<dnn_validation::DnnDetection> selected;
        selected.reserve(selected_indices.size());
        int previous = -1;
        for (const int index : selected_indices) {
          if (index < 0 ||
              static_cast<std::size_t>(index) >= impl_->values.size() ||
              index == previous) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<OpenCvDnnDetections>{};
          }
          selected.push_back(impl_->values[static_cast<std::size_t>(index)]);
          previous = index;
        }
        if (selected.size() > static_cast<std::size_t>(max_detections)) {
          selected.resize(static_cast<std::size_t>(max_detections));
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(selected)));
      });
}

std::int64_t OpenCvDnnOutput::rank() const noexcept {
  return impl_ == nullptr ? 0 : impl_->value.dims;
}

std::unique_ptr<std::vector<std::int64_t>> OpenCvDnnOutput::shape_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_dnn_output(impl_->value)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto shape = std::make_unique<std::vector<std::int64_t>>();
    shape->reserve(static_cast<std::size_t>(impl_->value.dims));
    for (int dimension = 0; dimension < impl_->value.dims; ++dimension) {
      if (impl_->value.size[dimension] <= 0) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      shape->push_back(impl_->value.size[dimension]);
    }
    return shape;
  });
}

std::unique_ptr<std::vector<double>> OpenCvDnnOutput::values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_dnn_output(impl_->value)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return copy_output_values(impl_->value, 0, impl_->value.total()); });
}

std::int64_t OpenCvDnnOutput::value_count() const noexcept {
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      impl_->value.total() >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return 0;
  }
  return static_cast<std::int64_t>(impl_->value.total());
}

std::unique_ptr<std::vector<double>> OpenCvDnnOutput::values_slice_owned(
    std::int64_t offset, std::int64_t max_values) const noexcept {
  failure::clear();
  std::size_t begin = 0;
  std::size_t end = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::output_slice_bounds(impl_->value.total(), offset,
                                           max_values, begin, end)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, begin, end] { return copy_output_values(impl_->value, begin, end); });
}

std::unique_ptr<OpenCvDnnClassifications>
OpenCvDnnOutput::softmax_classifications_owned(
    std::int64_t max_classes) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_classification_output(impl_->value) ||
      max_classes <= 0 ||
      max_classes > static_cast<std::int64_t>(
                        dnn_validation::kMaxTopClassificationCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnClassifications>([this, max_classes] {
    const float* scores = impl_->value.ptr<float>();
    const std::size_t count = static_cast<std::size_t>(impl_->value.size[1]);
    std::vector<std::size_t> indices;
    std::vector<double> probabilities;
    if (!dnn_validation::softmax_classifications(
            scores, count, static_cast<std::size_t>(max_classes), indices,
            probabilities)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvDnnClassifications>{};
    }
    return copy_probability_classifications(indices, probabilities);
  });
}

std::unique_ptr<OpenCvDnnClassifications>
OpenCvDnnOutput::sigmoid_classifications_owned(
    std::int64_t max_classes) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_classification_output(impl_->value) ||
      max_classes <= 0 ||
      max_classes > static_cast<std::int64_t>(
                        dnn_validation::kMaxTopClassificationCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnClassifications>([this, max_classes] {
    const float* scores = impl_->value.ptr<float>();
    const std::size_t count = static_cast<std::size_t>(impl_->value.size[1]);
    std::vector<std::size_t> indices;
    std::vector<double> probabilities;
    if (!dnn_validation::sigmoid_classifications(
            scores, count, static_cast<std::size_t>(max_classes), indices,
            probabilities)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvDnnClassifications>{};
    }
    return copy_probability_classifications(indices, probabilities);
  });
}

std::unique_ptr<OpenCvDnnClassifications>
OpenCvDnnOutput::sigmoid_classification_candidates_owned(
    double minimum_probability, std::int64_t max_classes) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_classification_output(impl_->value) ||
      !dnn_validation::valid_probability(minimum_probability) ||
      max_classes <= 0 ||
      max_classes > static_cast<std::int64_t>(
                        dnn_validation::kMaxTopClassificationCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnClassifications>(
      [this, minimum_probability, max_classes] {
        const float* scores = impl_->value.ptr<float>();
        const std::size_t count =
            static_cast<std::size_t>(impl_->value.size[1]);
        std::vector<std::size_t> indices;
        std::vector<double> probabilities;
        if (!dnn_validation::sigmoid_classification_candidates(
                scores, count, minimum_probability,
                static_cast<std::size_t>(max_classes), indices,
                probabilities)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnClassifications>{};
        }
        return copy_probability_classifications(indices, probabilities);
      });
}

std::unique_ptr<OpenCvDnnClassification> top_classification_owned(
    const OpenCvDnnOutput& output) noexcept {
  failure::clear();
  if (output.impl_ == nullptr ||
      !valid_classification_output(output.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnClassification>([&output] {
    const float* scores = output.impl_->value.ptr<float>();
    std::int64_t best_index = 0;
    double best_score = 0.0;
    if (!dnn_validation::top_classification(
            scores, static_cast<std::size_t>(output.impl_->value.size[1]),
            best_index, best_score)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvDnnClassification>{};
    }
    return std::make_unique<OpenCvDnnClassification>(OpenCvDnnClassification{
        best_index, best_score});
  });
}

std::unique_ptr<OpenCvDnnClassifications> top_classifications_owned(
    const OpenCvDnnOutput& output, std::int64_t max_classes) noexcept {
  failure::clear();
  if (output.impl_ == nullptr ||
      !valid_classification_output(output.impl_->value) || max_classes <= 0 ||
      max_classes >
          static_cast<std::int64_t>(
              dnn_validation::kMaxTopClassificationCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnClassifications>([&output, max_classes] {
    const float* scores = output.impl_->value.ptr<float>();
    const std::size_t count =
        static_cast<std::size_t>(output.impl_->value.size[1]);
    std::vector<std::size_t> indices;
    if (!dnn_validation::top_classification_indices(
            scores, count, static_cast<std::size_t>(max_classes), indices)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvDnnClassifications>{};
    }
    return copy_classifications(scores, count, indices);
  });
}

std::unique_ptr<OpenCvDnnClassifications> classification_candidates_owned(
    const OpenCvDnnOutput& output, double minimum_score,
    std::int64_t max_classes) noexcept {
  failure::clear();
  if (output.impl_ == nullptr ||
      !valid_classification_output(output.impl_->value) ||
      !std::isfinite(minimum_score) || max_classes <= 0 ||
      max_classes >
          static_cast<std::int64_t>(
              dnn_validation::kMaxTopClassificationCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnClassifications>(
      [&output, minimum_score, max_classes] {
        const float* scores = output.impl_->value.ptr<float>();
        const std::size_t count =
            static_cast<std::size_t>(output.impl_->value.size[1]);
        std::vector<std::size_t> indices;
        if (!dnn_validation::classification_candidate_indices(
                scores, count, minimum_score,
                static_cast<std::size_t>(max_classes), indices)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnClassifications>{};
        }
        return copy_classifications(scores, count, indices);
      });
}

std::unique_ptr<OpenCvMat> channel_mask_owned(
    const OpenCvDnnOutput& output, std::int64_t channel,
    double threshold) noexcept {
  failure::clear();
  if (output.impl_ == nullptr || !valid_dnn_output(output.impl_->value) ||
      output.impl_->value.dims != 4 || output.impl_->value.size[0] != 1 ||
      output.impl_->value.size[1] <= 0 || output.impl_->value.size[2] <= 0 ||
      output.impl_->value.size[3] <= 0 || channel < 0 ||
      channel >= output.impl_->value.size[1] || !std::isfinite(threshold) ||
      !validation::image_shape(output.impl_->value.size[2],
                               output.impl_->value.size[3], 1)
           .valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([&output, channel, threshold] {
    const cv::Mat& value = output.impl_->value;
    cv::Mat mask(value.size[2], value.size[3], CV_8UC1);
    if (!dnn_validation::channel_mask(
            value.ptr<float>(), value.total(),
            static_cast<std::size_t>(value.size[1]),
            static_cast<std::size_t>(value.size[2]),
            static_cast<std::size_t>(value.size[3]),
            static_cast<std::size_t>(channel), threshold,
            mask.ptr<std::uint8_t>())) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvMat>{};
    }
    return std::make_unique<OpenCvMat>(
        std::make_unique<OpenCvMat::Impl>(std::move(mask)));
  });
}

std::unique_ptr<OpenCvMat> class_index_mask_owned(
    const OpenCvDnnOutput& output) noexcept {
  failure::clear();
  if (output.impl_ == nullptr || !valid_dnn_output(output.impl_->value) ||
      output.impl_->value.dims != 4 || output.impl_->value.size[0] != 1 ||
      output.impl_->value.size[1] <
          static_cast<int>(dnn_validation::kMinClassMaskCount) ||
      output.impl_->value.size[1] >
          static_cast<int>(dnn_validation::kMaxClassMaskCount) ||
      output.impl_->value.size[2] <= 0 || output.impl_->value.size[3] <= 0 ||
      !validation::image_shape(output.impl_->value.size[2],
                               output.impl_->value.size[3], 1)
           .valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([&output] {
    const cv::Mat& value = output.impl_->value;
    cv::Mat mask(value.size[2], value.size[3], CV_8UC1);
    if (!dnn_validation::class_index_mask(
            value.ptr<float>(), value.total(),
            static_cast<std::size_t>(value.size[1]),
            static_cast<std::size_t>(value.size[2]),
            static_cast<std::size_t>(value.size[3]),
            mask.ptr<std::uint8_t>())) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvMat>{};
    }
    return std::make_unique<OpenCvMat>(
        std::make_unique<OpenCvMat::Impl>(std::move(mask)));
  });
}

std::unique_ptr<OpenCvMat> class_confidence_mask_owned(
    const OpenCvDnnOutput& output) noexcept {
  failure::clear();
  if (output.impl_ == nullptr || !valid_dnn_output(output.impl_->value) ||
      output.impl_->value.dims != 4 || output.impl_->value.size[0] != 1 ||
      output.impl_->value.size[1] <
          static_cast<int>(dnn_validation::kMinClassMaskCount) ||
      output.impl_->value.size[1] >
          static_cast<int>(dnn_validation::kMaxClassMaskCount) ||
      output.impl_->value.size[2] <= 0 || output.impl_->value.size[3] <= 0 ||
      !validation::image_shape(output.impl_->value.size[2],
                               output.impl_->value.size[3], 1)
           .valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([&output] {
    const cv::Mat& value = output.impl_->value;
    cv::Mat mask(value.size[2], value.size[3], CV_8UC1);
    if (!dnn_validation::class_confidence_mask(
            value.ptr<float>(), value.total(),
            static_cast<std::size_t>(value.size[1]),
            static_cast<std::size_t>(value.size[2]),
            static_cast<std::size_t>(value.size[3]),
            mask.ptr<std::uint8_t>())) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvMat>{};
    }
    return std::make_unique<OpenCvMat>(
        std::make_unique<OpenCvMat::Impl>(std::move(mask)));
  });
}

std::unique_ptr<OpenCvDnnDetections> ssd_detections_owned(
    const OpenCvDnnOutput& output, double threshold,
    std::int64_t max_detections) noexcept {
  failure::clear();
  if (output.impl_ == nullptr || !valid_dnn_output(output.impl_->value) ||
      output.impl_->value.dims != 4 || output.impl_->value.size[0] != 1 ||
      output.impl_->value.size[1] != 1 || output.impl_->value.size[2] <= 0 ||
      output.impl_->value.size[3] != 7 || !std::isfinite(threshold) ||
      threshold < 0.0 || threshold > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [&output, threshold, max_detections] {
        const cv::Mat& value = output.impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::ssd_detections(
                value.ptr<float>(), value.total(),
                static_cast<std::size_t>(value.size[2]), threshold,
                static_cast<std::size_t>(max_detections), detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_xyxy_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_xyxy_detections(
                value.ptr<float>(), value.total(), rows, minimum_confidence,
                static_cast<std::size_t>(max_detections), detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_center_xywh_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_center_xywh_detections(
                value.ptr<float>(), value.total(), rows, minimum_confidence,
                static_cast<std::size_t>(max_detections), detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_top_left_xywh_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_top_left_xywh_detections(
                value.ptr<float>(), value.total(), rows, minimum_confidence,
                static_cast<std::size_t>(max_detections), detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_objectness_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  std::size_t classes = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_objectness_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows, classes) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, classes, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_objectness_detections(
                value.ptr<float>(), value.total(), rows, classes,
                minimum_confidence, static_cast<std::size_t>(max_detections),
                detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_class_score_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  std::size_t classes = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_class_score_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows, classes) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, classes, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_class_score_detections(
                value.ptr<float>(), value.total(), rows, classes,
                minimum_confidence, static_cast<std::size_t>(max_detections),
                detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_transposed_class_score_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  std::size_t classes = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_transposed_class_score_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows, classes) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, classes, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_transposed_class_score_detections(
                value.ptr<float>(), value.total(), rows, classes,
                minimum_confidence, static_cast<std::size_t>(max_detections),
                detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

std::unique_ptr<OpenCvDnnDetections>
OpenCvDnnOutput::normalized_transposed_objectness_detections_owned(
    double minimum_confidence, std::int64_t max_detections) const noexcept {
  failure::clear();
  std::size_t rows = 0;
  std::size_t classes = 0;
  if (impl_ == nullptr || !valid_dnn_output(impl_->value) ||
      !dnn_validation::normalized_transposed_objectness_detection_shape(
          impl_->value.size.p, static_cast<std::size_t>(impl_->value.dims),
          rows, classes) ||
      !std::isfinite(minimum_confidence) || minimum_confidence < 0.0 ||
      minimum_confidence > 1.0 || max_detections <= 0 ||
      max_detections >
          static_cast<std::int64_t>(dnn_validation::kMaxDetectionCount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnDetections>(
      [this, rows, classes, minimum_confidence, max_detections] {
        const cv::Mat& value = impl_->value;
        std::vector<dnn_validation::DnnDetection> detections;
        if (!dnn_validation::normalized_transposed_objectness_detections(
                value.ptr<float>(), value.total(), rows, classes,
                minimum_confidence, static_cast<std::size_t>(max_detections),
                detections)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDnnDetections>{};
        }
        return std::make_unique<OpenCvDnnDetections>(
            std::make_unique<OpenCvDnnDetections::Impl>(
                std::move(detections)));
      });
}

OpenCvOnnxModel::OpenCvOnnxModel(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_onnx_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvOnnxModel::~OpenCvOnnxModel() noexcept {
  live_onnx_models.fetch_sub(1, std::memory_order_relaxed);
}

std::unique_ptr<OpenCvDnnOutput> OpenCvOnnxModel::forward_owned(
    const OpenCvDnnBlob& blob) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->network.empty() || blob.impl_ == nullptr ||
      blob.impl_->value.dims != 4 || blob.impl_->value.type() != CV_32FC1 ||
      !blob.impl_->value.isContinuous()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDnnOutput>([this, &blob] {
    cv::Mat input = blob.impl_->value.clone();
    impl_->network.setInput(input);
    cv::Mat output = impl_->network.forward();
    if (!valid_dnn_output(output)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvDnnOutput>{};
    }
    return std::make_unique<OpenCvDnnOutput>(
        std::make_unique<OpenCvDnnOutput::Impl>(std::move(output)));
  });
}

std::unique_ptr<OpenCvOnnxModel> load_onnx_model_owned(
    rust::Slice<const std::uint8_t> bytes) noexcept {
  failure::clear();
  if (!valid_model_bytes(bytes)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvOnnxModel>([bytes] {
    const std::vector<uchar> model(bytes.begin(), bytes.end());
    cv::dnn::Net network = cv::dnn::readNetFromONNX(model);
    if (network.empty()) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvOnnxModel>{};
    }
    network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return std::make_unique<OpenCvOnnxModel>(
        std::make_unique<OpenCvOnnxModel::Impl>(std::move(network)));
  });
}

std::int64_t live_onnx_model_count() noexcept {
  return live_onnx_models.load(std::memory_order_relaxed);
}

std::int64_t live_dnn_output_count() noexcept {
  return live_dnn_outputs.load(std::memory_order_relaxed);
}

std::int64_t live_dnn_detections_count() noexcept {
  return live_dnn_detections.load(std::memory_order_relaxed);
}

std::int64_t live_dnn_classifications_count() noexcept {
  return live_dnn_classifications.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
