#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#if defined(TERLAN_OPENCV_METADATA)
#include "opencv_metadata_rust.hpp"
#else
#include "rust/cxx.h"
#endif

namespace terlan_opencv {

class OpenCvMat;
class OpenCvOnnxModel;
class OpenCvDnnDetections; class OpenCvDnnClassifications;

/// Uniquely owns one immutable CPU NCHW floating-point inference blob.
class OpenCvDnnBlob final {
 public:
  struct Impl;
  /// Constructs a blob from its private OpenCV representation.
  explicit OpenCvDnnBlob(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned inference blob.
  ~OpenCvDnnBlob() noexcept;
  /// Returns the immutable batch count.
  std::int64_t batch_count() const noexcept;
  /// Returns the immutable channel count.
  std::int64_t channel_count() const noexcept;
  /// Returns the immutable image height.
  std::int64_t height() const noexcept;
  /// Returns the immutable image width.
  std::int64_t width() const noexcept;
  /// Copies finite NCHW floating-point values into ordinary owned storage.
  std::unique_ptr<std::vector<double>> values_owned() const noexcept;

 private:
  friend class OpenCvOnnxModel;
  std::unique_ptr<Impl> impl_;
};

/// Returns the number of live inference-blob owners.
std::int64_t live_dnn_blob_count() noexcept;

/// Holds one copied deterministic top-one classification result.
struct OpenCvDnnClassification final {
  std::int64_t class_index;  ///< Zero-based output class index.
  double score;              ///< Finite raw model score.
  /// Returns the zero-based output class index.
  std::int64_t class_index_value() const noexcept { return class_index; }
  /// Returns the finite raw model score.
  double score_value() const noexcept { return score; }
};

/// Uniquely owns one immutable finite CPU inference result.
class OpenCvDnnOutput final {
 public:
  struct Impl;
  /// Constructs an output from its private OpenCV representation.
  explicit OpenCvDnnOutput(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned inference output.
  ~OpenCvDnnOutput() noexcept;
  /// Returns the number of output dimensions.
  std::int64_t rank() const noexcept;
  /// Copies output dimensions in outermost-to-innermost order.
  std::unique_ptr<std::vector<std::int64_t>> shape_owned() const noexcept;
  /// Copies finite output values in OpenCV's contiguous storage order.
  std::unique_ptr<std::vector<double>> values_owned() const noexcept; /** Returns the contiguous output value count without copying storage. */ std::int64_t value_count() const noexcept; /** Copies one bounded contiguous range of finite output values. */ std::unique_ptr<std::vector<double>> values_slice_owned(std::int64_t offset, std::int64_t max_values) const noexcept; /** Copies bounded ranked stable-softmax probabilities. */ std::unique_ptr<OpenCvDnnClassifications> softmax_classifications_owned(std::int64_t max_classes) const noexcept; /** Projects per-pixel stable-softmax confidence into a grayscale mask. */ std::unique_ptr<OpenCvMat> softmax_class_confidence_mask_owned() const noexcept; /** Projects one class's per-pixel stable-softmax probability. */ std::unique_ptr<OpenCvMat> softmax_class_probability_mask_owned(std::int64_t class_index) const noexcept; /** Copies bounded ranked stable-sigmoid probabilities. */ std::unique_ptr<OpenCvDnnClassifications> sigmoid_classifications_owned(std::int64_t max_classes) const noexcept; /** Copies thresholded stable-sigmoid probabilities. */ std::unique_ptr<OpenCvDnnClassifications> sigmoid_classification_candidates_owned(double minimum_probability, std::int64_t max_classes) const noexcept; /** Projects one class's per-pixel stable-sigmoid probability. */ std::unique_ptr<OpenCvMat> sigmoid_class_probability_mask_owned(std::int64_t class_index) const noexcept; /** Thresholds one class's per-pixel stable-sigmoid probability. */ std::unique_ptr<OpenCvMat> sigmoid_class_mask_owned(std::int64_t class_index, double minimum_probability) const noexcept; /** Thresholds one class's per-pixel stable-softmax probability. */ std::unique_ptr<OpenCvMat> softmax_class_mask_owned(std::int64_t class_index, double minimum_probability) const noexcept; /** Thresholds whether any per-pixel stable-sigmoid class matches. */ std::unique_ptr<OpenCvMat> sigmoid_any_class_mask_owned(double minimum_probability) const noexcept; /** Projects maximum per-pixel stable-sigmoid confidence. */ std::unique_ptr<OpenCvMat> sigmoid_class_confidence_mask_owned() const noexcept; /** Decodes normalized XYXY detection rows into copied storage. */ std::unique_ptr<OpenCvDnnDetections> normalized_xyxy_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept; /** Converts normalized center-XYWH detection rows into copied XYXY storage. */ std::unique_ptr<OpenCvDnnDetections> normalized_center_xywh_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept; /** Converts normalized top-left-XYWH detection rows into copied XYXY storage. */ std::unique_ptr<OpenCvDnnDetections> normalized_top_left_xywh_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept; /** Decodes normalized center-XYWH objectness rows with class probabilities. */ std::unique_ptr<OpenCvDnnDetections> normalized_objectness_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept; /** Decodes normalized center-XYWH rows with class probabilities. */ std::unique_ptr<OpenCvDnnDetections> normalized_class_score_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept; /** Decodes field-major normalized center-XYWH rows with class probabilities. */ std::unique_ptr<OpenCvDnnDetections> normalized_transposed_class_score_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept; /** Decodes field-major normalized center-XYWH objectness planes with class probabilities. */ std::unique_ptr<OpenCvDnnDetections> normalized_transposed_objectness_detections_owned(double minimum_confidence, std::int64_t max_detections) const noexcept;

 private:
  friend std::unique_ptr<OpenCvDnnClassification> top_classification_owned(
      const OpenCvDnnOutput& output) noexcept; friend std::unique_ptr<OpenCvDnnClassifications> top_classifications_owned(const OpenCvDnnOutput& output, std::int64_t max_classes) noexcept; friend std::unique_ptr<OpenCvDnnClassifications> classification_candidates_owned(const OpenCvDnnOutput& output, double minimum_score, std::int64_t max_classes) noexcept;
  friend std::unique_ptr<OpenCvMat> channel_mask_owned(
      const OpenCvDnnOutput& output, std::int64_t channel,
      double threshold) noexcept;
  friend std::unique_ptr<OpenCvMat> class_index_mask_owned(
      const OpenCvDnnOutput& output) noexcept;
  friend std::unique_ptr<OpenCvMat> class_confidence_mask_owned(
      const OpenCvDnnOutput& output) noexcept;
  friend std::unique_ptr<OpenCvDnnDetections> ssd_detections_owned(
      const OpenCvDnnOutput& output, double threshold,
      std::int64_t max_detections) noexcept;
  std::unique_ptr<Impl> impl_;
};

/// Uniquely owns one copied deterministic collection of normalized detections.
class OpenCvDnnDetections final {
 public:
  struct Impl;
  /// Constructs detections from their private validated representation.
  explicit OpenCvDnnDetections(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned detection collection.
  ~OpenCvDnnDetections() noexcept;
  /// Returns the immutable detection count.
  std::int64_t count() const noexcept;
  /// Copies model-local class indexes in canonical order.
  std::unique_ptr<std::vector<std::int64_t>> class_indices_owned()
      const noexcept;
  /// Copies normalized confidence values in canonical order.
  std::unique_ptr<std::vector<double>> confidences_owned() const noexcept;
  /// Copies normalized left, top, right, and bottom coordinates by row.
  std::unique_ptr<std::vector<double>> boxes_owned() const noexcept;
  /// Applies fixed-policy class-aware OpenCV non-maximum suppression.
  std::unique_ptr<OpenCvDnnDetections> non_max_suppression_owned(
      double overlap_threshold, std::int64_t max_detections) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Uniquely owns one in-memory ONNX network fixed to OpenCV's CPU backend.
class OpenCvOnnxModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvOnnxModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned ONNX model.
  ~OpenCvOnnxModel() noexcept;
  /// Runs one synchronous inference from an immutable copied input blob.
  std::unique_ptr<OpenCvDnnOutput> forward_owned(
      const OpenCvDnnBlob& blob) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Loads copied in-memory ONNX bytes into a fixed OpenCV CPU network.
std::unique_ptr<OpenCvOnnxModel> load_onnx_model_owned(
    rust::Slice<const std::uint8_t> bytes) noexcept;

/// Returns the number of live ONNX model owners.
std::int64_t live_onnx_model_count() noexcept;

/// Returns the number of live inference-output owners.
std::int64_t live_dnn_output_count() noexcept;

/// Copies the deterministic top-one result from one batch of class scores.
std::unique_ptr<OpenCvDnnClassification> top_classification_owned(
    const OpenCvDnnOutput& output) noexcept;

/// Thresholds one output channel into a newly owned grayscale byte mask.
std::unique_ptr<OpenCvMat> channel_mask_owned(
    const OpenCvDnnOutput& output, std::int64_t channel,
    double threshold) noexcept;

/// Selects one class index per output pixel into a new grayscale byte mask.
std::unique_ptr<OpenCvMat> class_index_mask_owned(
    const OpenCvDnnOutput& output) noexcept;

/// Projects each pixel's highest probability into a new grayscale byte mask.
std::unique_ptr<OpenCvMat> class_confidence_mask_owned(
    const OpenCvDnnOutput& output) noexcept;

/// Decodes one normalized `[1, 1, rows, 7]` SSD output collection.
std::unique_ptr<OpenCvDnnDetections> ssd_detections_owned(
    const OpenCvDnnOutput& output, double threshold,
    std::int64_t max_detections) noexcept;

/// Returns the number of live copied detection-collection owners.
std::int64_t live_dnn_detections_count() noexcept;

/// Uniquely owns one bounded deterministic collection of ranked classes.
class OpenCvDnnClassifications final {
 public:
  struct Impl;
  /// Constructs ranked classes from their private validated representation.
  explicit OpenCvDnnClassifications(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the owned ranked classification collection.
  ~OpenCvDnnClassifications() noexcept;
  /// Returns the immutable ranked classification count.
  std::int64_t count() const noexcept;
  /// Copies ranked model-local class indexes.
  std::unique_ptr<std::vector<std::int64_t>> class_indices_owned()
      const noexcept;
  /// Copies ranked finite raw model scores.
  std::unique_ptr<std::vector<double>> scores_owned() const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Copies a bounded deterministic ranking from one class-score batch.
std::unique_ptr<OpenCvDnnClassifications> top_classifications_owned(
    const OpenCvDnnOutput& output, std::int64_t max_classes) noexcept;

/// Returns the number of live ranked classification-collection owners.
std::int64_t live_dnn_classifications_count() noexcept;

/// Copies bounded canonically ranked classes meeting an inclusive raw-score threshold.
std::unique_ptr<OpenCvDnnClassifications> classification_candidates_owned(
    const OpenCvDnnOutput& output, double minimum_score,
    std::int64_t max_classes) noexcept;

}  // namespace terlan_opencv
