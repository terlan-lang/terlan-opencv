#ifndef TERLAN_OPENCV_DNN_DETECTION_VALIDATION_HPP
#define TERLAN_OPENCV_DNN_DETECTION_VALIDATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace terlan_opencv {
namespace dnn_validation {

/// Maximum source rows accepted by one detection projection.
constexpr std::size_t kMaxDetectionRowCount = 100000;

/// Maximum selected detections returned by one projection.
constexpr std::size_t kMaxDetectionCount = 10000;

/// Maximum exactly representable nonnegative class index admitted by policy.
constexpr std::int64_t kMaxDetectionClassIndex = 1000000;

/// Maximum class-probability columns accepted by objectness decoding.
constexpr std::size_t kMaxObjectnessClassCount = 10000;

/// One validated copied normalized detection row.
struct DnnDetection final {
  std::int64_t class_index;  ///< Nonnegative model-local class index.
  double confidence;         ///< Finite normalized confidence.
  double left;               ///< Normalized inclusive left coordinate.
  double top;                ///< Normalized inclusive top coordinate.
  double right;              ///< Normalized right coordinate.
  double bottom;             ///< Normalized bottom coordinate.
  std::size_t source_index;  ///< Original row index for stable ordering.
};

/// Orders detections by confidence, class, and original source row, then bounds them.
inline void canonicalize_detections(std::vector<DnnDetection>& detections,
                                    std::size_t max_detections) {
  std::sort(detections.begin(), detections.end(),
            [](const DnnDetection& first, const DnnDetection& second) {
              if (first.confidence != second.confidence) {
                return first.confidence > second.confidence;
              }
              if (first.class_index != second.class_index) {
                return first.class_index < second.class_index;
              }
              return first.source_index < second.source_index;
            });
  if (detections.size() > max_detections) {
    detections.resize(max_detections);
  }
}

/// Validates shared confidence-threshold and result-bound detection policy.
inline bool detection_parameters(double minimum_confidence,
                                 std::size_t max_detections) noexcept {
  return std::isfinite(minimum_confidence) && minimum_confidence >= 0.0 &&
         minimum_confidence <= 1.0 && max_detections > 0 &&
         max_detections <= kMaxDetectionCount;
}

/// Returns whether one model-local detection class index is representable.
inline bool valid_detection_class(float class_index) noexcept {
  return std::isfinite(class_index) && class_index >= 0.0F &&
         class_index <= static_cast<float>(kMaxDetectionClassIndex) &&
         std::floor(class_index) == class_index;
}

/// Returns whether one finite box is ordered inside normalized coordinates.
inline bool valid_normalized_box(double left, double top, double right,
                                 double bottom) noexcept {
  return std::isfinite(left) && std::isfinite(top) && std::isfinite(right) &&
         std::isfinite(bottom) && left >= 0.0 && top >= 0.0 && right <= 1.0 &&
         bottom <= 1.0 && left <= right && top <= bottom;
}

/// Returns whether every field in one detection row is finite.
inline bool finite_detection_fields(const float* fields,
                                    std::size_t count) noexcept {
  if (fields == nullptr) {
    return false;
  }
  for (std::size_t field = 0; field < count; ++field) {
    if (!std::isfinite(fields[field])) {
      return false;
    }
  }
  return true;
}

/// Supported normalized six-field box encodings.
enum class DnnDetectionBoxEncoding {
  Xyxy,        ///< Left, top, right, and bottom.
  CenterXywh,  ///< Center x, center y, width, and height.
  TopLeftXywh  ///< Left, top, width, and height.
};

/// Converts one row's first four box coordinates into normalized XYXY.
inline bool decode_normalized_detection_box(
    const float* fields, DnnDetectionBoxEncoding encoding, double& left,
    double& top, double& right, double& bottom) noexcept {
  if (fields == nullptr) {
    return false;
  }
  if (encoding == DnnDetectionBoxEncoding::Xyxy) {
    left = fields[0];
    top = fields[1];
    right = fields[2];
    bottom = fields[3];
    return valid_normalized_box(left, top, right, bottom);
  }
  if (fields[2] <= 0.0F || fields[2] > 1.0F || fields[3] <= 0.0F ||
      fields[3] > 1.0F) {
    return false;
  }
  if (encoding == DnnDetectionBoxEncoding::CenterXywh) {
    const double half_width = static_cast<double>(fields[2]) / 2.0;
    const double half_height = static_cast<double>(fields[3]) / 2.0;
    left = static_cast<double>(fields[0]) - half_width;
    top = static_cast<double>(fields[1]) - half_height;
    right = static_cast<double>(fields[0]) + half_width;
    bottom = static_cast<double>(fields[1]) + half_height;
  } else {
    left = fields[0];
    top = fields[1];
    right = static_cast<double>(fields[0]) + fields[2];
    bottom = static_cast<double>(fields[1]) + fields[3];
  }
  return valid_normalized_box(left, top, right, bottom);
}

/// Validates a rank-two or single-batch rank-three six-field layout.
inline bool normalized_detection_shape(const int* dimensions,
                                       std::size_t rank,
                                       std::size_t& rows) noexcept {
  if (dimensions == nullptr || (rank != 2 && rank != 3)) {
    return false;
  }
  const int row_count = rank == 2 ? dimensions[0] : dimensions[1];
  const int field_count = rank == 2 ? dimensions[1] : dimensions[2];
  if ((rank == 3 && dimensions[0] != 1) || row_count <= 0 ||
      row_count > static_cast<int>(kMaxDetectionRowCount) ||
      field_count != 6) {
    return false;
  }
  rows = static_cast<std::size_t>(row_count);
  return true;
}

/// Supported memory layouts for variable-class detection tensors.
enum class DnnDetectionTensorLayout {
  RowMajor,    ///< Each detection row stores all fields contiguously.
  FieldMajor,  ///< Each field stores values for all rows contiguously.
};

/// Validates one rank-two or single-batch variable-class detection layout.
inline bool normalized_variable_class_detection_shape(
    const int* dimensions, std::size_t rank, std::size_t fixed_fields,
    DnnDetectionTensorLayout layout, std::size_t& rows,
    std::size_t& classes) noexcept {
  if (dimensions == nullptr || (rank != 2 && rank != 3)) {
    return false;
  }
  if (fixed_fields >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  int row_count = 0;
  int field_count = 0;
  if (layout == DnnDetectionTensorLayout::RowMajor) {
    row_count = rank == 2 ? dimensions[0] : dimensions[1];
    field_count = rank == 2 ? dimensions[1] : dimensions[2];
  } else if (layout == DnnDetectionTensorLayout::FieldMajor) {
    field_count = rank == 2 ? dimensions[0] : dimensions[1];
    row_count = rank == 2 ? dimensions[1] : dimensions[2];
  } else {
    return false;
  }
  const int fixed_field_count = static_cast<int>(fixed_fields);
  if ((rank == 3 && dimensions[0] != 1) || row_count <= 0 ||
      row_count > static_cast<int>(kMaxDetectionRowCount) ||
      field_count <= fixed_field_count ||
      field_count - fixed_field_count >
          static_cast<int>(kMaxObjectnessClassCount)) {
    return false;
  }
  rows = static_cast<std::size_t>(row_count);
  classes = static_cast<std::size_t>(field_count - fixed_field_count);
  return true;
}

/// Validates a variable-class objectness detection layout.
inline bool normalized_objectness_detection_shape(
    const int* dimensions, std::size_t rank, std::size_t& rows,
    std::size_t& classes) noexcept {
  return normalized_variable_class_detection_shape(
      dimensions, rank, 5, DnnDetectionTensorLayout::RowMajor, rows, classes);
}

/// Validates a field-major variable-class objectness detection layout.
inline bool normalized_transposed_objectness_detection_shape(
    const int* dimensions, std::size_t rank, std::size_t& rows,
    std::size_t& classes) noexcept {
  return normalized_variable_class_detection_shape(
      dimensions, rank, 5, DnnDetectionTensorLayout::FieldMajor, rows,
      classes);
}

/// Validates a variable-class score-only detection layout.
inline bool normalized_class_score_detection_shape(
    const int* dimensions, std::size_t rank, std::size_t& rows,
    std::size_t& classes) noexcept {
  return normalized_variable_class_detection_shape(
      dimensions, rank, 4, DnnDetectionTensorLayout::RowMajor, rows, classes);
}

/// Validates a field-major variable-class score-only detection layout.
inline bool normalized_transposed_class_score_detection_shape(
    const int* dimensions, std::size_t rank, std::size_t& rows,
    std::size_t& classes) noexcept {
  return normalized_variable_class_detection_shape(
      dimensions, rank, 4, DnnDetectionTensorLayout::FieldMajor, rows,
      classes);
}

/// Validates and projects one normalized six-field detection collection.
inline bool normalized_six_field_detections(
    const float* values, std::size_t count, std::size_t rows,
    double minimum_confidence, std::size_t max_detections,
    DnnDetectionBoxEncoding encoding,
    std::vector<DnnDetection>& detections) {
  constexpr std::size_t kFieldsPerDetection = 6;
  if (values == nullptr || rows == 0 || rows > kMaxDetectionRowCount ||
      count != rows * kFieldsPerDetection ||
      !detection_parameters(minimum_confidence, max_detections)) {
    return false;
  }
  std::vector<DnnDetection> selected;
  selected.reserve(rows < max_detections ? rows : max_detections);
  for (std::size_t row = 0; row < rows; ++row) {
    const float* fields = values + row * kFieldsPerDetection;
    double left = 0.0;
    double top = 0.0;
    double right = 0.0;
    double bottom = 0.0;
    if (!finite_detection_fields(fields, kFieldsPerDetection) ||
        !decode_normalized_detection_box(fields, encoding, left, top, right,
                                         bottom) ||
        fields[4] < 0.0F || fields[4] > 1.0F ||
        !valid_detection_class(fields[5])) {
      return false;
    }
    if (fields[4] < minimum_confidence) {
      continue;
    }
    selected.push_back(DnnDetection{
        static_cast<std::int64_t>(fields[5]), static_cast<double>(fields[4]),
        left, top, right, bottom, row});
  }
  canonicalize_detections(selected, max_detections);
  detections = std::move(selected);
  return true;
}

/// Validates, filters, orders, and bounds normalized XYXY detection rows.
inline bool normalized_xyxy_detections(
    const float* values, std::size_t count, std::size_t rows,
    double minimum_confidence, std::size_t max_detections,
    std::vector<DnnDetection>& detections) {
  return normalized_six_field_detections(
      values, count, rows, minimum_confidence, max_detections,
      DnnDetectionBoxEncoding::Xyxy, detections);
}

/// Converts normalized center-XYWH rows into ordered bounded XYXY detections.
inline bool normalized_center_xywh_detections(
    const float* values, std::size_t count, std::size_t rows,
    double minimum_confidence, std::size_t max_detections,
    std::vector<DnnDetection>& detections) {
  return normalized_six_field_detections(
      values, count, rows, minimum_confidence, max_detections,
      DnnDetectionBoxEncoding::CenterXywh, detections);
}

/// Converts normalized top-left-XYWH rows into ordered bounded XYXY detections.
inline bool normalized_top_left_xywh_detections(
    const float* values, std::size_t count, std::size_t rows,
    double minimum_confidence, std::size_t max_detections,
    std::vector<DnnDetection>& detections) {
  return normalized_six_field_detections(
      values, count, rows, minimum_confidence, max_detections,
      DnnDetectionBoxEncoding::TopLeftXywh, detections);
}

/// Supported confidence encodings for variable-class detection rows.
enum class DnnDetectionConfidenceEncoding {
  ClassProbability,   ///< Confidence is the maximum class probability.
  ObjectnessProduct,  ///< Confidence multiplies objectness and class probability.
};

/// Decodes variable-class center-XYWH rows into canonical detections.
inline bool normalized_variable_class_detections(
    const float* values, std::size_t count, std::size_t rows,
    std::size_t classes, double minimum_confidence,
    std::size_t max_detections, DnnDetectionConfidenceEncoding encoding,
    DnnDetectionTensorLayout layout,
    std::vector<DnnDetection>& detections) {
  if (values == nullptr || rows == 0 || rows > kMaxDetectionRowCount ||
      classes == 0 || classes > kMaxObjectnessClassCount ||
      !detection_parameters(minimum_confidence, max_detections)) {
    return false;
  }
  std::size_t class_field_offset = 0;
  bool has_objectness = false;
  if (encoding == DnnDetectionConfidenceEncoding::ClassProbability) {
    class_field_offset = 4;
  } else if (encoding == DnnDetectionConfidenceEncoding::ObjectnessProduct) {
    class_field_offset = 5;
    has_objectness = true;
  } else {
    return false;
  }
  if (layout != DnnDetectionTensorLayout::RowMajor &&
      layout != DnnDetectionTensorLayout::FieldMajor) {
    return false;
  }
  const std::size_t fields_per_detection = classes + class_field_offset;
  if (count != rows * fields_per_detection) {
    return false;
  }
  std::vector<DnnDetection> selected;
  selected.reserve(rows < max_detections ? rows : max_detections);
  for (std::size_t row = 0; row < rows; ++row) {
    const auto field = [values, rows, fields_per_detection, row,
                        layout](std::size_t field_index) {
      if (layout == DnnDetectionTensorLayout::RowMajor) {
        return values[row * fields_per_detection + field_index];
      }
      return values[field_index * rows + row];
    };
    bool finite = true;
    for (std::size_t field_index = 0; field_index < fields_per_detection;
         ++field_index) {
      finite = finite && std::isfinite(field(field_index));
    }
    const float box_fields[] = {field(0), field(1), field(2), field(3)};
    double left = 0.0;
    double top = 0.0;
    double right = 0.0;
    double bottom = 0.0;
    if (!finite ||
        !decode_normalized_detection_box(
            box_fields, DnnDetectionBoxEncoding::CenterXywh, left, top, right,
            bottom)) {
      return false;
    }
    float objectness = 1.0F;
    if (has_objectness) {
      objectness = field(4);
      if (objectness < 0.0F || objectness > 1.0F) {
        return false;
      }
    }
    std::size_t class_index = 0;
    float class_probability = field(class_field_offset);
    for (std::size_t class_index_offset = 0; class_index_offset < classes;
         ++class_index_offset) {
      const float probability = field(class_field_offset + class_index_offset);
      if (probability < 0.0F || probability > 1.0F) {
        return false;
      }
      if (probability > class_probability) {
        class_probability = probability;
        class_index = class_index_offset;
      }
    }
    const double confidence =
        static_cast<double>(objectness) * class_probability;
    if (confidence < minimum_confidence) {
      continue;
    }
    selected.push_back(DnnDetection{static_cast<std::int64_t>(class_index),
                                    confidence, left, top, right, bottom, row});
  }
  canonicalize_detections(selected, max_detections);
  detections = std::move(selected);
  return true;
}

/// Decodes center-XYWH rows with objectness and class probabilities.
inline bool normalized_objectness_detections(
    const float* values, std::size_t count, std::size_t rows,
    std::size_t classes, double minimum_confidence,
    std::size_t max_detections, std::vector<DnnDetection>& detections) {
  return normalized_variable_class_detections(
      values, count, rows, classes, minimum_confidence, max_detections,
      DnnDetectionConfidenceEncoding::ObjectnessProduct,
      DnnDetectionTensorLayout::RowMajor, detections);
}

/// Decodes field-major center-XYWH rows with objectness and class probabilities.
inline bool normalized_transposed_objectness_detections(
    const float* values, std::size_t count, std::size_t rows,
    std::size_t classes, double minimum_confidence,
    std::size_t max_detections, std::vector<DnnDetection>& detections) {
  return normalized_variable_class_detections(
      values, count, rows, classes, minimum_confidence, max_detections,
      DnnDetectionConfidenceEncoding::ObjectnessProduct,
      DnnDetectionTensorLayout::FieldMajor, detections);
}

/// Decodes center-XYWH rows with class probabilities and no objectness field.
inline bool normalized_class_score_detections(
    const float* values, std::size_t count, std::size_t rows,
    std::size_t classes, double minimum_confidence,
    std::size_t max_detections, std::vector<DnnDetection>& detections) {
  return normalized_variable_class_detections(
      values, count, rows, classes, minimum_confidence, max_detections,
      DnnDetectionConfidenceEncoding::ClassProbability,
      DnnDetectionTensorLayout::RowMajor, detections);
}

/// Decodes field-major center-XYWH rows with class probabilities.
inline bool normalized_transposed_class_score_detections(
    const float* values, std::size_t count, std::size_t rows,
    std::size_t classes, double minimum_confidence,
    std::size_t max_detections, std::vector<DnnDetection>& detections) {
  return normalized_variable_class_detections(
      values, count, rows, classes, minimum_confidence, max_detections,
      DnnDetectionConfidenceEncoding::ClassProbability,
      DnnDetectionTensorLayout::FieldMajor, detections);
}

/// Validates, filters, orders, and bounds normalized SSD detection rows.
inline bool ssd_detections(const float* values, std::size_t count,
                           std::size_t rows, double threshold,
                           std::size_t max_detections,
                           std::vector<DnnDetection>& detections) {
  constexpr std::size_t kFieldsPerDetection = 7;
  if (values == nullptr || rows == 0 || rows > kMaxDetectionRowCount ||
      count != rows * kFieldsPerDetection ||
      !detection_parameters(threshold, max_detections)) {
    return false;
  }
  std::vector<DnnDetection> selected;
  selected.reserve(rows < max_detections ? rows : max_detections);
  for (std::size_t row = 0; row < rows; ++row) {
    const float* fields = values + row * kFieldsPerDetection;
    if (!finite_detection_fields(fields, kFieldsPerDetection) ||
        fields[0] != 0.0F || !valid_detection_class(fields[1]) ||
        fields[2] < 0.0F || fields[2] > 1.0F ||
        !valid_normalized_box(fields[3], fields[4], fields[5], fields[6])) {
      return false;
    }
    if (fields[1] == 0.0F || fields[2] < threshold) {
      continue;
    }
    selected.push_back(DnnDetection{
        static_cast<std::int64_t>(fields[1]), static_cast<double>(fields[2]),
        static_cast<double>(fields[3]), static_cast<double>(fields[4]),
        static_cast<double>(fields[5]), static_cast<double>(fields[6]), row});
  }
  canonicalize_detections(selected, max_detections);
  detections = std::move(selected);
  return true;
}

/// Validates class-aware non-maximum-suppression policy parameters.
inline bool nms_parameters(double overlap_threshold,
                           std::size_t max_detections) noexcept {
  return std::isfinite(overlap_threshold) && overlap_threshold >= 0.0 &&
         overlap_threshold <= 1.0 && max_detections > 0 &&
         max_detections <= kMaxDetectionCount;
}

}  // namespace dnn_validation
}  // namespace terlan_opencv

#endif
