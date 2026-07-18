#include "opencv_dnn_detection_validation.hpp"

#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

using terlan_opencv::dnn_validation::DnnDetection;
using terlan_opencv::dnn_validation::DnnDetectionConfidenceEncoding;
using terlan_opencv::dnn_validation::DnnDetectionTensorLayout;
using terlan_opencv::dnn_validation::kMaxDetectionCount;
using terlan_opencv::dnn_validation::normalized_class_score_detection_shape;
using terlan_opencv::dnn_validation::normalized_class_score_detections;
using terlan_opencv::dnn_validation::normalized_objectness_detection_shape;
using terlan_opencv::dnn_validation::normalized_objectness_detections;
using terlan_opencv::dnn_validation::normalized_transposed_class_score_detection_shape;
using terlan_opencv::dnn_validation::normalized_transposed_class_score_detections;
using terlan_opencv::dnn_validation::normalized_transposed_objectness_detection_shape;
using terlan_opencv::dnn_validation::normalized_transposed_objectness_detections;
using terlan_opencv::dnn_validation::normalized_variable_class_detection_shape;
using terlan_opencv::dnn_validation::normalized_variable_class_detections;

int main() {
  std::size_t rows = 77;
  std::size_t classes = 88;
  const int rank_two[] = {3, 7};
  assert(normalized_objectness_detection_shape(rank_two, 2, rows, classes));
  assert(rows == 3 && classes == 2);
  const int transposed_rank_three[] = {1, 6, 3};
  assert(normalized_transposed_class_score_detection_shape(
      transposed_rank_three, 3, rows, classes));
  assert(rows == 3 && classes == 2);
  const int transposed_objectness_rank_three[] = {1, 7, 3};
  assert(normalized_transposed_objectness_detection_shape(
      transposed_objectness_rank_three, 3, rows, classes));
  assert(rows == 3 && classes == 2);
  const int transposed_objectness_rank_two[] = {7, 3};
  assert(normalized_transposed_objectness_detection_shape(
      transposed_objectness_rank_two, 2, rows, classes));
  assert(rows == 3 && classes == 2);
  const auto invalid_layout = static_cast<DnnDetectionTensorLayout>(99);
  rows = 77;
  classes = 88;
  assert(!normalized_variable_class_detection_shape(
      transposed_rank_three, 3, 4, invalid_layout, rows, classes));
  assert(rows == 77 && classes == 88);
  const int score_rank_two[] = {3, 6};
  assert(normalized_class_score_detection_shape(score_rank_two, 2, rows,
                                                classes));
  assert(rows == 3 && classes == 2);
  const int rank_three[] = {1, 3, 7};
  assert(normalized_objectness_detection_shape(rank_three, 3, rows, classes));
  assert(rows == 3 && classes == 2);

  const int wrong_batch[] = {2, 3, 7};
  const int no_classes[] = {3, 5};
  const int too_many_classes[] = {1, 1, 10006};
  const int hostile_field_count[] = {
      3, std::numeric_limits<int>::min(),
  };
  rows = 77;
  classes = 88;
  assert(!normalized_objectness_detection_shape(nullptr, 2, rows, classes));
  assert(!normalized_objectness_detection_shape(rank_two, 1, rows, classes));
  assert(!normalized_objectness_detection_shape(wrong_batch, 3, rows, classes));
  assert(!normalized_objectness_detection_shape(no_classes, 2, rows, classes));
  assert(!normalized_objectness_detection_shape(hostile_field_count, 2, rows,
                                                 classes));
  assert(!normalized_objectness_detection_shape(too_many_classes, 3, rows,
                                                 classes));
  assert(rows == 77 && classes == 88);
  const int no_transposed_classes[] = {5, 3};
  assert(!normalized_transposed_objectness_detection_shape(
      no_transposed_classes, 2, rows, classes));

  const float values[] = {
      0.25F, 0.25F, 0.30F, 0.30F, 0.80F, 0.80F, 0.20F,
      0.40F, 0.40F, 0.40F, 0.40F, 0.90F, 0.10F, 0.90F,
      0.80F, 0.80F, 0.20F, 0.20F, 0.50F, 0.50F, 0.50F,
  };
  std::vector<DnnDetection> detections;
  assert(normalized_objectness_detections(values, 21, 3, 2, 0.25, 3,
                                          detections));
  assert(detections.size() == 3);
  assert(detections[0].class_index == 1 &&
         std::abs(detections[0].confidence - 0.81) < 1.0e-6);
  assert(detections[1].class_index == 0 &&
         std::abs(detections[1].confidence - 0.64) < 1.0e-6);
  assert(detections[2].class_index == 0 &&
         std::abs(detections[2].confidence - 0.25) < 1.0e-6);
  assert(std::abs(detections[0].left - 0.2) < 1.0e-6);
  assert(std::abs(detections[0].top - 0.2) < 1.0e-6);
  assert(std::abs(detections[0].right - 0.6) < 1.0e-6);
  assert(std::abs(detections[0].bottom - 0.6) < 1.0e-6);

  assert(normalized_objectness_detections(values, 21, 3, 2, 0.64, 1,
                                          detections));
  assert(detections.size() == 1 && detections[0].class_index == 1);
  assert(normalized_objectness_detections(values, 21, 3, 2, 1.0, 3,
                                          detections));
  assert(detections.empty());

  detections.push_back(DnnDetection{9, 0.9, 0.0, 0.0, 1.0, 1.0, 9});
  const auto unchanged = detections;
  assert(!normalized_objectness_detections(nullptr, 21, 3, 2, 0.0, 3,
                                           detections));
  assert(!normalized_objectness_detections(values, 20, 3, 2, 0.0, 3,
                                           detections));
  assert(!normalized_objectness_detections(values, 21, 0, 2, 0.0, 3,
                                           detections));
  assert(!normalized_objectness_detections(values, 21, 3, 0, 0.0, 3,
                                           detections));
  assert(!normalized_objectness_detections(
      values, 21, 3, std::numeric_limits<std::size_t>::max(), 0.0, 3,
      detections));
  assert(!normalized_objectness_detections(values, 21, 3, 2, -0.1, 3,
                                           detections));
  assert(!normalized_objectness_detections(values, 21, 3, 2, 1.1, 3,
                                           detections));
  assert(!normalized_objectness_detections(values, 21, 3, 2, 0.0, 0,
                                           detections));
  assert(!normalized_objectness_detections(values, 21, 3, 2, 0.0,
                                           kMaxDetectionCount + 1,
                                           detections));
  assert(detections.size() == unchanged.size());
  assert(detections[0].class_index == unchanged[0].class_index);

  const float class_scores[] = {
      0.25F, 0.25F, 0.30F, 0.30F, 0.80F, 0.20F,
      0.40F, 0.40F, 0.40F, 0.40F, 0.10F, 0.90F,
      0.80F, 0.80F, 0.20F, 0.20F, 0.50F, 0.50F,
  };
  detections.clear();
  assert(normalized_class_score_detections(class_scores, 18, 3, 2, 0.5, 3,
                                           detections));
  assert(detections.size() == 3);
  assert(detections[0].class_index == 1 &&
         std::abs(detections[0].confidence - 0.9) < 1.0e-6);
  assert(detections[1].class_index == 0 &&
         std::abs(detections[1].confidence - 0.8) < 1.0e-6);
  assert(detections[2].class_index == 0 &&
         std::abs(detections[2].confidence - 0.5) < 1.0e-6);
  assert(normalized_class_score_detections(class_scores, 18, 3, 2, 0.8, 3,
                                           detections));
  assert(detections.size() == 2 && detections[0].class_index == 1 &&
         detections[1].class_index == 0);
  const auto unchanged_class_scores = detections;
  const auto invalid_encoding =
      static_cast<DnnDetectionConfidenceEncoding>(99);
  assert(!normalized_variable_class_detections(
      class_scores, 18, 3, 2, 0.0, 3, invalid_encoding,
      DnnDetectionTensorLayout::RowMajor, detections));
  assert(detections[0].class_index ==
         unchanged_class_scores[0].class_index);
  const float bad_class_scores[] = {
      0.5F, 0.5F, 0.5F, 0.5F, 0.9F, -0.1F,
  };
  assert(!normalized_class_score_detections(
      bad_class_scores, 6, 1, 2, 0.0, 3, detections));
  assert(detections.size() == unchanged_class_scores.size());

  const float transposed_class_scores[] = {
      0.25F, 0.40F, 0.80F, 0.25F, 0.40F, 0.80F,
      0.30F, 0.40F, 0.20F, 0.30F, 0.40F, 0.20F,
      0.80F, 0.10F, 0.50F, 0.20F, 0.90F, 0.50F,
  };
  assert(normalized_transposed_class_score_detections(
      transposed_class_scores, 18, 3, 2, 0.5, 3, detections));
  assert(detections.size() == 3 && detections[0].class_index == 1 &&
         detections[1].class_index == 0 && detections[2].class_index == 0);
  assert(std::abs(detections[0].confidence - 0.9) < 1.0e-6);
  assert(std::abs(detections[0].left - 0.2) < 1.0e-6);
  assert(std::abs(detections[0].right - 0.6) < 1.0e-6);
  const auto unchanged_transposed = detections;
  const float bad_transposed_class_scores[] = {
      0.25F, 0.40F, 0.80F, 0.25F, 0.40F, 0.80F,
      0.30F, 0.40F, 0.20F, 0.30F, 0.40F, 0.20F,
      0.80F, 0.10F, 0.50F, 0.20F, 0.90F, -0.10F,
  };
  assert(!normalized_transposed_class_score_detections(
      bad_transposed_class_scores, 18, 3, 2, 0.0, 3, detections));
  assert(detections.size() == unchanged_transposed.size());

  const float transposed_objectness[] = {
      0.25F, 0.40F, 0.80F, 0.25F, 0.40F, 0.80F, 0.30F,
      0.40F, 0.20F, 0.30F, 0.40F, 0.20F, 0.80F, 0.90F,
      0.50F, 0.80F, 0.10F, 0.50F, 0.20F, 0.90F, 0.50F,
  };
  assert(normalized_transposed_objectness_detections(
      transposed_objectness, 21, 3, 2, 0.25, 3, detections));
  assert(detections.size() == 3 && detections[0].class_index == 1 &&
         detections[1].class_index == 0 && detections[2].class_index == 0);
  assert(std::abs(detections[0].confidence - 0.81) < 1.0e-6);
  assert(std::abs(detections[1].confidence - 0.64) < 1.0e-6);
  assert(std::abs(detections[2].confidence - 0.25) < 1.0e-6);
  const auto unchanged_transposed_objectness = detections;
  const float bad_transposed_objectness[] = {
      0.25F, 0.40F, 0.80F, 0.25F, 0.40F, 0.80F, 0.30F,
      0.40F, 0.20F, 0.30F, 0.40F, 0.20F, 0.80F, 1.10F,
      0.50F, 0.80F, 0.10F, 0.50F, 0.20F, 0.90F, 0.50F,
  };
  assert(!normalized_transposed_objectness_detections(
      bad_transposed_objectness, 21, 3, 2, 0.0, 3, detections));
  assert(detections.size() == unchanged_transposed_objectness.size());

  const float zero_width[] = {0.5F, 0.5F, 0.0F, 0.5F, 0.9F, 0.9F};
  const float outside[] = {0.1F, 0.5F, 0.4F, 0.5F, 0.9F, 0.9F};
  const float bad_objectness[] = {0.5F, 0.5F, 0.5F, 0.5F, 1.1F, 0.9F};
  const float bad_probability[] = {0.5F, 0.5F, 0.5F, 0.5F, 0.9F, -0.1F};
  const float nonfinite[] = {
      0.5F, 0.5F, 0.5F, 0.5F, 0.9F,
      std::numeric_limits<float>::quiet_NaN(),
  };
  for (const float* malformed :
       {zero_width, outside, bad_objectness, bad_probability, nonfinite}) {
    assert(!normalized_objectness_detections(malformed, 6, 1, 1, 0.0, 1,
                                             detections));
  }
  assert(detections.size() == unchanged_transposed.size());
  return 0;
}
