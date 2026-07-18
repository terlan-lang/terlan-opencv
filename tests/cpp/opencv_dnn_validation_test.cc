#include "opencv_dnn_validation.hpp"

#include <cassert>
#include <cstdint>
#include <limits>

using terlan_opencv::dnn_validation::kMaxClassificationCount;
using terlan_opencv::dnn_validation::kMaxTopClassificationCount;
using terlan_opencv::dnn_validation::kMaxOutputSliceValueCount;
using terlan_opencv::dnn_validation::kMaxClassMaskCount;
using terlan_opencv::dnn_validation::kMaxMaskPixelCount;
using terlan_opencv::dnn_validation::class_index_mask;
using terlan_opencv::dnn_validation::class_confidence_mask;
using terlan_opencv::dnn_validation::softmax_class_confidence_mask;
using terlan_opencv::dnn_validation::softmax_class_probability_mask;
using terlan_opencv::dnn_validation::softmax_class_mask;
using terlan_opencv::dnn_validation::sigmoid_class_probability_mask;
using terlan_opencv::dnn_validation::sigmoid_class_mask;
using terlan_opencv::dnn_validation::sigmoid_any_class_mask;
using terlan_opencv::dnn_validation::sigmoid_class_confidence_mask;
using terlan_opencv::dnn_validation::DnnDetection;
using terlan_opencv::dnn_validation::DnnNormalization;
using terlan_opencv::dnn_validation::grayscale_normalization_parameters;
using terlan_opencv::dnn_validation::kMaxDetectionCount;
using terlan_opencv::dnn_validation::kMaxDetectionRowCount;
using terlan_opencv::dnn_validation::nms_parameters;
using terlan_opencv::dnn_validation::normalization_parameters;
using terlan_opencv::dnn_validation::normalized_center_xywh_detections;
using terlan_opencv::dnn_validation::normalized_detection_shape;
using terlan_opencv::dnn_validation::normalized_top_left_xywh_detections;
using terlan_opencv::dnn_validation::normalized_xyxy_detections;
using terlan_opencv::dnn_validation::output_slice_bounds;
using terlan_opencv::dnn_validation::rgb_normalization_parameters;
using terlan_opencv::dnn_validation::ssd_detections;
using terlan_opencv::dnn_validation::channel_mask;
using terlan_opencv::dnn_validation::classification_candidate_indices;
using terlan_opencv::dnn_validation::softmax_classifications;
using terlan_opencv::dnn_validation::sigmoid_classifications;
using terlan_opencv::dnn_validation::sigmoid_classification_candidates;
using terlan_opencv::dnn_validation::stable_sigmoid;
using terlan_opencv::dnn_validation::top_classification;
using terlan_opencv::dnn_validation::top_classification_indices;

int main() {
  const double rgb_means[] = {-1.0, 0.5, 1.0};
  const double rgb_scales[] = {2.0, 4.0, 8.0};
  DnnNormalization normalization{{9.0, 9.0, 9.0}, {8.0, 8.0, 8.0}};
  assert(rgb_normalization_parameters(rgb_means, 3, rgb_scales, 3,
                                      normalization));
  assert(normalization.byte_means[0] == -255.0 &&
         normalization.byte_means[1] == 127.5 &&
         normalization.byte_means[2] == 255.0);
  assert(normalization.byte_scales[0] == 2.0 / 255.0 &&
         normalization.byte_scales[1] == 4.0 / 255.0 &&
         normalization.byte_scales[2] == 8.0 / 255.0);
  const DnnNormalization unchanged = normalization;
  const double zero_scale[] = {1.0, 0.0, 1.0};
  const double negative_scale[] = {1.0, -1.0, 1.0};
  const double nonfinite_mean[] = {
      0.0, std::numeric_limits<double>::quiet_NaN(), 0.0};
  const double nonfinite_scale[] = {
      1.0, std::numeric_limits<double>::infinity(), 1.0};
  const double excessive_mean[] = {
      0.0, std::numeric_limits<double>::max(), 0.0};
  const double excessive_scale[] = {
      1.0, std::numeric_limits<double>::max(), 1.0};
  const double underflow_scale[] = {
      1.0, std::numeric_limits<double>::denorm_min(), 1.0};
  assert(!rgb_normalization_parameters(nullptr, 3, rgb_scales, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, nullptr, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 2, rgb_scales, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, rgb_scales, 4,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, zero_scale, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, negative_scale, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(nonfinite_mean, 3, rgb_scales, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, nonfinite_scale, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(excessive_mean, 3, rgb_scales, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, excessive_scale, 3,
                                       normalization));
  assert(!rgb_normalization_parameters(rgb_means, 3, underflow_scale, 3,
                                       normalization));
  assert(normalization.byte_means == unchanged.byte_means &&
         normalization.byte_scales == unchanged.byte_scales);

  const double grayscale_mean[] = {0.5};
  const double grayscale_scale[] = {4.0};
  DnnNormalization grayscale_normalization{{7.0, 7.0, 7.0},
                                            {6.0, 6.0, 6.0}};
  assert(grayscale_normalization_parameters(
      grayscale_mean, 1, grayscale_scale, 1, grayscale_normalization));
  assert(grayscale_normalization.byte_means[0] == 127.5 &&
         grayscale_normalization.byte_means[1] == 0.0 &&
         grayscale_normalization.byte_means[2] == 0.0);
  assert(grayscale_normalization.byte_scales[0] == 4.0 / 255.0 &&
         grayscale_normalization.byte_scales[1] == 0.0 &&
         grayscale_normalization.byte_scales[2] == 0.0);
  const DnnNormalization unchanged_grayscale = grayscale_normalization;
  const double grayscale_zero_scale[] = {0.0};
  const double grayscale_nonfinite_mean[] = {
      std::numeric_limits<double>::quiet_NaN()};
  const double grayscale_nonfinite_scale[] = {
      std::numeric_limits<double>::infinity()};
  const double grayscale_excessive_mean[] = {
      std::numeric_limits<double>::max()};
  const double grayscale_excessive_scale[] = {
      std::numeric_limits<double>::max()};
  const double grayscale_underflow_scale[] = {
      std::numeric_limits<double>::denorm_min()};
  assert(!grayscale_normalization_parameters(
      nullptr, 1, grayscale_scale, 1, grayscale_normalization));
  assert(!grayscale_normalization_parameters(
      grayscale_mean, 1, nullptr, 1, grayscale_normalization));
  assert(!grayscale_normalization_parameters(
      grayscale_mean, 0, grayscale_scale, 1, grayscale_normalization));
  assert(!grayscale_normalization_parameters(
      grayscale_mean, 1, grayscale_scale, 2, grayscale_normalization));
  assert(!grayscale_normalization_parameters(
      grayscale_mean, 1, grayscale_zero_scale, 1, grayscale_normalization));
  assert(!grayscale_normalization_parameters(grayscale_nonfinite_mean, 1,
                                             grayscale_scale, 1,
                                             grayscale_normalization));
  assert(!grayscale_normalization_parameters(grayscale_mean, 1,
                                             grayscale_nonfinite_scale, 1,
                                             grayscale_normalization));
  assert(!grayscale_normalization_parameters(grayscale_excessive_mean, 1,
                                             grayscale_scale, 1,
                                             grayscale_normalization));
  assert(!grayscale_normalization_parameters(grayscale_mean, 1,
                                             grayscale_excessive_scale, 1,
                                             grayscale_normalization));
  assert(!grayscale_normalization_parameters(grayscale_mean, 1,
                                             grayscale_underflow_scale, 1,
                                             grayscale_normalization));
  assert(!normalization_parameters(grayscale_mean, 1, grayscale_scale, 1, 2,
                                   grayscale_normalization));
  assert(grayscale_normalization.byte_means ==
             unchanged_grayscale.byte_means &&
         grayscale_normalization.byte_scales ==
             unchanged_grayscale.byte_scales);

  std::size_t slice_begin = 91;
  std::size_t slice_end = 92;
  assert(output_slice_bounds(12, 3, 4, slice_begin, slice_end));
  assert(slice_begin == 3 && slice_end == 7);
  assert(output_slice_bounds(12, 10, 99, slice_begin, slice_end));
  assert(slice_begin == 10 && slice_end == 12);
  assert(output_slice_bounds(12, 12, 1, slice_begin, slice_end));
  assert(slice_begin == 12 && slice_end == 12);
  slice_begin = 91;
  slice_end = 92;
  assert(!output_slice_bounds(0, 0, 1, slice_begin, slice_end));
  assert(!output_slice_bounds(12, -1, 1, slice_begin, slice_end));
  assert(!output_slice_bounds(12, 0, 0, slice_begin, slice_end));
  assert(!output_slice_bounds(12, 0, -1, slice_begin, slice_end));
  assert(!output_slice_bounds(
      12, 0, static_cast<std::int64_t>(kMaxOutputSliceValueCount) + 1,
      slice_begin, slice_end));
  assert(!output_slice_bounds(12, 13, 1, slice_begin, slice_end));
  assert(!output_slice_bounds(12, std::numeric_limits<std::int64_t>::max(),
                              1, slice_begin, slice_end));
  assert(slice_begin == 91 && slice_end == 92);

  std::int64_t index = -1;
  double score = -1.0;
  const float one[] = {1.0F};
  assert(!top_classification(nullptr, 2, index, score));
  assert(!top_classification(one, 0, index, score));
  assert(!top_classification(one, 1, index, score));
  assert(!top_classification(one, kMaxClassificationCount + 1, index, score));
  assert(index == -1 && score == -1.0);

  const float unique[] = {-4.0F, -1.0F, 3.5F, 2.0F};
  assert(top_classification(unique, 4, index, score));
  assert(index == 2 && score == 3.5);

  const float negative[] = {-8.0F, -2.0F, -5.0F};
  assert(top_classification(negative, 3, index, score));
  assert(index == 1 && score == -2.0);

  const float tie[] = {7.0F, 7.0F, 6.0F, 7.0F};
  assert(top_classification(tie, 4, index, score));
  assert(index == 0 && score == 7.0);

  const float nan_first[] = {std::numeric_limits<float>::quiet_NaN(), 1.0F};
  const float nan_later[] = {1.0F, std::numeric_limits<float>::quiet_NaN()};
  const float positive_infinity[] = {
      1.0F, std::numeric_limits<float>::infinity()};
  const float negative_infinity[] = {
      1.0F, -std::numeric_limits<float>::infinity()};
  assert(!top_classification(nan_first, 2, index, score));
  assert(!top_classification(nan_later, 2, index, score));
  assert(!top_classification(positive_infinity, 2, index, score));
  assert(!top_classification(negative_infinity, 2, index, score));

  std::vector<std::size_t> ranked = {9};
  assert(top_classification_indices(unique, 4, 3, ranked));
  assert((ranked == std::vector<std::size_t>{2, 3, 1}));
  assert(top_classification_indices(tie, 4, 3, ranked));
  assert((ranked == std::vector<std::size_t>{0, 1, 3}));
  assert(top_classification_indices(negative, 3, 8, ranked));
  assert((ranked == std::vector<std::size_t>{1, 2, 0}));
  ranked = {9};
  assert(!top_classification_indices(nullptr, 4, 3, ranked));
  assert(!top_classification_indices(one, 1, 1, ranked));
  assert(!top_classification_indices(unique, 4, 0, ranked));
  assert(!top_classification_indices(unique, 4,
                                     kMaxTopClassificationCount + 1, ranked));
  assert(!top_classification_indices(nan_later, 2, 1, ranked));
  assert((ranked == std::vector<std::size_t>{9}));

  std::vector<std::size_t> candidates = {9};
  assert(classification_candidate_indices(unique, 4, -1.0, 4,
                                          candidates));
  assert((candidates == std::vector<std::size_t>{2, 3, 1}));
  assert(classification_candidate_indices(unique, 4, 3.5, 4, candidates));
  assert((candidates == std::vector<std::size_t>{2}));
  assert(classification_candidate_indices(unique, 4, 4.0, 4, candidates));
  assert(candidates.empty());
  assert(classification_candidate_indices(unique, 4, -100.0, 2,
                                          candidates));
  assert((candidates == std::vector<std::size_t>{2, 3}));
  assert(classification_candidate_indices(tie, 4, 7.0, 2, candidates));
  assert((candidates == std::vector<std::size_t>{0, 1}));
  candidates = {9};
  assert(!classification_candidate_indices(
      unique, 4, std::numeric_limits<double>::quiet_NaN(), 2, candidates));
  assert(!classification_candidate_indices(
      unique, 4, std::numeric_limits<double>::infinity(), 2, candidates));
  assert(!classification_candidate_indices(unique, 4, 0.0, 0, candidates));
  assert(!classification_candidate_indices(
      unique, 4, 0.0, kMaxTopClassificationCount + 1, candidates));
  assert(!classification_candidate_indices(nan_later, 2, 0.0, 1,
                                            candidates));

  std::vector<std::size_t> softmax_indices = {9};
  std::vector<double> softmax_probabilities = {9.0};
  assert(softmax_classifications(unique, 4, 3, softmax_indices,
                                 softmax_probabilities));
  assert((softmax_indices == std::vector<std::size_t>{2, 3, 1}));
  assert(softmax_probabilities.size() == 3);
  const double unique_total = std::exp(-7.5) + std::exp(-4.5) + 1.0 +
                              std::exp(-1.5);
  assert(std::abs(softmax_probabilities[0] - 1.0 / unique_total) < 1.0e-12);
  assert(std::abs(softmax_probabilities[1] - std::exp(-1.5) / unique_total) <
         1.0e-12);
  assert(std::abs(softmax_probabilities[2] - std::exp(-4.5) / unique_total) <
         1.0e-12);
  const float equal_logits[] = {7.0F, 7.0F, 7.0F};
  assert(softmax_classifications(equal_logits, 3, 9, softmax_indices,
                                 softmax_probabilities));
  assert((softmax_indices == std::vector<std::size_t>{0, 1, 2}));
  assert(softmax_probabilities.size() == 3);
  for (const double probability : softmax_probabilities) {
    assert(std::abs(probability - 1.0 / 3.0) < 1.0e-12);
  }
  const float extreme_logits[] = {std::numeric_limits<float>::max(),
                                  -std::numeric_limits<float>::max()};
  assert(softmax_classifications(extreme_logits, 2, 2, softmax_indices,
                                 softmax_probabilities));
  assert((softmax_indices == std::vector<std::size_t>{0, 1}));
  assert(softmax_probabilities[0] == 1.0 && softmax_probabilities[1] == 0.0);
  softmax_indices = {9};
  softmax_probabilities = {9.0};
  assert(!softmax_classifications(nullptr, 4, 2, softmax_indices,
                                  softmax_probabilities));
  assert(!softmax_classifications(one, 1, 1, softmax_indices,
                                  softmax_probabilities));
  assert(!softmax_classifications(unique, 4, 0, softmax_indices,
                                  softmax_probabilities));
  assert(!softmax_classifications(unique, 4,
                                  kMaxTopClassificationCount + 1,
                                  softmax_indices, softmax_probabilities));
  assert(!softmax_classifications(nan_later, 2, 1, softmax_indices,
                                  softmax_probabilities));
  assert((softmax_indices == std::vector<std::size_t>{9}));
  assert((softmax_probabilities == std::vector<double>{9.0}));
  assert((candidates == std::vector<std::size_t>{9}));

  std::vector<std::size_t> sigmoid_indices = {9};
  std::vector<double> sigmoid_probabilities = {9.0};
  assert(sigmoid_classifications(unique, 4, 3, sigmoid_indices,
                                 sigmoid_probabilities));
  assert((sigmoid_indices == std::vector<std::size_t>{2, 3, 1}));
  assert(sigmoid_probabilities.size() == 3);
  assert(std::abs(sigmoid_probabilities[0] - stable_sigmoid(3.5F)) <
         1.0e-12);
  assert(std::abs(sigmoid_probabilities[1] - stable_sigmoid(2.0F)) <
         1.0e-12);
  assert(std::abs(sigmoid_probabilities[2] - stable_sigmoid(-1.0F)) <
         1.0e-12);
  assert(sigmoid_classifications(equal_logits, 3, 9, sigmoid_indices,
                                 sigmoid_probabilities));
  assert((sigmoid_indices == std::vector<std::size_t>{0, 1, 2}));
  for (const double probability : sigmoid_probabilities) {
    assert(std::abs(probability - stable_sigmoid(7.0F)) < 1.0e-12);
  }
  assert(sigmoid_classifications(extreme_logits, 2, 2, sigmoid_indices,
                                 sigmoid_probabilities));
  assert((sigmoid_indices == std::vector<std::size_t>{0, 1}));
  assert(sigmoid_probabilities[0] == 1.0 && sigmoid_probabilities[1] == 0.0);
  assert(stable_sigmoid(0.0F) == 0.5);
  sigmoid_indices = {9};
  sigmoid_probabilities = {9.0};
  assert(!sigmoid_classifications(nullptr, 4, 2, sigmoid_indices,
                                  sigmoid_probabilities));
  assert(!sigmoid_classifications(one, 1, 1, sigmoid_indices,
                                  sigmoid_probabilities));
  assert(!sigmoid_classifications(unique, 4, 0, sigmoid_indices,
                                  sigmoid_probabilities));
  assert(!sigmoid_classifications(unique, 4,
                                  kMaxTopClassificationCount + 1,
                                  sigmoid_indices, sigmoid_probabilities));
  assert(!sigmoid_classifications(nan_later, 2, 1, sigmoid_indices,
                                  sigmoid_probabilities));
  assert((sigmoid_indices == std::vector<std::size_t>{9}));
  assert((sigmoid_probabilities == std::vector<double>{9.0}));

  std::vector<std::size_t> sigmoid_candidates = {9};
  std::vector<double> sigmoid_candidate_probabilities = {9.0};
  assert(sigmoid_classification_candidates(
      unique, 4, 0.5, 4, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert((sigmoid_candidates == std::vector<std::size_t>{2, 3}));
  assert(sigmoid_candidate_probabilities.size() == 2);
  assert(sigmoid_classification_candidates(
      unique, 4, stable_sigmoid(2.0F), 4, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert((sigmoid_candidates == std::vector<std::size_t>{2, 3}));
  assert(sigmoid_classification_candidates(
      unique, 4, 0.99, 4, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert(sigmoid_candidates.empty() &&
         sigmoid_candidate_probabilities.empty());
  assert(sigmoid_classification_candidates(
      tie, 4, stable_sigmoid(7.0F), 2, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert((sigmoid_candidates == std::vector<std::size_t>{0, 1}));
  sigmoid_candidates = {9};
  sigmoid_candidate_probabilities = {9.0};
  assert(!sigmoid_classification_candidates(
      unique, 4, -0.1, 2, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert(!sigmoid_classification_candidates(
      unique, 4, 1.1, 2, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert(!sigmoid_classification_candidates(
      unique, 4, std::numeric_limits<double>::quiet_NaN(), 2,
      sigmoid_candidates, sigmoid_candidate_probabilities));
  assert(!sigmoid_classification_candidates(
      unique, 4, 0.5, 0, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert(!sigmoid_classification_candidates(
      unique, 4, 0.5, kMaxTopClassificationCount + 1,
      sigmoid_candidates, sigmoid_candidate_probabilities));
  assert(!sigmoid_classification_candidates(
      nan_later, 2, 0.5, 2, sigmoid_candidates,
      sigmoid_candidate_probabilities));
  assert((sigmoid_candidates == std::vector<std::size_t>{9}));
  assert((sigmoid_candidate_probabilities == std::vector<double>{9.0}));

  const float channels[] = {
      0.0F, 0.5F, 1.0F, -1.0F, 1.0F, 0.5F, 0.25F, 0.75F};
  std::uint8_t mask[] = {7, 7, 7, 7};
  assert(channel_mask(channels, 8, 2, 2, 2, 1, 0.5, mask));
  assert(mask[0] == 255 && mask[1] == 0 && mask[2] == 0 && mask[3] == 255);

  std::uint8_t untouched[] = {9, 9, 9, 9};
  assert(!channel_mask(nullptr, 8, 2, 2, 2, 1, 0.5, untouched));
  assert(!channel_mask(channels, 8, 2, 2, 2, 1, 0.5, nullptr));
  assert(!channel_mask(channels, 8, 0, 2, 2, 0, 0.5, untouched));
  assert(!channel_mask(channels, 8, 2, 0, 2, 0, 0.5, untouched));
  assert(!channel_mask(channels, 8, 2, 2, 0, 0, 0.5, untouched));
  assert(!channel_mask(channels, 8, 2, 2, 2, 2, 0.5, untouched));
  assert(!channel_mask(channels, 7, 2, 2, 2, 1, 0.5, untouched));
  assert(!channel_mask(channels, 8, 2, 2, 2, 1,
                       std::numeric_limits<double>::infinity(), untouched));
  assert(!channel_mask(channels, 8, 2, kMaxMaskPixelCount + 1, 1, 1, 0.5,
                       untouched));
  assert(untouched[0] == 9 && untouched[1] == 9 && untouched[2] == 9 &&
         untouched[3] == 9);

  const float nonfinite_selected[] = {
      0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
      std::numeric_limits<float>::quiet_NaN(), 0.0F, 1.0F};
  assert(!channel_mask(nonfinite_selected, 8, 2, 2, 2, 1, 0.5,
                       untouched));
  assert(untouched[0] == 9 && untouched[1] == 9 && untouched[2] == 9 &&
         untouched[3] == 9);

  const float class_scores[] = {
      1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
      0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F};
  std::uint8_t class_mask[] = {8, 8, 8, 8};
  assert(class_index_mask(class_scores, 12, 3, 2, 2, class_mask));
  assert(class_mask[0] == 0 && class_mask[1] == 1 && class_mask[2] == 2 &&
         class_mask[3] == 0);

  const float tied_scores[] = {0.0F, 0.0F, 0.0F, 0.0F,
                               0.0F, 0.0F, 0.0F, 0.0F};
  assert(class_index_mask(tied_scores, 8, 2, 2, 2, class_mask));
  assert(class_mask[0] == 0 && class_mask[1] == 0 && class_mask[2] == 0 &&
         class_mask[3] == 0);

  const float negative_scores[] = {-4.0F, -2.0F, -3.0F, -5.0F,
                                   -3.0F, -6.0F, -1.0F, -5.0F};
  assert(class_index_mask(negative_scores, 8, 2, 2, 2, class_mask));
  assert(class_mask[0] == 1 && class_mask[1] == 0 && class_mask[2] == 1 &&
         class_mask[3] == 0);

  std::uint8_t class_untouched[] = {6, 6, 6, 6};
  assert(!class_index_mask(nullptr, 12, 3, 2, 2, class_untouched));
  assert(!class_index_mask(class_scores, 12, 3, 2, 2, nullptr));
  assert(!class_index_mask(class_scores, 4, 1, 2, 2, class_untouched));
  assert(!class_index_mask(class_scores, 12, kMaxClassMaskCount + 1, 2, 2,
                           class_untouched));
  assert(!class_index_mask(class_scores, 12, 3, 0, 2, class_untouched));
  assert(!class_index_mask(class_scores, 12, 3, 2, 0, class_untouched));
  assert(!class_index_mask(class_scores, 11, 3, 2, 2, class_untouched));
  assert(!class_index_mask(class_scores, 12, 3, kMaxMaskPixelCount + 1, 1,
                           class_untouched));
  const float nonfinite_class_scores[] = {
      1.0F, 0.0F, 0.0F, 1.0F, 0.0F,
      1.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F,
      0.0F, 0.0F, 1.0F, 1.0F};
  assert(!class_index_mask(nonfinite_class_scores, 12, 3, 2, 2,
                           class_untouched));
  assert(class_untouched[0] == 6 && class_untouched[1] == 6 &&
         class_untouched[2] == 6 && class_untouched[3] == 6);

  const float probability_scores[] = {
      1.0F, 0.0F, 0.0F, 32.0F / 255.0F, 0.0F, 128.0F / 255.0F,
      0.0F, 32.0F / 255.0F, 0.0F, 0.0F, 64.0F / 255.0F,
      32.0F / 255.0F};
  std::uint8_t confidence_mask[] = {5, 5, 5, 5};
  assert(class_confidence_mask(probability_scores, 12, 3, 2, 2,
                               confidence_mask));
  assert(confidence_mask[0] == 255 && confidence_mask[1] == 128 &&
         confidence_mask[2] == 64 && confidence_mask[3] == 32);
  assert(class_confidence_mask(tied_scores, 8, 2, 2, 2, confidence_mask));
  assert(confidence_mask[0] == 0 && confidence_mask[1] == 0 &&
         confidence_mask[2] == 0 && confidence_mask[3] == 0);

  std::uint8_t confidence_untouched[] = {4, 4, 4, 4};
  assert(!class_confidence_mask(nullptr, 12, 3, 2, 2,
                                confidence_untouched));
  assert(!class_confidence_mask(probability_scores, 12, 3, 2, 2, nullptr));
  assert(!class_confidence_mask(probability_scores, 4, 1, 2, 2,
                                confidence_untouched));
  assert(!class_confidence_mask(probability_scores, 12,
                                kMaxClassMaskCount + 1, 2, 2,
                                confidence_untouched));
  assert(!class_confidence_mask(probability_scores, 12, 3, 0, 2,
                                confidence_untouched));
  assert(!class_confidence_mask(probability_scores, 11, 3, 2, 2,
                                confidence_untouched));
  const float negative_probability[] = {0.0F, 0.0F, -0.1F, 0.0F,
                                        0.0F, 0.0F, 0.0F,  0.0F};
  const float excessive_probability[] = {0.0F, 0.0F, 1.1F, 0.0F,
                                         0.0F, 0.0F, 0.0F, 0.0F};
  assert(!class_confidence_mask(negative_probability, 8, 2, 2, 2,
                                confidence_untouched));
  assert(!class_confidence_mask(excessive_probability, 8, 2, 2, 2,
                                confidence_untouched));
  assert(!class_confidence_mask(nonfinite_class_scores, 12, 3, 2, 2,
                                confidence_untouched));
  assert(confidence_untouched[0] == 4 && confidence_untouched[1] == 4 &&
         confidence_untouched[2] == 4 && confidence_untouched[3] == 4);

  std::uint8_t softmax_confidence[] = {5, 5, 5, 5};
  assert(softmax_class_confidence_mask(probability_scores, 12, 3, 2, 2,
                                       softmax_confidence));
  assert(softmax_confidence[0] == 147 && softmax_confidence[1] == 115 &&
         softmax_confidence[2] == 100 && softmax_confidence[3] == 85);
  assert(softmax_class_confidence_mask(tied_scores, 8, 2, 2, 2,
                                       softmax_confidence));
  assert(softmax_confidence[0] == 128 && softmax_confidence[1] == 128 &&
         softmax_confidence[2] == 128 && softmax_confidence[3] == 128);
  const float extreme_pixel_logits[] = {
      std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};
  assert(softmax_class_confidence_mask(extreme_pixel_logits, 2, 2, 1, 1,
                                       softmax_confidence));
  assert(softmax_confidence[0] == 255);

  std::uint8_t softmax_untouched[] = {3, 3, 3, 3};
  assert(!softmax_class_confidence_mask(nullptr, 12, 3, 2, 2,
                                        softmax_untouched));
  assert(!softmax_class_confidence_mask(probability_scores, 12, 3, 2, 2,
                                        nullptr));
  assert(!softmax_class_confidence_mask(probability_scores, 4, 1, 2, 2,
                                        softmax_untouched));
  assert(!softmax_class_confidence_mask(
      probability_scores, 12, kMaxClassMaskCount + 1, 2, 2,
      softmax_untouched));
  assert(!softmax_class_confidence_mask(probability_scores, 12, 3, 0, 2,
                                        softmax_untouched));
  assert(!softmax_class_confidence_mask(probability_scores, 11, 3, 2, 2,
                                        softmax_untouched));
  assert(!softmax_class_confidence_mask(nonfinite_class_scores, 12, 3, 2, 2,
                                        softmax_untouched));
  assert(softmax_untouched[0] == 3 && softmax_untouched[1] == 3 &&
         softmax_untouched[2] == 3 && softmax_untouched[3] == 3);

  std::uint8_t class_probability[] = {2, 2, 2, 2};
  assert(softmax_class_probability_mask(probability_scores, 12, 3, 2, 2, 0,
                                        class_probability));
  assert(class_probability[0] == 147 && class_probability[1] == 70 &&
         class_probability[2] == 78 && class_probability[3] == 85);
  assert(softmax_class_probability_mask(tied_scores, 8, 2, 2, 2, 1,
                                        class_probability));
  assert(class_probability[0] == 128 && class_probability[1] == 128 &&
         class_probability[2] == 128 && class_probability[3] == 128);
  assert(softmax_class_probability_mask(extreme_pixel_logits, 2, 2, 1, 1, 1,
                                        class_probability));
  assert(class_probability[0] == 0);

  std::uint8_t probability_untouched[] = {2, 2, 2, 2};
  assert(!softmax_class_probability_mask(nullptr, 12, 3, 2, 2, 0,
                                         probability_untouched));
  assert(!softmax_class_probability_mask(probability_scores, 12, 3, 2, 2, 0,
                                         nullptr));
  assert(!softmax_class_probability_mask(probability_scores, 4, 1, 2, 2, 0,
                                         probability_untouched));
  assert(!softmax_class_probability_mask(
      probability_scores, 12, kMaxClassMaskCount + 1, 2, 2, 0,
      probability_untouched));
  assert(!softmax_class_probability_mask(probability_scores, 12, 3, 2, 2, 3,
                                         probability_untouched));
  assert(!softmax_class_probability_mask(probability_scores, 11, 3, 2, 2, 0,
                                         probability_untouched));
  assert(!softmax_class_probability_mask(nonfinite_class_scores, 12, 3, 2, 2,
                                         0, probability_untouched));
  assert(probability_untouched[0] == 2 && probability_untouched[1] == 2 &&
         probability_untouched[2] == 2 && probability_untouched[3] == 2);

  std::uint8_t softmax_binary_mask[] = {5, 5, 5, 5};
  assert(softmax_class_mask(probability_scores, 12, 3, 2, 2, 0, 0.5,
                            softmax_binary_mask));
  assert(softmax_binary_mask[0] == 255 && softmax_binary_mask[1] == 0 &&
         softmax_binary_mask[2] == 0 && softmax_binary_mask[3] == 0);
  assert(softmax_class_mask(probability_scores, 12, 3, 2, 2, 2, 0.3,
                            softmax_binary_mask));
  assert(softmax_binary_mask[0] == 0 && softmax_binary_mask[1] == 0 &&
         softmax_binary_mask[2] == 255 && softmax_binary_mask[3] == 255);
  assert(softmax_class_mask(tied_scores, 8, 2, 2, 2, 1, 0.5,
                            softmax_binary_mask));
  assert(softmax_binary_mask[0] == 255 && softmax_binary_mask[1] == 255 &&
         softmax_binary_mask[2] == 255 && softmax_binary_mask[3] == 255);

  std::uint8_t softmax_binary_untouched[] = {5, 5, 5, 5};
  assert(!softmax_class_mask(nullptr, 12, 3, 2, 2, 0, 0.5,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 12, 3, 2, 2, 0, 0.5,
                             nullptr));
  assert(!softmax_class_mask(probability_scores, 4, 1, 2, 2, 0, 0.5,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 12,
                             kMaxClassMaskCount + 1, 2, 2, 0, 0.5,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 12, 3, 2, 2, 3, 0.5,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 11, 3, 2, 2, 0, 0.5,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 12, 3, 2, 2, 0, -0.01,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 12, 3, 2, 2, 0, 1.01,
                             softmax_binary_untouched));
  assert(!softmax_class_mask(probability_scores, 12, 3, 2, 2, 0,
                             std::numeric_limits<double>::infinity(),
                             softmax_binary_untouched));
  assert(!softmax_class_mask(nonfinite_class_scores, 12, 3, 2, 2, 0, 0.5,
                             softmax_binary_untouched));
  assert(softmax_binary_untouched[0] == 5 &&
         softmax_binary_untouched[1] == 5 &&
         softmax_binary_untouched[2] == 5 &&
         softmax_binary_untouched[3] == 5);

  std::uint8_t sigmoid_probability_mask[] = {1, 1, 1, 1};
  assert(sigmoid_class_probability_mask(probability_scores, 12, 3, 2, 2, 0,
                                        sigmoid_probability_mask));
  assert(sigmoid_probability_mask[0] == 186 &&
         sigmoid_probability_mask[1] == 128 &&
         sigmoid_probability_mask[2] == 128 &&
         sigmoid_probability_mask[3] == 135);
  assert(sigmoid_class_probability_mask(probability_scores, 12, 3, 2, 2, 2,
                                        sigmoid_probability_mask));
  assert(sigmoid_probability_mask[0] == 128 &&
         sigmoid_probability_mask[1] == 128 &&
         sigmoid_probability_mask[2] == 143 &&
         sigmoid_probability_mask[3] == 135);
  const float binary_logits[] = {std::numeric_limits<float>::max(), 0.0F,
                                 -std::numeric_limits<float>::max()};
  assert(sigmoid_class_probability_mask(binary_logits, 3, 1, 1, 3, 0,
                                        sigmoid_probability_mask));
  assert(sigmoid_probability_mask[0] == 255 &&
         sigmoid_probability_mask[1] == 128 &&
         sigmoid_probability_mask[2] == 0);

  std::uint8_t sigmoid_mask_untouched[] = {1, 1, 1, 1};
  assert(!sigmoid_class_probability_mask(nullptr, 12, 3, 2, 2, 0,
                                         sigmoid_mask_untouched));
  assert(!sigmoid_class_probability_mask(probability_scores, 12, 3, 2, 2, 0,
                                         nullptr));
  assert(!sigmoid_class_probability_mask(probability_scores, 12, 0, 2, 2, 0,
                                         sigmoid_mask_untouched));
  assert(!sigmoid_class_probability_mask(
      probability_scores, 12, kMaxClassMaskCount + 1, 2, 2, 0,
      sigmoid_mask_untouched));
  assert(!sigmoid_class_probability_mask(probability_scores, 12, 3, 2, 2, 3,
                                         sigmoid_mask_untouched));
  assert(!sigmoid_class_probability_mask(probability_scores, 11, 3, 2, 2, 0,
                                         sigmoid_mask_untouched));
  assert(!sigmoid_class_probability_mask(nonfinite_class_scores, 12, 3, 2, 2,
                                         0, sigmoid_mask_untouched));
  assert(sigmoid_mask_untouched[0] == 1 &&
         sigmoid_mask_untouched[1] == 1 &&
         sigmoid_mask_untouched[2] == 1 &&
         sigmoid_mask_untouched[3] == 1);

  std::uint8_t sigmoid_binary_mask[] = {4, 4, 4, 4};
  assert(sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 0, 0.5,
                            sigmoid_binary_mask));
  assert(sigmoid_binary_mask[0] == 255 && sigmoid_binary_mask[1] == 255 &&
         sigmoid_binary_mask[2] == 255 && sigmoid_binary_mask[3] == 255);
  assert(sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 2, 0.55,
                            sigmoid_binary_mask));
  assert(sigmoid_binary_mask[0] == 0 && sigmoid_binary_mask[1] == 0 &&
         sigmoid_binary_mask[2] == 255 && sigmoid_binary_mask[3] == 0);
  assert(sigmoid_class_mask(binary_logits, 3, 1, 1, 3, 0, 0.5,
                            sigmoid_binary_mask));
  assert(sigmoid_binary_mask[0] == 255 && sigmoid_binary_mask[1] == 255 &&
         sigmoid_binary_mask[2] == 0);

  std::uint8_t sigmoid_binary_untouched[] = {4, 4, 4, 4};
  assert(!sigmoid_class_mask(nullptr, 12, 3, 2, 2, 0, 0.5,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 0, 0.5,
                             nullptr));
  assert(!sigmoid_class_mask(probability_scores, 12, 0, 2, 2, 0, 0.5,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 12,
                             kMaxClassMaskCount + 1, 2, 2, 0, 0.5,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 3, 0.5,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 11, 3, 2, 2, 0, 0.5,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 0, -0.01,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 0, 1.01,
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(probability_scores, 12, 3, 2, 2, 0,
                             std::numeric_limits<double>::quiet_NaN(),
                             sigmoid_binary_untouched));
  assert(!sigmoid_class_mask(nonfinite_class_scores, 12, 3, 2, 2, 0, 0.5,
                             sigmoid_binary_untouched));
  assert(sigmoid_binary_untouched[0] == 4 &&
         sigmoid_binary_untouched[1] == 4 &&
         sigmoid_binary_untouched[2] == 4 &&
         sigmoid_binary_untouched[3] == 4);

  std::uint8_t sigmoid_any_mask[] = {6, 6, 6, 6};
  assert(sigmoid_any_class_mask(probability_scores, 12, 3, 2, 2, 0.55,
                                sigmoid_any_mask));
  assert(sigmoid_any_mask[0] == 255 && sigmoid_any_mask[1] == 255 &&
         sigmoid_any_mask[2] == 255 && sigmoid_any_mask[3] == 0);
  assert(sigmoid_any_class_mask(tied_scores, 8, 2, 2, 2, 0.5,
                                sigmoid_any_mask));
  assert(sigmoid_any_mask[0] == 255 && sigmoid_any_mask[1] == 255 &&
         sigmoid_any_mask[2] == 255 && sigmoid_any_mask[3] == 255);
  assert(sigmoid_any_class_mask(binary_logits, 3, 1, 1, 3, 0.5,
                                sigmoid_any_mask));
  assert(sigmoid_any_mask[0] == 255 && sigmoid_any_mask[1] == 255 &&
         sigmoid_any_mask[2] == 0);

  std::uint8_t sigmoid_confidence[] = {7, 7, 7, 7};
  assert(sigmoid_class_confidence_mask(probability_scores, 12, 3, 2, 2,
                                       sigmoid_confidence));
  assert(sigmoid_confidence[0] == 186 && sigmoid_confidence[1] == 159 &&
         sigmoid_confidence[2] == 143 && sigmoid_confidence[3] == 135);
  assert(sigmoid_class_confidence_mask(binary_logits, 3, 1, 1, 3,
                                       sigmoid_confidence));
  assert(sigmoid_confidence[0] == 255 && sigmoid_confidence[1] == 128 &&
         sigmoid_confidence[2] == 0);

  std::uint8_t sigmoid_any_untouched[] = {6, 6, 6, 6};
  assert(!sigmoid_any_class_mask(nullptr, 12, 3, 2, 2, 0.5,
                                 sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(probability_scores, 12, 3, 2, 2, 0.5,
                                 nullptr));
  assert(!sigmoid_any_class_mask(probability_scores, 12, 0, 2, 2, 0.5,
                                 sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(probability_scores, 12,
                                 kMaxClassMaskCount + 1, 2, 2, 0.5,
                                 sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(probability_scores, 11, 3, 2, 2, 0.5,
                                 sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(probability_scores, 12, 3, 2, 2, -0.01,
                                 sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(probability_scores, 12, 3, 2, 2, 1.01,
                                 sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(
      probability_scores, 12, 3, 2, 2,
      std::numeric_limits<double>::quiet_NaN(), sigmoid_any_untouched));
  assert(!sigmoid_any_class_mask(nonfinite_class_scores, 12, 3, 2, 2, 0.5,
                                 sigmoid_any_untouched));
  assert(sigmoid_any_untouched[0] == 6 && sigmoid_any_untouched[1] == 6 &&
         sigmoid_any_untouched[2] == 6 && sigmoid_any_untouched[3] == 6);

  std::uint8_t sigmoid_confidence_untouched[] = {7, 7, 7, 7};
  assert(!sigmoid_class_confidence_mask(nullptr, 12, 3, 2, 2,
                                        sigmoid_confidence_untouched));
  assert(!sigmoid_class_confidence_mask(probability_scores, 12, 3, 2, 2,
                                        nullptr));
  assert(!sigmoid_class_confidence_mask(probability_scores, 12, 0, 2, 2,
                                        sigmoid_confidence_untouched));
  assert(!sigmoid_class_confidence_mask(
      probability_scores, 12, kMaxClassMaskCount + 1, 2, 2,
      sigmoid_confidence_untouched));
  assert(!sigmoid_class_confidence_mask(probability_scores, 11, 3, 2, 2,
                                        sigmoid_confidence_untouched));
  assert(!sigmoid_class_confidence_mask(nonfinite_class_scores, 12, 3, 2, 2,
                                        sigmoid_confidence_untouched));
  assert(sigmoid_confidence_untouched[0] == 7 &&
         sigmoid_confidence_untouched[1] == 7 &&
         sigmoid_confidence_untouched[2] == 7 &&
         sigmoid_confidence_untouched[3] == 7);

  const float detection_rows[] = {
      0.0F, 2.0F, 0.75F, 0.1F, 0.2F, 0.6F, 0.7F,
      0.0F, 1.0F, 0.9F,  0.0F, 0.1F, 0.5F, 0.6F,
      0.0F, 0.0F, 0.99F, 0.0F, 0.0F, 1.0F, 1.0F};
  std::vector<DnnDetection> detections;
  assert(ssd_detections(detection_rows, 21, 3, 0.5, 3, detections));
  assert(detections.size() == 2);
  assert(detections[0].class_index == 1 && detections[0].confidence > 0.89 &&
         detections[0].source_index == 1);
  assert(detections[1].class_index == 2 && detections[1].confidence == 0.75 &&
         detections[1].left > 0.09 && detections[1].bottom < 0.71);
  assert(ssd_detections(detection_rows, 21, 3, 0.85, 3, detections));
  assert(detections.size() == 1 && detections[0].class_index == 1);
  assert(ssd_detections(detection_rows, 21, 3, 0.75, 3, detections));
  assert(detections.size() == 2);
  assert(ssd_detections(detection_rows, 21, 3, 0.0, 1, detections));
  assert(detections.size() == 1 && detections[0].class_index == 1);

  const float tied_detection_rows[] = {
      0.0F, 2.0F, 0.8F, 0.0F, 0.0F, 0.5F, 0.5F,
      0.0F, 1.0F, 0.8F, 0.1F, 0.1F, 0.6F, 0.6F,
      0.0F, 1.0F, 0.8F, 0.2F, 0.2F, 0.7F, 0.7F};
  assert(ssd_detections(tied_detection_rows, 21, 3, 0.5, 3, detections));
  assert(detections.size() == 3 && detections[0].class_index == 1 &&
         detections[0].source_index == 1 && detections[1].source_index == 2 &&
         detections[2].class_index == 2);

  detections.assign(1, DnnDetection{9, 0.5, 0.0, 0.0, 1.0, 1.0, 9});
  assert(!ssd_detections(nullptr, 21, 3, 0.5, 3, detections));
  assert(!ssd_detections(detection_rows, 21, 0, 0.5, 3, detections));
  assert(!ssd_detections(detection_rows, 21, kMaxDetectionRowCount + 1, 0.5,
                         3, detections));
  assert(!ssd_detections(detection_rows, 20, 3, 0.5, 3, detections));
  assert(!ssd_detections(detection_rows, 21, 3, -0.1, 3, detections));
  assert(!ssd_detections(detection_rows, 21, 3, 1.1, 3, detections));
  assert(!ssd_detections(detection_rows, 21, 3,
                         std::numeric_limits<double>::quiet_NaN(), 3,
                         detections));
  assert(!ssd_detections(detection_rows, 21, 3, 0.5, 0, detections));
  assert(!ssd_detections(detection_rows, 21, 3, 0.5,
                         kMaxDetectionCount + 1, detections));
  const float invalid_batch[] = {1.0F, 1.0F, 0.8F, 0.0F, 0.0F, 1.0F, 1.0F};
  const float invalid_class[] = {0.0F, 1.5F, 0.8F, 0.0F, 0.0F, 1.0F, 1.0F};
  const float excessive_class[] = {
      0.0F, 1000001.0F, 0.8F, 0.0F, 0.0F, 1.0F, 1.0F};
  const float invalid_confidence[] = {
      0.0F, 1.0F, 1.1F, 0.0F, 0.0F, 1.0F, 1.0F};
  const float invalid_box[] = {0.0F, 1.0F, 0.8F, 0.7F, 0.0F, 0.6F, 1.0F};
  const float excessive_box[] = {0.0F, 1.0F, 0.8F, 0.0F, 0.0F, 1.1F, 1.0F};
  const float nonfinite_detection[] = {
      0.0F, 1.0F, std::numeric_limits<float>::quiet_NaN(),
      0.0F, 0.0F, 1.0F, 1.0F};
  assert(!ssd_detections(invalid_batch, 7, 1, 0.5, 1, detections));
  assert(!ssd_detections(invalid_class, 7, 1, 0.5, 1, detections));
  assert(!ssd_detections(excessive_class, 7, 1, 0.5, 1, detections));
  assert(!ssd_detections(invalid_confidence, 7, 1, 0.5, 1, detections));
  assert(!ssd_detections(invalid_box, 7, 1, 0.5, 1, detections));
  assert(!ssd_detections(excessive_box, 7, 1, 0.5, 1, detections));
  assert(!ssd_detections(nonfinite_detection, 7, 1, 0.5, 1, detections));
  assert(detections.size() == 1 && detections[0].class_index == 9 &&
         detections[0].source_index == 9);

  const int xyxy_rank_two[] = {3, 6};
  const int xyxy_rank_three[] = {1, 3, 6};
  const int xyxy_bad_batch[] = {2, 3, 6};
  const int xyxy_bad_fields[] = {1, 3, 7};
  const int xyxy_empty[] = {1, 0, 6};
  std::size_t xyxy_rows = 99;
  assert(normalized_detection_shape(xyxy_rank_two, 2, xyxy_rows));
  assert(xyxy_rows == 3);
  assert(normalized_detection_shape(xyxy_rank_three, 3, xyxy_rows));
  assert(xyxy_rows == 3);
  assert(!normalized_detection_shape(nullptr, 3, xyxy_rows));
  assert(!normalized_detection_shape(xyxy_rank_three, 1, xyxy_rows));
  assert(!normalized_detection_shape(xyxy_bad_batch, 3, xyxy_rows));
  assert(!normalized_detection_shape(xyxy_bad_fields, 3, xyxy_rows));
  assert(!normalized_detection_shape(xyxy_empty, 3, xyxy_rows));
  assert(xyxy_rows == 3);

  const float xyxy_rows_values[] = {
      0.1F, 0.2F, 0.6F, 0.7F, 0.75F, 2.0F,
      0.0F, 0.1F, 0.5F, 0.6F, 0.9F,  1.0F,
      0.2F, 0.2F, 0.7F, 0.7F, 0.9F,  0.0F};
  assert(normalized_xyxy_detections(xyxy_rows_values, 18, 3, 0.5, 3,
                                    detections));
  assert(detections.size() == 3);
  assert(detections[0].class_index == 0 &&
         detections[0].source_index == 2);
  assert(detections[1].class_index == 1 &&
         detections[1].source_index == 1);
  assert(detections[2].class_index == 2 &&
         detections[2].source_index == 0 && detections[2].left > 0.09 &&
         detections[2].bottom < 0.71);
  assert(normalized_xyxy_detections(xyxy_rows_values, 18, 3,
                                    static_cast<double>(0.9F), 3,
                                    detections));
  assert(detections.size() == 2);
  assert(normalized_xyxy_detections(xyxy_rows_values, 18, 3, 0.0, 1,
                                    detections));
  assert(detections.size() == 1 && detections[0].class_index == 0);

  detections.assign(1, DnnDetection{9, 0.5, 0.0, 0.0, 1.0, 1.0, 9});
  assert(!normalized_xyxy_detections(nullptr, 18, 3, 0.5, 3, detections));
  assert(!normalized_xyxy_detections(xyxy_rows_values, 17, 3, 0.5, 3,
                                     detections));
  assert(!normalized_xyxy_detections(xyxy_rows_values, 18, 0, 0.5, 3,
                                     detections));
  assert(!normalized_xyxy_detections(xyxy_rows_values, 18, 3, -0.1, 3,
                                     detections));
  assert(!normalized_xyxy_detections(xyxy_rows_values, 18, 3, 1.1, 3,
                                     detections));
  assert(!normalized_xyxy_detections(
      xyxy_rows_values, 18, 3, std::numeric_limits<double>::infinity(), 3,
      detections));
  assert(!normalized_xyxy_detections(xyxy_rows_values, 18, 3, 0.5, 0,
                                     detections));
  const float xyxy_bad_class[] = {0.0F, 0.0F, 1.0F, 1.0F, 0.8F, 1.5F};
  const float xyxy_bad_confidence[] = {0.0F, 0.0F, 1.0F, 1.0F, 1.1F, 1.0F};
  const float xyxy_bad_box[] = {0.8F, 0.0F, 0.7F, 1.0F, 0.8F, 1.0F};
  const float xyxy_nonfinite[] = {
      0.0F, 0.0F, 1.0F, 1.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F};
  assert(!normalized_xyxy_detections(xyxy_bad_class, 6, 1, 0.5, 1,
                                     detections));
  assert(!normalized_xyxy_detections(xyxy_bad_confidence, 6, 1, 0.5, 1,
                                     detections));
  assert(!normalized_xyxy_detections(xyxy_bad_box, 6, 1, 0.5, 1,
                                     detections));
  assert(!normalized_xyxy_detections(xyxy_nonfinite, 6, 1, 0.5, 1,
                                     detections));
  assert(detections.size() == 1 && detections[0].class_index == 9);

  const float center_xywh_rows[] = {
      0.25F, 0.25F, 0.3F, 0.3F, 0.8F, 0.0F,
      0.4F,  0.4F,  0.4F, 0.4F, 0.9F, 1.0F,
      0.8F,  0.8F,  0.2F, 0.2F, 0.4F, 0.0F};
  assert(normalized_center_xywh_detections(center_xywh_rows, 18, 3, 0.5, 3,
                                           detections));
  assert(detections.size() == 2);
  assert(detections[0].class_index == 1 &&
         detections[0].source_index == 1 && detections[0].left > 0.19 &&
         detections[0].right < 0.61);
  assert(detections[1].class_index == 0 &&
         detections[1].source_index == 0 && detections[1].left > 0.09 &&
         detections[1].bottom < 0.41);
  assert(normalized_center_xywh_detections(
      center_xywh_rows, 18, 3, static_cast<double>(0.8F), 3, detections));
  assert(detections.size() == 2);
  assert(normalized_center_xywh_detections(center_xywh_rows, 18, 3, 0.0, 1,
                                           detections));
  assert(detections.size() == 1 && detections[0].class_index == 1);

  detections.assign(1, DnnDetection{9, 0.5, 0.0, 0.0, 1.0, 1.0, 9});
  assert(!normalized_center_xywh_detections(nullptr, 18, 3, 0.5, 3,
                                            detections));
  assert(!normalized_center_xywh_detections(center_xywh_rows, 17, 3, 0.5, 3,
                                            detections));
  assert(!normalized_center_xywh_detections(center_xywh_rows, 18, 0, 0.5, 3,
                                            detections));
  assert(!normalized_center_xywh_detections(center_xywh_rows, 18, 3, -0.1, 3,
                                            detections));
  assert(!normalized_center_xywh_detections(center_xywh_rows, 18, 3, 0.5, 0,
                                            detections));
  const float center_zero_width[] = {0.5F, 0.5F, 0.0F, 0.2F, 0.8F, 1.0F};
  const float center_outside[] = {0.1F, 0.5F, 0.4F, 0.2F, 0.8F, 1.0F};
  const float center_bad_class[] = {0.5F, 0.5F, 0.2F, 0.2F, 0.8F, -1.0F};
  const float center_bad_confidence[] = {0.5F, 0.5F, 0.2F, 0.2F, -0.1F,
                                         1.0F};
  const float center_nonfinite[] = {
      0.5F, std::numeric_limits<float>::infinity(), 0.2F, 0.2F, 0.8F, 1.0F};
  assert(!normalized_center_xywh_detections(center_zero_width, 6, 1, 0.5, 1,
                                            detections));
  assert(!normalized_center_xywh_detections(center_outside, 6, 1, 0.5, 1,
                                            detections));
  assert(!normalized_center_xywh_detections(center_bad_class, 6, 1, 0.5, 1,
                                            detections));
  assert(!normalized_center_xywh_detections(center_bad_confidence, 6, 1, 0.5,
                                            1, detections));
  assert(!normalized_center_xywh_detections(center_nonfinite, 6, 1, 0.5, 1,
                                            detections));
  assert(detections.size() == 1 && detections[0].class_index == 9);

  const float top_left_xywh_rows[] = {
      0.1F, 0.1F, 0.3F, 0.3F, 0.8F, 0.0F,
      0.2F, 0.2F, 0.4F, 0.4F, 0.9F, 1.0F,
      0.7F, 0.7F, 0.2F, 0.2F, 0.4F, 0.0F};
  assert(normalized_top_left_xywh_detections(top_left_xywh_rows, 18, 3, 0.5,
                                             3, detections));
  assert(detections.size() == 2);
  assert(detections[0].class_index == 1 &&
         detections[0].source_index == 1 && detections[0].left > 0.19 &&
         detections[0].right < 0.61);
  assert(detections[1].class_index == 0 &&
         detections[1].source_index == 0 && detections[1].left > 0.09 &&
         detections[1].bottom < 0.41);
  assert(normalized_top_left_xywh_detections(
      top_left_xywh_rows, 18, 3, static_cast<double>(0.8F), 3, detections));
  assert(detections.size() == 2);
  assert(normalized_top_left_xywh_detections(top_left_xywh_rows, 18, 3, 0.0,
                                             1, detections));
  assert(detections.size() == 1 && detections[0].class_index == 1);

  detections.assign(1, DnnDetection{9, 0.5, 0.0, 0.0, 1.0, 1.0, 9});
  assert(!normalized_top_left_xywh_detections(nullptr, 18, 3, 0.5, 3,
                                              detections));
  assert(!normalized_top_left_xywh_detections(top_left_xywh_rows, 17, 3, 0.5,
                                              3, detections));
  assert(!normalized_top_left_xywh_detections(top_left_xywh_rows, 18, 0, 0.5,
                                              3, detections));
  assert(!normalized_top_left_xywh_detections(top_left_xywh_rows, 18, 3, 1.1,
                                              3, detections));
  assert(!normalized_top_left_xywh_detections(top_left_xywh_rows, 18, 3, 0.5,
                                              0, detections));
  const float top_left_zero_height[] = {0.5F, 0.5F, 0.2F, 0.0F, 0.8F, 1.0F};
  const float top_left_outside[] = {0.9F, 0.5F, 0.2F, 0.2F, 0.8F, 1.0F};
  assert(!normalized_top_left_xywh_detections(top_left_zero_height, 6, 1, 0.5,
                                              1, detections));
  assert(!normalized_top_left_xywh_detections(top_left_outside, 6, 1, 0.5, 1,
                                              detections));
  assert(!normalized_top_left_xywh_detections(center_bad_class, 6, 1, 0.5, 1,
                                              detections));
  assert(!normalized_top_left_xywh_detections(center_bad_confidence, 6, 1,
                                              0.5, 1, detections));
  assert(!normalized_top_left_xywh_detections(center_nonfinite, 6, 1, 0.5, 1,
                                              detections));
  assert(detections.size() == 1 && detections[0].class_index == 9);

  assert(nms_parameters(0.0, 1));
  assert(nms_parameters(0.5, kMaxDetectionCount));
  assert(nms_parameters(1.0, 3));
  assert(!nms_parameters(-0.1, 3));
  assert(!nms_parameters(1.1, 3));
  assert(!nms_parameters(std::numeric_limits<double>::infinity(), 3));
  assert(!nms_parameters(std::numeric_limits<double>::quiet_NaN(), 3));
  assert(!nms_parameters(0.5, 0));
  assert(!nms_parameters(0.5, kMaxDetectionCount + 1));
}
