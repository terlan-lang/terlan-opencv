#ifndef TERLAN_OPENCV_DNN_VALIDATION_HPP
#define TERLAN_OPENCV_DNN_VALIDATION_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "opencv_dnn_detection_validation.hpp"

namespace terlan_opencv {
namespace dnn_validation {

/// Minimum score count accepted as a classification output.
constexpr std::size_t kMinClassificationCount = 2;

/// Maximum score count accepted as a classification output.
constexpr std::size_t kMaxClassificationCount = 1000000;

/// Maximum number of ranked classifications returned by one projection.
constexpr std::size_t kMaxTopClassificationCount = 10000;

/// Maximum number of inference values copied by one slice projection.
constexpr std::size_t kMaxOutputSliceValueCount = 1024U * 1024U;

/// Maximum grayscale mask allocation admitted by one output projection.
constexpr std::size_t kMaxMaskPixelCount = 256U * 1024U * 1024U;

/// Minimum class count accepted by a class-index mask projection.
constexpr std::size_t kMinClassMaskCount = 2;

/// Maximum class count representable by one grayscale class-index byte.
constexpr std::size_t kMaxClassMaskCount = 256;

/// Byte-domain channel parameters for maintained image-to-blob preprocessing.
struct DnnNormalization final {
  std::array<double, 3> byte_means;   ///< Per-channel means in byte units.
  std::array<double, 3> byte_scales;  ///< Per-channel byte-domain scales.
};

/// Converts one- or three-channel normalization into bounded byte-domain values.
inline bool normalization_parameters(
    const double* means, std::size_t mean_count, const double* scales,
    std::size_t scale_count, std::size_t channel_count,
    DnnNormalization& parameters) noexcept {
  if (means == nullptr || scales == nullptr ||
      (channel_count != 1 && channel_count != 3) ||
      mean_count != channel_count || scale_count != channel_count) {
    return false;
  }
  DnnNormalization checked{};
  for (std::size_t channel = 0; channel < channel_count; ++channel) {
    if (!std::isfinite(means[channel]) || !std::isfinite(scales[channel]) ||
        scales[channel] <= 0.0) {
      return false;
    }
    const double byte_mean = means[channel] * 255.0;
    const double byte_scale = scales[channel] / 255.0;
    const double minimum = -means[channel] * scales[channel];
    const double maximum = (1.0 - means[channel]) * scales[channel];
    const double float_limit =
        static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(byte_mean) || !std::isfinite(byte_scale) ||
        byte_scale <= 0.0 || !std::isfinite(minimum) ||
        !std::isfinite(maximum) || std::abs(minimum) > float_limit ||
        std::abs(maximum) > float_limit) {
      return false;
    }
    checked.byte_means[channel] = byte_mean;
    checked.byte_scales[channel] = byte_scale;
  }
  parameters = checked;
  return true;
}

/// Converts normalized RGB mean/scale lists into bounded byte-domain values.
inline bool rgb_normalization_parameters(
    const double* means, std::size_t mean_count, const double* scales,
    std::size_t scale_count, DnnNormalization& parameters) noexcept {
  return normalization_parameters(means, mean_count, scales, scale_count, 3,
                                  parameters);
}

/// Converts one grayscale mean and scale into bounded byte-domain values.
inline bool grayscale_normalization_parameters(
    const double* means, std::size_t mean_count, const double* scales,
    std::size_t scale_count, DnnNormalization& parameters) noexcept {
  return normalization_parameters(means, mean_count, scales, scale_count, 1,
                                  parameters);
}

/// Resolves one bounded half-open slice without overflowing native indexes.
inline bool output_slice_bounds(std::size_t total, std::int64_t offset,
                                std::int64_t maximum, std::size_t& begin,
                                std::size_t& end) noexcept {
  if (total == 0 || offset < 0 || maximum <= 0 ||
      maximum > static_cast<std::int64_t>(kMaxOutputSliceValueCount) ||
      static_cast<std::uint64_t>(offset) > total) {
    return false;
  }
  const std::size_t checked_begin = static_cast<std::size_t>(offset);
  const std::size_t selected = std::min(
      total - checked_begin, static_cast<std::size_t>(maximum));
  begin = checked_begin;
  end = checked_begin + selected;
  return true;
}

/// Validates bounded channel-major tensor dimensions and returns pixel count.
inline bool bounded_channel_tensor_size(
    std::size_t count, std::size_t channels, std::size_t minimum_channels,
    std::size_t height, std::size_t width,
    std::size_t& pixel_count) noexcept {
  if (channels < minimum_channels || channels > kMaxClassMaskCount ||
      height == 0 || width == 0 || height > kMaxMaskPixelCount / width) {
    return false;
  }
  const std::size_t checked_pixel_count = height * width;
  if (checked_pixel_count > kMaxMaskPixelCount ||
      channels >
          std::numeric_limits<std::size_t>::max() / checked_pixel_count ||
      count != channels * checked_pixel_count) {
    return false;
  }
  pixel_count = checked_pixel_count;
  return true;
}

/// Validates one bounded multi-class tensor and returns its pixel count.
inline bool class_tensor_size(std::size_t count, std::size_t classes,
                              std::size_t height, std::size_t width,
                              std::size_t& pixel_count) noexcept {
  return bounded_channel_tensor_size(count, classes, kMinClassMaskCount,
                                     height, width, pixel_count);
}

/// Reports whether one class-score row is finite and within package bounds.
inline bool classification_scores_valid(const float* scores,
                                        std::size_t count) noexcept {
  if (scores == nullptr || count < kMinClassificationCount ||
      count > kMaxClassificationCount) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(scores[index])) {
      return false;
    }
  }
  return true;
}

/// Orders class indexes by descending score and then ascending index.
inline bool classification_precedes(const float* scores, std::size_t first,
                                    std::size_t second) noexcept {
  return scores[first] > scores[second] ||
         (scores[first] == scores[second] && first < second);
}

/// Selects the first highest finite score from one bounded class-score row.
inline bool top_classification(const float* scores, std::size_t count,
                               std::int64_t& class_index,
                               double& score) noexcept {
  if (!classification_scores_valid(scores, count)) {
    return false;
  }
  std::size_t best_index = 0;
  float best_score = scores[0];
  for (std::size_t index = 1; index < count; ++index) {
    if (classification_precedes(scores, index, best_index)) {
      best_index = index;
      best_score = scores[index];
    }
  }
  class_index = static_cast<std::int64_t>(best_index);
  score = static_cast<double>(best_score);
  return true;
}

/// Ranks a bounded prefix of one finite class-score row.
inline bool top_classification_indices(const float* scores, std::size_t count,
                                       std::size_t maximum,
                                       std::vector<std::size_t>& indices) {
  if (!classification_scores_valid(scores, count) || maximum == 0 ||
      maximum > kMaxTopClassificationCount) {
    return false;
  }
  std::vector<std::size_t> ranked(count);
  std::iota(ranked.begin(), ranked.end(), 0);
  const std::size_t selected = std::min(count, maximum);
  std::partial_sort(
      ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected),
      ranked.end(), [scores](std::size_t first, std::size_t second) {
        return classification_precedes(scores, first, second);
      });
  ranked.resize(selected);
  indices = std::move(ranked);
  return true;
}

/// Selects a bounded canonical ranking whose raw scores meet one threshold.
inline bool classification_candidate_indices(
    const float* scores, std::size_t count, double minimum_score,
    std::size_t maximum, std::vector<std::size_t>& indices) {
  if (!std::isfinite(minimum_score)) {
    return false;
  }
  std::vector<std::size_t> ranked;
  if (!top_classification_indices(scores, count, maximum, ranked)) {
    return false;
  }
  ranked.erase(
      std::remove_if(ranked.begin(), ranked.end(),
                     [scores, minimum_score](std::size_t index) {
                       return static_cast<double>(scores[index]) < minimum_score;
                     }),
      ranked.end());
  indices = std::move(ranked);
  return true;
}

/// Computes and ranks bounded stable-softmax probabilities from finite logits.
inline bool softmax_classifications(
    const float* scores, std::size_t count, std::size_t maximum,
    std::vector<std::size_t>& indices,
    std::vector<double>& probabilities) {
  if (!classification_scores_valid(scores, count) || maximum == 0 ||
      maximum > kMaxTopClassificationCount) {
    return false;
  }
  const float maximum_score = *std::max_element(scores, scores + count);
  std::vector<double> weights(count);
  double total = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double weight =
        std::exp(static_cast<double>(scores[index]) -
                 static_cast<double>(maximum_score));
    if (!std::isfinite(weight) || weight < 0.0) {
      return false;
    }
    weights[index] = weight;
    total += weight;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    return false;
  }
  std::vector<std::size_t> ranked(count);
  std::iota(ranked.begin(), ranked.end(), 0);
  const std::size_t selected = std::min(count, maximum);
  std::partial_sort(
      ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected),
      ranked.end(), [&weights](std::size_t first, std::size_t second) {
        return weights[first] > weights[second] ||
               (weights[first] == weights[second] && first < second);
      });
  ranked.resize(selected);
  std::vector<double> selected_probabilities;
  selected_probabilities.reserve(selected);
  for (const std::size_t index : ranked) {
    const double probability = weights[index] / total;
    if (!std::isfinite(probability) || probability < 0.0 ||
        probability > 1.0) {
      return false;
    }
    selected_probabilities.push_back(probability);
  }
  indices = std::move(ranked);
  probabilities = std::move(selected_probabilities);
  return true;
}

/// Computes one finite logit's stable sigmoid probability.
inline double stable_sigmoid(float score) noexcept {
  const double value = static_cast<double>(score);
  if (value >= 0.0) {
    return 1.0 / (1.0 + std::exp(-value));
  }
  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}

/// Returns whether one probability or probability threshold is admissible.
inline bool valid_probability(double value) noexcept {
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

/// Computes and ranks bounded stable-sigmoid probabilities from finite logits.
inline bool sigmoid_classifications(
    const float* scores, std::size_t count, std::size_t maximum,
    std::vector<std::size_t>& indices,
    std::vector<double>& probabilities) {
  if (!classification_scores_valid(scores, count) || maximum == 0 ||
      maximum > kMaxTopClassificationCount) {
    return false;
  }
  std::vector<double> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = stable_sigmoid(scores[index]);
  }
  std::vector<std::size_t> ranked(count);
  std::iota(ranked.begin(), ranked.end(), 0);
  const std::size_t selected = std::min(count, maximum);
  std::partial_sort(
      ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(selected),
      ranked.end(), [&values](std::size_t first, std::size_t second) {
        return values[first] > values[second] ||
               (values[first] == values[second] && first < second);
      });
  ranked.resize(selected);
  std::vector<double> selected_probabilities;
  selected_probabilities.reserve(selected);
  for (const std::size_t index : ranked) {
    selected_probabilities.push_back(values[index]);
  }
  indices = std::move(ranked);
  probabilities = std::move(selected_probabilities);
  return true;
}

/// Selects a bounded stable-sigmoid ranking at an inclusive probability floor.
inline bool sigmoid_classification_candidates(
    const float* scores, std::size_t count, double minimum_probability,
    std::size_t maximum, std::vector<std::size_t>& indices,
    std::vector<double>& probabilities) {
  if (!valid_probability(minimum_probability)) {
    return false;
  }
  std::vector<std::size_t> ranked;
  std::vector<double> ranked_probabilities;
  if (!sigmoid_classifications(scores, count, maximum, ranked,
                               ranked_probabilities)) {
    return false;
  }
  std::vector<std::size_t> selected_indices;
  std::vector<double> selected_probabilities;
  selected_indices.reserve(ranked.size());
  selected_probabilities.reserve(ranked_probabilities.size());
  for (std::size_t position = 0; position < ranked.size(); ++position) {
    if (ranked_probabilities[position] >= minimum_probability) {
      selected_indices.push_back(ranked[position]);
      selected_probabilities.push_back(ranked_probabilities[position]);
    }
  }
  indices = std::move(selected_indices);
  probabilities = std::move(selected_probabilities);
  return true;
}

/// Thresholds one channel-major tensor plane after validating the full shape.
inline bool channel_mask(const float* values, std::size_t count,
                         std::size_t channels, std::size_t height,
                         std::size_t width, std::size_t channel,
                         double threshold, std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr || channels == 0 || height == 0 ||
      width == 0 || channel >= channels || !std::isfinite(threshold) ||
      height > kMaxMaskPixelCount / width) {
    return false;
  }
  const std::size_t pixel_count = height * width;
  if (pixel_count > kMaxMaskPixelCount ||
      channels > std::numeric_limits<std::size_t>::max() / pixel_count ||
      count != channels * pixel_count) {
    return false;
  }
  const float* selected = values + channel * pixel_count;
  for (std::size_t index = 0; index < pixel_count; ++index) {
    if (!std::isfinite(selected[index])) {
      return false;
    }
  }
  for (std::size_t index = 0; index < pixel_count; ++index) {
    mask[index] = selected[index] > threshold ? 255U : 0U;
  }
  return true;
}

/// Selects the first highest finite class score for every output pixel.
inline bool class_index_mask(const float* values, std::size_t count,
                             std::size_t classes, std::size_t height,
                             std::size_t width,
                             std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!class_tensor_size(count, classes, height, width, pixel_count)) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    std::size_t best_class = 0;
    float best_score = values[pixel];
    for (std::size_t class_index = 1; class_index < classes; ++class_index) {
      const float score = values[class_index * pixel_count + pixel];
      if (score > best_score) {
        best_class = class_index;
        best_score = score;
      }
    }
    mask[pixel] = static_cast<std::uint8_t>(best_class);
  }
  return true;
}

/// Selects and byte-scales the highest finite probability for every pixel.
inline bool class_confidence_mask(const float* values, std::size_t count,
                                  std::size_t classes, std::size_t height,
                                  std::size_t width,
                                  std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!class_tensor_size(count, classes, height, width, pixel_count)) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index]) || values[index] < 0.0F ||
        values[index] > 1.0F) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    float best_score = values[pixel];
    for (std::size_t class_index = 1; class_index < classes; ++class_index) {
      const float score = values[class_index * pixel_count + pixel];
      if (score > best_score) {
        best_score = score;
      }
    }
    mask[pixel] = static_cast<std::uint8_t>(best_score * 255.0F + 0.5F);
  }
  return true;
}

/// Computes one selected class probability with stable per-pixel softmax.
inline double pixel_softmax_probability(const float* values,
                                        std::size_t classes,
                                        std::size_t pixel_count,
                                        std::size_t pixel,
                                        std::size_t selected_class) noexcept {
  float maximum = values[pixel];
  for (std::size_t class_index = 1; class_index < classes; ++class_index) {
    maximum =
        std::max(maximum, values[class_index * pixel_count + pixel]);
  }
  double total = 0.0;
  double selected_weight = 0.0;
  for (std::size_t class_index = 0; class_index < classes; ++class_index) {
    const double weight = std::exp(
        static_cast<double>(values[class_index * pixel_count + pixel]) -
        static_cast<double>(maximum));
    total += weight;
    if (class_index == selected_class) {
      selected_weight = weight;
    }
  }
  return selected_weight / total;
}

/// Stable-softmax normalizes and byte-scales the highest logit per pixel.
inline bool softmax_class_confidence_mask(const float* values,
                                          std::size_t count,
                                          std::size_t classes,
                                          std::size_t height,
                                          std::size_t width,
                                          std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!class_tensor_size(count, classes, height, width, pixel_count)) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    std::size_t best_class = 0;
    float best_score = values[pixel];
    for (std::size_t class_index = 1; class_index < classes; ++class_index) {
      const float score = values[class_index * pixel_count + pixel];
      if (score > best_score) {
        best_class = class_index;
        best_score = score;
      }
    }
    const double confidence = pixel_softmax_probability(
        values, classes, pixel_count, pixel, best_class);
    mask[pixel] = static_cast<std::uint8_t>(confidence * 255.0 + 0.5);
  }
  return true;
}

/// Stable-softmax normalizes and byte-scales one selected class per pixel.
inline bool softmax_class_probability_mask(
    const float* values, std::size_t count, std::size_t classes,
    std::size_t height, std::size_t width, std::size_t selected_class,
    std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!class_tensor_size(count, classes, height, width, pixel_count) ||
      selected_class >= classes) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    const double probability = pixel_softmax_probability(
        values, classes, pixel_count, pixel, selected_class);
    mask[pixel] = static_cast<std::uint8_t>(probability * 255.0 + 0.5);
  }
  return true;
}

/// Stable-softmax thresholds one selected class into a binary byte mask.
inline bool softmax_class_mask(const float* values, std::size_t count,
                               std::size_t classes, std::size_t height,
                               std::size_t width, std::size_t selected_class,
                               double minimum_probability,
                               std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr ||
      !valid_probability(minimum_probability)) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!class_tensor_size(count, classes, height, width, pixel_count) ||
      selected_class >= classes) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    const double probability = pixel_softmax_probability(
        values, classes, pixel_count, pixel, selected_class);
    mask[pixel] = probability >= minimum_probability ? std::uint8_t{255}
                                                     : std::uint8_t{0};
  }
  return true;
}

/// Stable-sigmoid normalizes and byte-scales one selected class per pixel.
inline bool sigmoid_class_probability_mask(
    const float* values, std::size_t count, std::size_t classes,
    std::size_t height, std::size_t width, std::size_t selected_class,
    std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!bounded_channel_tensor_size(count, classes, 1, height, width,
                                   pixel_count) ||
      selected_class >= classes) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  const float* selected = values + selected_class * pixel_count;
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    const double probability = stable_sigmoid(selected[pixel]);
    mask[pixel] = static_cast<std::uint8_t>(probability * 255.0 + 0.5);
  }
  return true;
}

/// Stable-sigmoid thresholds one selected class into a binary byte mask.
inline bool sigmoid_class_mask(const float* values, std::size_t count,
                               std::size_t classes, std::size_t height,
                               std::size_t width, std::size_t selected_class,
                               double minimum_probability,
                               std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr ||
      !valid_probability(minimum_probability)) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!bounded_channel_tensor_size(count, classes, 1, height, width,
                                   pixel_count) ||
      selected_class >= classes) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  const float* selected = values + selected_class * pixel_count;
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    mask[pixel] = stable_sigmoid(selected[pixel]) >= minimum_probability
                      ? std::uint8_t{255}
                      : std::uint8_t{0};
  }
  return true;
}

/// Returns the maximum independently normalized sigmoid probability at a pixel.
inline double pixel_max_sigmoid_probability(const float* values,
                                            std::size_t classes,
                                            std::size_t pixel_count,
                                            std::size_t pixel) noexcept {
  double maximum = stable_sigmoid(values[pixel]);
  for (std::size_t class_index = 1; class_index < classes; ++class_index) {
    maximum = std::max(
        maximum,
        stable_sigmoid(values[class_index * pixel_count + pixel]));
  }
  return maximum;
}

/// Stable-sigmoid byte-scales the maximum independent class probability.
inline bool sigmoid_class_confidence_mask(
    const float* values, std::size_t count, std::size_t classes,
    std::size_t height, std::size_t width, std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!bounded_channel_tensor_size(count, classes, 1, height, width,
                                   pixel_count)) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    const double probability =
        pixel_max_sigmoid_probability(values, classes, pixel_count, pixel);
    mask[pixel] = static_cast<std::uint8_t>(probability * 255.0 + 0.5);
  }
  return true;
}

/// Stable-sigmoid thresholds whether any class matches at each pixel.
inline bool sigmoid_any_class_mask(const float* values, std::size_t count,
                                   std::size_t classes, std::size_t height,
                                   std::size_t width,
                                   double minimum_probability,
                                   std::uint8_t* mask) noexcept {
  if (values == nullptr || mask == nullptr ||
      !valid_probability(minimum_probability)) {
    return false;
  }
  std::size_t pixel_count = 0;
  if (!bounded_channel_tensor_size(count, classes, 1, height, width,
                                   pixel_count)) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
    const double probability =
        pixel_max_sigmoid_probability(values, classes, pixel_count, pixel);
    mask[pixel] = probability >= minimum_probability ? std::uint8_t{255}
                                                     : std::uint8_t{0};
  }
  return true;
}

}  // namespace dnn_validation
}  // namespace terlan_opencv

#endif
