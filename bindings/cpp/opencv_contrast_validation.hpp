#pragma once

#include "opencv_validation.hpp"

#include <cmath>
#include <cstdint>

namespace terlan_opencv {
namespace validation {

/// Largest accepted CLAHE contrast clip limit.
constexpr double kMaxAdaptiveEqualizationClipLimit = 256.0;

/// Largest accepted tile count along either image axis.
constexpr std::int64_t kMaxAdaptiveEqualizationTiles = 64;

/// Checks one bounded grayscale adaptive-histogram equalization request.
inline bool valid_adaptive_equalization(std::int64_t rows, std::int64_t cols,
                                        int channels, double clip_limit,
                                        std::int64_t tiles_x,
                                        std::int64_t tiles_y) noexcept {
  return valid_analysis_image(rows, cols, channels) &&
         std::isfinite(clip_limit) && clip_limit > 0.0 &&
         clip_limit <= kMaxAdaptiveEqualizationClipLimit && tiles_x > 0 &&
         tiles_x <= kMaxAdaptiveEqualizationTiles && tiles_x <= cols &&
         tiles_y > 0 && tiles_y <= kMaxAdaptiveEqualizationTiles &&
         tiles_y <= rows;
}

}  // namespace validation
}  // namespace terlan_opencv
