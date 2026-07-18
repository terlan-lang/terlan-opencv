#pragma once

#include "opencv_validation.hpp"

#include <cstdint>

namespace terlan_opencv {
namespace validation {

/// Checks one unsigned-byte drawing value.
constexpr bool valid_drawing_value(std::int64_t value) noexcept {
  return value >= 0 && value <= 255;
}

/// Checks one byte value and positive bounded drawing thickness.
constexpr bool valid_drawing_style(std::int64_t value,
                                   std::int64_t thickness) noexcept {
  return valid_drawing_value(value) && thickness > 0 &&
         thickness <= kMaxKernelDimension;
}

/// Checks one bounded grayscale line-drawing request.
inline bool valid_line_drawing(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t start_x, std::int64_t start_y, std::int64_t end_x,
    std::int64_t end_y, std::int64_t value,
    std::int64_t thickness) noexcept {
  return valid_analysis_image(rows, cols, channels) && start_x >= 0 &&
         start_x < cols && start_y >= 0 && start_y < rows && end_x >= 0 &&
         end_x < cols && end_y >= 0 && end_y < rows &&
         valid_drawing_style(value, thickness);
}

/// Checks one positive contained rectangle in a bounded grayscale image.
inline bool valid_rectangle_region(
    std::int64_t rows, std::int64_t cols, int channels, std::int64_t x,
    std::int64_t y, std::int64_t width, std::int64_t height,
    std::int64_t value) noexcept {
  return valid_analysis_image(rows, cols, channels) && x >= 0 && y >= 0 &&
         width > 0 && height > 0 && width <= cols && height <= rows &&
         x <= cols - width && y <= rows - height &&
         valid_drawing_value(value);
}

/// Checks one bounded grayscale rectangle-drawing request.
inline bool valid_rectangle_drawing(
    std::int64_t rows, std::int64_t cols, int channels, std::int64_t x,
    std::int64_t y, std::int64_t width, std::int64_t height,
    std::int64_t value, std::int64_t thickness) noexcept {
  return valid_rectangle_region(rows, cols, channels, x, y, width, height,
                                value) &&
         valid_drawing_style(value, thickness);
}

/// Checks one bounded grayscale rectangle-fill request.
inline bool valid_rectangle_fill(std::int64_t rows, std::int64_t cols,
                                 int channels, std::int64_t x,
                                 std::int64_t y, std::int64_t width,
                                 std::int64_t height,
                                 std::int64_t value) noexcept {
  return valid_rectangle_region(rows, cols, channels, x, y, width, height,
                                value);
}

/// Checks one contained grayscale circle region.
inline bool valid_circle_region(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius,
    std::int64_t value) noexcept {
  if (!valid_analysis_image(rows, cols, channels) ||
      !valid_drawing_value(value) || radius < 0 || radius >= rows ||
      radius >= cols) {
    return false;
  }
  return center_x >= radius && center_x <= cols - 1 - radius &&
         center_y >= radius && center_y <= rows - 1 - radius;
}

/// Checks one bounded grayscale circle-drawing request without clipping.
inline bool valid_circle_drawing(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius,
    std::int64_t value, std::int64_t thickness) noexcept {
  if (!valid_circle_region(rows, cols, channels, center_x, center_y, radius,
                           value) ||
      !valid_drawing_style(value, thickness) ||
      (radius == 0 && thickness != 1)) {
    return false;
  }
  const auto margin = radius == 0 ? 0 : radius + thickness / 2;
  return center_x >= margin && center_x <= cols - 1 - margin &&
         center_y >= margin && center_y <= rows - 1 - margin;
}

/// Checks one bounded grayscale circle-fill request.
inline bool valid_circle_fill(std::int64_t rows, std::int64_t cols,
                              int channels, std::int64_t center_x,
                              std::int64_t center_y, std::int64_t radius,
                              std::int64_t value) noexcept {
  return valid_circle_region(rows, cols, channels, center_x, center_y, radius,
                             value);
}

/// Checks one fully contained fixed-cross drawing request.
inline bool valid_cross_drawing(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t center_x, std::int64_t center_y, std::int64_t size,
    std::int64_t value, std::int64_t thickness) noexcept {
  if (!valid_analysis_image(rows, cols, channels) ||
      !valid_drawing_style(value, thickness) || size <= 0 ||
      size > kMaxKernelDimension || size % 2 == 0) {
    return false;
  }
  const auto margin = size / 2 + thickness / 2;
  return center_x >= margin && center_x <= cols - 1 - margin &&
         center_y >= margin && center_y <= rows - 1 - margin;
}

/// Checks one positive fully contained axis-aligned grayscale ellipse region.
inline bool valid_ellipse_region(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x,
    std::int64_t radius_y, std::int64_t value) noexcept {
  if (!valid_analysis_image(rows, cols, channels) ||
      !valid_drawing_value(value) || radius_x <= 0 || radius_y <= 0 ||
      radius_x >= cols || radius_y >= rows) {
    return false;
  }
  return center_x >= radius_x && center_x <= cols - 1 - radius_x &&
         center_y >= radius_y && center_y <= rows - 1 - radius_y;
}

/// Checks one bounded axis-aligned grayscale ellipse outline without clipping.
inline bool valid_ellipse_drawing(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x,
    std::int64_t radius_y, std::int64_t value,
    std::int64_t thickness) noexcept {
  if (!valid_ellipse_region(rows, cols, channels, center_x, center_y, radius_x,
                            radius_y, value) ||
      !valid_drawing_style(value, thickness)) {
    return false;
  }
  const auto margin_x = radius_x + thickness / 2;
  const auto margin_y = radius_y + thickness / 2;
  return center_x >= margin_x && center_x <= cols - 1 - margin_x &&
         center_y >= margin_y && center_y <= rows - 1 - margin_y;
}

/// Checks one bounded axis-aligned grayscale ellipse-fill request.
inline bool valid_ellipse_fill(
    std::int64_t rows, std::int64_t cols, int channels,
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x,
    std::int64_t radius_y, std::int64_t value) noexcept {
  return valid_ellipse_region(rows, cols, channels, center_x, center_y, radius_x,
                              radius_y, value);
}

}  // namespace validation
}  // namespace terlan_opencv
