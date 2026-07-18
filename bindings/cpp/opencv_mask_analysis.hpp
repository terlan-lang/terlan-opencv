#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace terlan_opencv {

/// Collects and validates every nonzero coordinate in one grayscale mask.
inline bool collect_non_zero_points(const cv::Mat& mask,
                                    std::vector<cv::Point>& points) {
  cv::findNonZero(mask, points);
  if (points.size() > mask.total()) {
    return false;
  }
  for (const auto& point : points) {
    if (point.x < 0 || point.x >= mask.cols || point.y < 0 ||
        point.y >= mask.rows) {
      return false;
    }
  }
  return true;
}

/// Rotates ordered points so the topmost then leftmost vertex appears first.
template <typename Point>
inline void rotate_topmost_leftmost(std::vector<Point>& points) {
  const auto first = std::min_element(
      points.begin(), points.end(), [](const Point& left, const Point& right) {
        return left.y < right.y ||
               (left.y == right.y && left.x < right.x);
      });
  std::rotate(points.begin(), first, points.end());
}

/// Computes one validated minimum circle for a nonempty bounded point set.
inline bool normalized_minimum_enclosing_circle(
    const std::vector<cv::Point>& points, std::int64_t rows,
    std::int64_t cols, cv::Point2f& center, float& radius) {
  if (points.empty() || rows <= 0 || cols <= 0) {
    return false;
  }
  if (points.size() == 1) {
    center = cv::Point2f(static_cast<float>(points.front().x),
                         static_cast<float>(points.front().y));
    radius = 0.0F;
  } else {
    cv::minEnclosingCircle(points, center, radius);
  }
  const double x = center.x;
  const double y = center.y;
  const double copied_radius = radius;
  const double maximum_radius =
      std::hypot(static_cast<double>(cols - 1),
                 static_cast<double>(rows - 1));
  return std::isfinite(x) && std::isfinite(y) &&
         std::isfinite(copied_radius) && x >= 0.0 && y >= 0.0 &&
         x <= static_cast<double>(cols - 1) &&
         y <= static_cast<double>(rows - 1) && copied_radius >= 0.0 &&
         copied_radius <= maximum_radius + 0.001;
}

/// Computes validated canonical unique corners for one minimum-area box.
inline bool normalized_minimum_area_box(
    const std::vector<cv::Point>& points, std::int64_t rows,
    std::int64_t cols, std::vector<cv::Point2f>& corners) {
  if (points.empty() || rows <= 0 || cols <= 0) {
    return false;
  }
  std::array<cv::Point2f, 4> raw_corners{};
  cv::minAreaRect(points).points(raw_corners.data());

  const double extent =
      std::hypot(static_cast<double>(cols - 1),
                 static_cast<double>(rows - 1)) +
      0.001;
  corners.clear();
  corners.reserve(4);
  for (const auto& corner : raw_corners) {
    const double x = corner.x;
    const double y = corner.y;
    if (!std::isfinite(x) || !std::isfinite(y) || x < -extent ||
        y < -extent || x > static_cast<double>(cols - 1) + extent ||
        y > static_cast<double>(rows - 1) + extent) {
      return false;
    }
    const auto duplicate = std::find_if(
        corners.begin(), corners.end(), [&corner](const cv::Point2f& other) {
          return std::hypot(static_cast<double>(corner.x - other.x),
                            static_cast<double>(corner.y - other.y)) <= 1.0e-5;
        });
    if (duplicate == corners.end()) {
      corners.push_back(corner);
    }
  }
  if (corners.empty() || corners.size() == 3 ||
      (corners.size() == 1 && points.size() != 1) ||
      (corners.size() == 4 && !cv::isContourConvex(corners))) {
    return false;
  }
  if (corners.size() == 4) {
    for (const auto& point : points) {
      const double distance = cv::pointPolygonTest(
          corners,
          cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)),
          true);
      if (!std::isfinite(distance) || distance < -0.001) {
        return false;
      }
    }
  }
  rotate_topmost_leftmost(corners);
  return true;
}

/// Computes one validated canonical minimum triangle for bounded points.
inline bool normalized_minimum_enclosing_triangle(
    const std::vector<cv::Point>& points, std::int64_t rows,
    std::int64_t cols, std::vector<cv::Point2f>& triangle) {
  if (points.empty() || rows <= 0 || cols <= 0) {
    return false;
  }
  std::vector<cv::Point> hull;
  cv::convexHull(points, hull, false, true);
  if (hull.empty() || hull.size() > points.size()) {
    return false;
  }

  triangle.clear();
  if (hull.size() <= 2) {
    triangle.reserve(hull.size());
    for (const auto& point : hull) {
      triangle.emplace_back(static_cast<float>(point.x),
                            static_cast<float>(point.y));
    }
  } else {
    std::vector<cv::Point2f> native_triangle;
    const double reported_area =
        cv::minEnclosingTriangle(hull, native_triangle);
    if (!std::isfinite(reported_area) || reported_area <= 0.0 ||
        native_triangle.size() != 3) {
      return false;
    }
    cv::convexHull(native_triangle, triangle, false, true);
    const double copied_area = cv::contourArea(triangle);
    const double maximum_area =
        2.0 * static_cast<double>(cols - 1) *
        static_cast<double>(rows - 1);
    const double area_tolerance = std::max(0.001, reported_area * 1.0e-4);
    if (triangle.size() != 3 || !cv::isContourConvex(triangle) ||
        !std::isfinite(copied_area) || copied_area <= 0.0 ||
        copied_area > maximum_area + area_tolerance ||
        std::abs(copied_area - reported_area) > area_tolerance) {
      return false;
    }
    for (const auto& point : points) {
      const double distance = cv::pointPolygonTest(
          triangle,
          cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y)),
          true);
      if (!std::isfinite(distance) || distance < -0.001) {
        return false;
      }
    }
  }

  const double extent =
      4.0 * std::hypot(static_cast<double>(cols - 1),
                       static_cast<double>(rows - 1)) +
      0.001;
  for (const auto& point : triangle) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        point.x < -extent || point.y < -extent ||
        point.x > static_cast<double>(cols - 1) + extent ||
        point.y > static_cast<double>(rows - 1) + extent) {
      return false;
    }
  }
  rotate_topmost_leftmost(triangle);
  return true;
}

/// Computes one validated canonical least-squares line for bounded points.
inline bool normalized_fitted_line(const std::vector<cv::Point>& points,
                                   std::int64_t rows, std::int64_t cols,
                                   std::array<double, 4>& line) {
  if (points.size() < 2 || rows <= 0 || cols <= 0) {
    return false;
  }
  cv::Vec4f native_line;
  cv::fitLine(points, native_line, cv::DIST_L2, 0.0, 0.01, 0.01);
  double vx = native_line[0];
  double vy = native_line[1];
  const double x0 = native_line[2];
  const double y0 = native_line[3];
  const double norm = std::hypot(vx, vy);
  if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(x0) ||
      !std::isfinite(y0) || !std::isfinite(norm) || norm < 0.999 ||
      norm > 1.001 || x0 < 0.0 || y0 < 0.0 ||
      x0 > static_cast<double>(cols - 1) ||
      y0 > static_cast<double>(rows - 1)) {
    return false;
  }
  vx /= norm;
  vy /= norm;
  if (vx < -1.0e-6 || (std::abs(vx) <= 1.0e-6 && vy < 0.0)) {
    vx = -vx;
    vy = -vy;
  }

  const double maximum_residual =
      std::hypot(static_cast<double>(cols - 1),
                 static_cast<double>(rows - 1)) +
      0.001;
  for (const auto& point : points) {
    const double residual =
        std::abs((static_cast<double>(point.x) - x0) * vy -
                 (static_cast<double>(point.y) - y0) * vx);
    if (!std::isfinite(residual) || residual > maximum_residual) {
      return false;
    }
  }
  line = {std::abs(vx) <= 1.0e-6 ? 0.0 : vx,
          std::abs(vy) <= 1.0e-6 ? 0.0 : vy,
          std::abs(x0) <= 1.0e-6 ? 0.0 : x0,
          std::abs(y0) <= 1.0e-6 ? 0.0 : y0};
  return true;
}

/// Returns whether bounded points can define a nondegenerate fitted ellipse.
inline bool supports_fitted_ellipse(const std::vector<cv::Point>& points) {
  if (points.size() < 5) {
    return false;
  }
  std::vector<cv::Point> hull;
  cv::convexHull(points, hull, false, true);
  return hull.size() >= 3 && cv::isContourConvex(hull) &&
         cv::contourArea(hull) > 1.0e-6;
}

/// Computes one validated canonical least-squares ellipse for bounded points.
inline bool normalized_fitted_ellipse(const std::vector<cv::Point>& points,
                                      std::int64_t rows, std::int64_t cols,
                                      std::array<double, 5>& ellipse) {
  if (!supports_fitted_ellipse(points) || rows <= 0 || cols <= 0) {
    return false;
  }
  const cv::RotatedRect native_ellipse = cv::fitEllipse(points);
  double center_x = native_ellipse.center.x;
  double center_y = native_ellipse.center.y;
  double major = native_ellipse.size.width;
  double minor = native_ellipse.size.height;
  double angle = native_ellipse.angle;
  if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
      !std::isfinite(major) || !std::isfinite(minor) ||
      !std::isfinite(angle) || major <= 0.0 || minor <= 0.0) {
    return false;
  }
  if (major < minor) {
    std::swap(major, minor);
    angle += 90.0;
  }
  angle = std::fmod(angle, 180.0);
  if (angle < 0.0) {
    angle += 180.0;
  }
  const double diagonal =
      std::hypot(static_cast<double>(cols - 1),
                 static_cast<double>(rows - 1));
  const double coordinate_margin = diagonal + 0.001;
  const double maximum_diameter = 4.0 * diagonal + 0.001;
  if (center_x < -coordinate_margin || center_y < -coordinate_margin ||
      center_x > static_cast<double>(cols - 1) + coordinate_margin ||
      center_y > static_cast<double>(rows - 1) + coordinate_margin ||
      major > maximum_diameter || minor > maximum_diameter) {
    return false;
  }
  if (std::abs(major - minor) <= 1.0e-4 * std::max(major, 1.0)) {
    angle = 0.0;
  }
  ellipse = {center_x, center_y, major, minor, angle};
  for (double& value : ellipse) {
    if (std::abs(value) <= 1.0e-6) {
      value = 0.0;
    }
  }
  return true;
}

/// Computes canonical I1, I2, and I3 scores for two supported shapes.
template <typename FirstShape, typename SecondShape>
inline bool normalized_shape_match_scores(const FirstShape& first,
                                          const SecondShape& second,
                                          std::array<double, 3>& scores) {
  constexpr std::array<int, 3> methods = {
      cv::CONTOURS_MATCH_I1, cv::CONTOURS_MATCH_I2, cv::CONTOURS_MATCH_I3};
  for (std::size_t index = 0; index < methods.size(); ++index) {
    const double score = cv::matchShapes(first, second, methods[index], 0.0);
    if (!std::isfinite(score) || score < 0.0) {
      return false;
    }
    scores[index] = score <= 1.0e-12 ? 0.0 : score;
  }
  return true;
}

}  // namespace terlan_opencv
