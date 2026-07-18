#include "opencv_adapter.hpp"
#include "opencv_drawing_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_mask_analysis.hpp"
#include "opencv_region_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <utility>

namespace terlan_opencv {
namespace {

std::atomic<std::int64_t> live_contour_analyses{0};
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kHuMomentCount = 7;
constexpr std::size_t kMinimumAreaBoxScalarCount = 8;
constexpr std::size_t kMinimumEnclosingCircleScalarCount = 3;
constexpr std::size_t kMinimumEnclosingTriangleScalarCount = 6;
constexpr std::size_t kFittedLineScalarCount = 4;
constexpr std::size_t kFittedEllipseScalarCount = 5;
constexpr std::size_t kShapeMatchScoreCount = 3;
constexpr std::size_t kConvexityDefectScalarCount = 7;

std::vector<cv::Point> preserve_degenerate_endpoints(
    const std::vector<cv::Point>& contour,
    std::vector<cv::Point> approximation) {
  if (contour.size() < 2 || approximation.size() >= 2 ||
      cv::contourArea(contour) != 0.0) {
    return approximation;
  }
  const cv::Rect bounds = cv::boundingRect(contour);
  const bool horizontal = bounds.width >= bounds.height;
  const auto ordered = [horizontal](const cv::Point& first,
                                    const cv::Point& second) {
    if (horizontal) {
      return first.x != second.x ? first.x < second.x : first.y < second.y;
    }
    return first.y != second.y ? first.y < second.y : first.x < second.x;
  };
  const auto endpoints = std::minmax_element(contour.begin(), contour.end(),
                                             ordered);
  if (*endpoints.first != *endpoints.second) {
    return {*endpoints.first, *endpoints.second};
  }
  return approximation;
}

}  // namespace

struct OpenCvContours::Impl final {
  Impl(std::int64_t rows, std::int64_t cols,
       std::vector<std::vector<cv::Point>> contours,
       std::vector<cv::Rect> bounds, std::vector<double> areas,
       std::vector<double> perimeters, std::vector<double> convex_areas,
       std::vector<cv::Point2d> centroids)
      : rows(rows),
        cols(cols),
        contours(std::move(contours)),
        bounds(std::move(bounds)),
        areas(std::move(areas)),
        perimeters(std::move(perimeters)),
        convex_areas(std::move(convex_areas)),
        centroids(std::move(centroids)) {}

  std::int64_t rows;
  std::int64_t cols;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Rect> bounds;
  std::vector<double> areas;
  std::vector<double> perimeters;
  std::vector<double> convex_areas;
  std::vector<cv::Point2d> centroids;
};

namespace {

std::unique_ptr<OpenCvContours> own_contours(
    std::int64_t rows, std::int64_t cols,
    std::vector<std::vector<cv::Point>> contours,
    std::vector<cv::Rect> bounds, std::vector<double> areas,
    std::vector<double> perimeters, std::vector<double> convex_areas,
    std::vector<cv::Point2d> centroids) {
  return std::make_unique<OpenCvContours>(
      std::make_unique<OpenCvContours::Impl>(
          rows, cols, std::move(contours), std::move(bounds), std::move(areas),
          std::move(perimeters), std::move(convex_areas),
          std::move(centroids)));
}

cv::Point2d contour_centroid(const std::vector<cv::Point>& contour) {
  const cv::Moments moments = cv::moments(contour);
  if (moments.m00 != 0.0) {
    return {moments.m10 / moments.m00, moments.m01 / moments.m00};
  }
  double x = 0.0;
  double y = 0.0;
  for (const auto& point : contour) {
    x += point.x;
    y += point.y;
  }
  const double count = static_cast<double>(contour.size());
  return {x / count, y / count};
}

bool valid_contour_result(const std::vector<std::vector<cv::Point>>& contours,
                          const std::vector<cv::Rect>& bounds,
                          const std::vector<double>& areas,
                          const std::vector<double>& perimeters,
                          const std::vector<double>& convex_areas,
                          const std::vector<cv::Point2d>& centroids,
                          std::int64_t rows, std::int64_t cols) {
  if (contours.size() != bounds.size() || contours.size() != areas.size() ||
      contours.size() != perimeters.size() ||
      contours.size() != convex_areas.size() ||
      contours.size() != centroids.size() ||
      contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels)) {
    return false;
  }
  std::size_t point_count = 0;
  for (std::size_t index = 0; index < contours.size(); ++index) {
    const auto& contour = contours[index];
    if (contour.empty() ||
        contour.size() >
            static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) -
                point_count ||
        !std::isfinite(areas[index]) || areas[index] < 0.0 ||
        !std::isfinite(perimeters[index]) || perimeters[index] < 0.0) {
      return false;
    }
    if (!std::isfinite(convex_areas[index]) || convex_areas[index] < 0.0 ||
        convex_areas[index] + 1.0e-9 < areas[index] ||
        convex_areas[index] > static_cast<double>(rows * cols)) {
      return false;
    }
    point_count += contour.size();
    for (const auto& point : contour) {
      if (point.x < 0 || point.y < 0 || point.x >= cols || point.y >= rows) {
        return false;
      }
    }
    const auto& bound = bounds[index];
    if (bound.x < 0 || bound.y < 0 || bound.width <= 0 || bound.height <= 0 ||
        bound.x > cols - bound.width || bound.y > rows - bound.height) {
      return false;
    }
    const double bound_area =
        static_cast<double>(bound.width) * static_cast<double>(bound.height);
    if (areas[index] > bound_area + 1.0e-9) {
      return false;
    }
    const auto& centroid = centroids[index];
    if (!std::isfinite(centroid.x) || !std::isfinite(centroid.y) ||
        centroid.x < static_cast<double>(bound.x) - 1.0e-9 ||
        centroid.y < static_cast<double>(bound.y) - 1.0e-9 ||
        centroid.x > static_cast<double>(bound.x + bound.width - 1) + 1.0e-9 ||
        centroid.y > static_cast<double>(bound.y + bound.height - 1) +
                         1.0e-9) {
      return false;
    }
  }
  return true;
}

/// Builds one fully validated contour owner and every retained projection.
std::unique_ptr<OpenCvContours> analyze_contours(
    std::vector<std::vector<cv::Point>> contours, std::int64_t rows,
    std::int64_t cols) {
  std::vector<cv::Rect> bounds;
  std::vector<double> areas;
  std::vector<double> perimeters;
  std::vector<double> convex_areas;
  std::vector<cv::Point2d> centroids;
  bounds.reserve(contours.size());
  areas.reserve(contours.size());
  perimeters.reserve(contours.size());
  convex_areas.reserve(contours.size());
  centroids.reserve(contours.size());
  for (const auto& contour : contours) {
    if (contour.empty()) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    bounds.push_back(cv::boundingRect(contour));
    areas.push_back(cv::contourArea(contour));
    perimeters.push_back(cv::arcLength(contour, true));
    convex_areas.push_back(cv::contourArea(hull));
    centroids.push_back(contour_centroid(contour));
  }
  if (!valid_contour_result(contours, bounds, areas, perimeters, convex_areas,
                            centroids, rows, cols)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return own_contours(rows, cols, std::move(contours), std::move(bounds),
                      std::move(areas), std::move(perimeters),
                      std::move(convex_areas), std::move(centroids));
}

/// Renders validated contour geometry into one checked grayscale matrix.
std::unique_ptr<OpenCvMat> render_contours_mask(
    const std::vector<std::vector<cv::Point>>& contours, std::int64_t rows,
    std::int64_t cols, int contour_index, std::int64_t value, int thickness) {
  cv::Mat result = cv::Mat::zeros(static_cast<int>(rows),
                                  static_cast<int>(cols), CV_8UC1);
  if (!contours.empty()) {
    cv::drawContours(result, contours, contour_index,
                     cv::Scalar(static_cast<double>(value)), thickness,
                     cv::LINE_8);
  }
  for (int row = 0; row < result.rows; ++row) {
    const auto* pixels = result.ptr<std::uint8_t>(row);
    for (int col = 0; col < result.cols; ++col) {
      if (pixels[col] != 0 && pixels[col] != value) {
        failure::set(failure::NativeFailure::NativeFailure);
        return nullptr;
      }
    }
  }
  return own_grayscale_mat(std::move(result), rows, cols);
}

}  // namespace

std::unique_ptr<OpenCvContours>
OpenCvMat::find_external_contours_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvContours>([this] {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(impl_->value.clone(), contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    return analyze_contours(std::move(contours), impl_->value.rows,
                            impl_->value.cols);
  });
}

OpenCvContours::OpenCvContours(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  ++live_contour_analyses;
}

OpenCvContours::~OpenCvContours() noexcept { --live_contour_analyses; }

std::int64_t OpenCvContours::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->contours.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvContours::points_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    return region::copy_points(impl_->contours);
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvContours::offsets_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    return region::copy_offsets(impl_->contours);
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvContours::bounds_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    return region::copy_bounds(impl_->bounds);
  });
}

std::unique_ptr<std::vector<double>> OpenCvContours::areas_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this] { return std::make_unique<std::vector<double>>(impl_->areas); });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::perimeters_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    return std::make_unique<std::vector<double>>(impl_->perimeters);
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::convex_areas_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    return std::make_unique<std::vector<double>>(impl_->convex_areas);
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::solidities_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->areas.size());
    for (std::size_t index = 0; index < impl_->areas.size(); ++index) {
      const double area = impl_->areas[index];
      const double convex_area = impl_->convex_areas[index];
      if (convex_area == 0.0) {
        if (area != 0.0) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<double>>{};
        }
        result->push_back(0.0);
        continue;
      }
      const double solidity = area / convex_area;
      if (!std::isfinite(solidity) || solidity < 0.0 ||
          solidity > 1.0 + 1.0e-9) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(solidity > 1.0 ? 1.0 : solidity);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::circularities_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->areas.size());
    for (std::size_t index = 0; index < impl_->areas.size(); ++index) {
      const double area = impl_->areas[index];
      const double perimeter = impl_->perimeters[index];
      if (perimeter == 0.0) {
        if (area != 0.0) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<double>>{};
        }
        result->push_back(0.0);
        continue;
      }
      const double circularity =
          4.0 * kPi * area / (perimeter * perimeter);
      if (!std::isfinite(circularity) || circularity < 0.0 ||
          circularity > 1.0 + 1.0e-9) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(circularity > 1.0 ? 1.0 : circularity);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::extents_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->areas.size());
    for (std::size_t index = 0; index < impl_->areas.size(); ++index) {
      const auto& bound = impl_->bounds[index];
      const double bound_area = static_cast<double>(bound.width) *
                                static_cast<double>(bound.height);
      const double extent = impl_->areas[index] / bound_area;
      if (!std::isfinite(extent) || extent < 0.0 ||
          extent > 1.0 + 1.0e-9) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(extent > 1.0 ? 1.0 : extent);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::centroids_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->centroids.size() * 2);
    for (const auto& centroid : impl_->centroids) {
      result->push_back(centroid.x);
      result->push_back(centroid.y);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::hu_moments_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kHuMomentCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() * kHuMomentCount);
    for (const auto& contour : impl_->contours) {
      double values[kHuMomentCount] = {};
      cv::HuMoments(cv::moments(contour), values);
      for (const double value : values) {
        if (!std::isfinite(value)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<double>>{};
        }
        result->push_back(value);
      }
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::minimum_area_boxes_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kMinimumAreaBoxScalarCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() * kMinimumAreaBoxScalarCount);
    for (std::size_t index = 0; index < impl_->contours.size(); ++index) {
      std::vector<cv::Point2f> corners;
      if (!normalized_minimum_area_box(impl_->contours[index],
                                       impl_->rows, impl_->cols, corners)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      const std::array<std::size_t, 4> corner_indexes =
          corners.size() == 1
              ? std::array<std::size_t, 4>{0, 0, 0, 0}
              : corners.size() == 2
                    ? std::array<std::size_t, 4>{0, 1, 1, 0}
                    : std::array<std::size_t, 4>{0, 1, 2, 3};
      for (const std::size_t corner_index : corner_indexes) {
        const auto& corner = corners[corner_index];
        result->push_back(std::abs(corner.x) <= 1.0e-6 ? 0.0 : corner.x);
        result->push_back(std::abs(corner.y) <= 1.0e-6 ? 0.0 : corner.y);
      }
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::minimum_enclosing_circles_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kMinimumEnclosingCircleScalarCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() *
                    kMinimumEnclosingCircleScalarCount);
    for (std::size_t index = 0; index < impl_->contours.size(); ++index) {
      const auto& bound = impl_->bounds[index];
      cv::Point2f center;
      float radius = 0.0F;
      if (!normalized_minimum_enclosing_circle(
              impl_->contours[index], bound.y + bound.height,
              bound.x + bound.width, center, radius)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(center.x);
      result->push_back(center.y);
      result->push_back(radius);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::minimum_enclosing_triangles_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kMinimumEnclosingTriangleScalarCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() *
                    kMinimumEnclosingTriangleScalarCount);
    for (std::size_t index = 0; index < impl_->contours.size(); ++index) {
      const auto& bound = impl_->bounds[index];
      std::vector<cv::Point2f> triangle;
      if (!normalized_minimum_enclosing_triangle(
              impl_->contours[index], bound.y + bound.height,
              bound.x + bound.width, triangle)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      const std::array<std::size_t, 3> vertex_indexes =
          triangle.size() == 1
              ? std::array<std::size_t, 3>{0, 0, 0}
              : triangle.size() == 2
                    ? std::array<std::size_t, 3>{0, 1, 1}
                    : std::array<std::size_t, 3>{0, 1, 2};
      for (const std::size_t vertex_index : vertex_indexes) {
        const auto& vertex = triangle[vertex_index];
        result->push_back(std::abs(vertex.x) <= 1.0e-6 ? 0.0 : vertex.x);
        result->push_back(std::abs(vertex.y) <= 1.0e-6 ? 0.0 : vertex.y);
      }
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::fitted_lines_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kFittedLineScalarCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() * kFittedLineScalarCount);
    for (std::size_t index = 0; index < impl_->contours.size(); ++index) {
      const auto& contour = impl_->contours[index];
      const auto& bound = impl_->bounds[index];
      std::array<double, 4> line{};
      if (contour.size() == 1) {
        line = {0.0, 0.0, static_cast<double>(contour.front().x),
                static_cast<double>(contour.front().y)};
      } else if (!normalized_fitted_line(contour, bound.y + bound.height,
                                         bound.x + bound.width, line)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->insert(result->end(), line.begin(), line.end());
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::fitted_ellipses_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kFittedEllipseScalarCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() * kFittedEllipseScalarCount);
    for (std::size_t index = 0; index < impl_->contours.size(); ++index) {
      const auto& contour = impl_->contours[index];
      const auto& bound = impl_->bounds[index];
      std::array<double, 5> ellipse{};
      if (!supports_fitted_ellipse(contour)) {
        ellipse = {impl_->centroids[index].x, impl_->centroids[index].y, 0.0,
                   0.0, 0.0};
      } else if (!normalized_fitted_ellipse(
                     contour, bound.y + bound.height,
                     bound.x + bound.width, ellipse)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->insert(result->end(), ellipse.begin(), ellipse.end());
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::shape_match_scores_owned(
    std::int64_t reference_index) const noexcept {
  failure::clear();
  if (impl_ == nullptr || reference_index < 0 ||
      static_cast<std::size_t>(reference_index) >= impl_->contours.size() ||
      impl_->contours.size() >
          static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
              kShapeMatchScoreCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, reference_index] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size() * kShapeMatchScoreCount);
    const auto& reference =
        impl_->contours[static_cast<std::size_t>(reference_index)];
    for (const auto& contour : impl_->contours) {
      std::array<double, 3> scores{};
      if (!normalized_shape_match_scores(reference, contour, scores)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->insert(result->end(), scores.begin(), scores.end());
    }
    return result;
  });
}

std::unique_ptr<OpenCvContours> OpenCvContours::approximate_owned(
    double epsilon) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_hull_approximation(impl_->rows, impl_->cols, 1,
                                            epsilon)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvContours>([this, epsilon] {
    std::vector<std::vector<cv::Point>> approximations;
    approximations.reserve(impl_->contours.size());
    for (const auto& contour : impl_->contours) {
      std::vector<cv::Point> approximation;
      cv::approxPolyDP(contour, approximation, epsilon, true);
      approximations.push_back(
          preserve_degenerate_endpoints(contour, std::move(approximation)));
    }
    return analyze_contours(std::move(approximations), impl_->rows,
                            impl_->cols);
  });
}

std::unique_ptr<OpenCvContours> OpenCvContours::convex_hulls_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvContours>([this] {
    std::vector<std::vector<cv::Point>> hulls;
    hulls.reserve(impl_->contours.size());
    for (const auto& contour : impl_->contours) {
      std::vector<cv::Point> hull;
      cv::convexHull(contour, hull);
      hulls.push_back(std::move(hull));
    }
    return analyze_contours(std::move(hulls), impl_->rows, impl_->cols);
  });
}

std::unique_ptr<std::vector<double>>
OpenCvContours::convexity_defects_owned(
    std::int64_t contour_index) const noexcept {
  failure::clear();
  if (impl_ == nullptr || contour_index < 0 ||
      static_cast<std::size_t>(contour_index) >= impl_->contours.size()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const auto& contour =
      impl_->contours[static_cast<std::size_t>(contour_index)];
  if (contour.size() >
      static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
          kConvexityDefectScalarCount) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, &contour] {
    auto result = std::make_unique<std::vector<double>>();
    if (contour.size() < 4) {
      return result;
    }
    std::vector<int> hull_indexes;
    cv::convexHull(contour, hull_indexes, false, false);
    if (hull_indexes.size() < 3 || hull_indexes.size() == contour.size()) {
      return result;
    }
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hull_indexes, defects);
    if (defects.size() >
        static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) /
            kConvexityDefectScalarCount) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(defects.size() * kConvexityDefectScalarCount);
    const double maximum_depth =
        std::hypot(static_cast<double>(impl_->cols - 1),
                   static_cast<double>(impl_->rows - 1));
    for (const auto& defect : defects) {
      if (defect[0] < 0 || defect[1] < 0 || defect[2] < 0 || defect[3] < 0 ||
          static_cast<std::size_t>(defect[0]) >= contour.size() ||
          static_cast<std::size_t>(defect[1]) >= contour.size() ||
          static_cast<std::size_t>(defect[2]) >= contour.size()) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      const auto& start = contour[static_cast<std::size_t>(defect[0])];
      const auto& end = contour[static_cast<std::size_t>(defect[1])];
      const auto& farthest = contour[static_cast<std::size_t>(defect[2])];
      const double depth = static_cast<double>(defect[3]) / 256.0;
      if (!std::isfinite(depth) || depth > maximum_depth + 1.0e-9) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->insert(result->end(),
                     {static_cast<double>(start.x),
                      static_cast<double>(start.y),
                      static_cast<double>(end.x), static_cast<double>(end.y),
                      static_cast<double>(farthest.x),
                      static_cast<double>(farthest.y), depth});
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>> OpenCvContours::signed_distances_owned(
    double x, double y) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_analysis_point(impl_->rows, impl_->cols, x, y)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, x, y] {
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(impl_->contours.size());
    const double maximum_distance =
        std::hypot(static_cast<double>(impl_->cols - 1),
                   static_cast<double>(impl_->rows - 1));
    for (const auto& contour : impl_->contours) {
      const double distance =
          cv::pointPolygonTest(contour, cv::Point2d{x, y}, true);
      if (!std::isfinite(distance) ||
          std::abs(distance) > maximum_distance + 1.0e-9) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(std::abs(distance) <= 1.0e-9 ? 0.0 : distance);
    }
    return result;
  });
}

std::unique_ptr<OpenCvMat> OpenCvContours::filled_mask_owned(
    std::int64_t value) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !validation::valid_drawing_value(value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, value] {
    return render_contours_mask(impl_->contours, impl_->rows, impl_->cols, -1,
                                value, cv::FILLED);
  });
}

std::unique_ptr<OpenCvMat> OpenCvContours::outline_mask_owned(
    std::int64_t value, std::int64_t thickness) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_drawing_style(value, thickness)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, value, thickness] {
    return render_contours_mask(impl_->contours, impl_->rows, impl_->cols, -1,
                                value, static_cast<int>(thickness));
  });
}

std::unique_ptr<OpenCvMat> OpenCvContours::selected_filled_mask_owned(
    std::int64_t contour_index, std::int64_t value) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !validation::valid_drawing_value(value) ||
      contour_index < 0 ||
      static_cast<std::size_t>(contour_index) >= impl_->contours.size()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, contour_index, value] {
    return render_contours_mask(
        impl_->contours, impl_->rows, impl_->cols,
        static_cast<int>(contour_index), value, cv::FILLED);
  });
}

std::int64_t live_contour_analysis_count() noexcept {
  return live_contour_analyses.load();
}

}  // namespace terlan_opencv
