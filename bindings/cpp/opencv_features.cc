#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_mask_analysis.hpp"
#include "opencv_validation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace terlan_opencv {

std::unique_ptr<std::vector<double>> OpenCvMat::hu_moments_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    double values[7] = {};
    cv::HuMoments(cv::moments(impl_->value, true), values);
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(7);
    for (const double value : values) {
      if (!std::isfinite(value)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(value);
    }
    return result;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::non_zero_bounds_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    const cv::Rect bounds = cv::boundingRect(impl_->value);
    auto result = std::make_unique<std::vector<std::int64_t>>();
    if (bounds.empty()) {
      return result;
    }
    if (bounds.x < 0 || bounds.y < 0 || bounds.width <= 0 ||
        bounds.height <= 0 || bounds.x + bounds.width > impl_->value.cols ||
        bounds.y + bounds.height > impl_->value.rows) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    result->reserve(4);
    result->push_back(bounds.x);
    result->push_back(bounds.y);
    result->push_back(bounds.width);
    result->push_back(bounds.height);
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::non_zero_centroid_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    const cv::Moments moments = cv::moments(impl_->value, true);
    auto result = std::make_unique<std::vector<double>>();
    if (!std::isfinite(moments.m00) || !std::isfinite(moments.m10) ||
        !std::isfinite(moments.m01) || moments.m00 < 0.0 ||
        moments.m00 > static_cast<double>(impl_->value.total()) ||
        std::trunc(moments.m00) != moments.m00) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    if (moments.m00 == 0.0) {
      if (moments.m10 != 0.0 || moments.m01 != 0.0) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      return result;
    }
    const double x = moments.m10 / moments.m00;
    const double y = moments.m01 / moments.m00;
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || y < 0.0 ||
        x > static_cast<double>(impl_->value.cols - 1) ||
        y > static_cast<double>(impl_->value.rows - 1)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(2);
    result->push_back(x);
    result->push_back(y);
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::non_zero_enclosing_circle_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    auto result = std::make_unique<std::vector<double>>();
    if (points.empty()) {
      return result;
    }
    cv::Point2f center;
    float radius = 0.0F;
    if (!normalized_minimum_enclosing_circle(
            points, impl_->value.rows, impl_->value.cols, center, radius)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(3);
    result->push_back(center.x);
    result->push_back(center.y);
    result->push_back(radius);
    return result;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::non_zero_convex_hull_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto result = std::make_unique<std::vector<std::int64_t>>();
    if (points.empty()) {
      return result;
    }

    std::vector<cv::Point> hull;
    cv::convexHull(points, hull, false, true);
    if (hull.empty() || hull.size() > points.size() ||
        (hull.size() >= 3 && !cv::isContourConvex(hull))) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }

    std::vector<unsigned char> seen(impl_->value.total(), 0);
    for (const auto& point : hull) {
      if (point.x < 0 || point.x >= impl_->value.cols || point.y < 0 ||
          point.y >= impl_->value.rows ||
          impl_->value.at<unsigned char>(point.y, point.x) == 0) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      const auto index = static_cast<std::size_t>(point.y) *
                             static_cast<std::size_t>(impl_->value.cols) +
                         static_cast<std::size_t>(point.x);
      if (seen[index] != 0) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      seen[index] = 1;
    }

    rotate_topmost_leftmost(hull);
    result->reserve(hull.size() * 2);
    for (const auto& point : hull) {
      result->push_back(point.x);
      result->push_back(point.y);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::non_zero_min_area_box_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    auto result = std::make_unique<std::vector<double>>();
    if (points.empty()) {
      return result;
    }

    std::vector<cv::Point2f> corners;
    if (!normalized_minimum_area_box(points, impl_->value.rows,
                                     impl_->value.cols, corners)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(corners.size() * 2);
    for (const auto& corner : corners) {
      const double x = std::abs(corner.x) <= 1.0e-6 ? 0.0 : corner.x;
      const double y = std::abs(corner.y) <= 1.0e-6 ? 0.0 : corner.y;
      result->push_back(x);
      result->push_back(y);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::non_zero_enclosing_triangle_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    auto result = std::make_unique<std::vector<double>>();
    if (points.empty()) {
      return result;
    }

    std::vector<cv::Point2f> triangle;
    if (!normalized_minimum_enclosing_triangle(
            points, impl_->value.rows, impl_->value.cols, triangle)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(triangle.size() * 2);
    for (const auto& point : triangle) {
      const double x = std::abs(point.x) <= 1.0e-6 ? 0.0 : point.x;
      const double y = std::abs(point.y) <= 1.0e-6 ? 0.0 : point.y;
      result->push_back(x);
      result->push_back(y);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::non_zero_fit_line_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    auto result = std::make_unique<std::vector<double>>();
    if (points.size() < 2) {
      return result;
    }

    std::array<double, 4> line{};
    if (!normalized_fitted_line(points, impl_->value.rows, impl_->value.cols,
                                line)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(4);
    result->insert(result->end(), line.begin(), line.end());
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::non_zero_fit_ellipse_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    auto result = std::make_unique<std::vector<double>>();
    if (!supports_fitted_ellipse(points)) {
      return result;
    }
    std::array<double, 5> ellipse{};
    if (!normalized_fitted_ellipse(points, impl_->value.rows,
                                   impl_->value.cols, ellipse)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    result->reserve(5);
    result->insert(result->end(), ellipse.begin(), ellipse.end());
    return result;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::non_zero_approximate_hull_owned(double epsilon) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_hull_approximation(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          epsilon)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this, epsilon] {
    std::vector<cv::Point> points;
    if (!collect_non_zero_points(impl_->value, points)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto result = std::make_unique<std::vector<std::int64_t>>();
    if (points.empty()) {
      return result;
    }

    std::vector<cv::Point> hull;
    cv::convexHull(points, hull, false, true);
    if (hull.empty() || hull.size() > points.size() ||
        (hull.size() >= 3 && !cv::isContourConvex(hull))) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }

    std::vector<cv::Point> approximation;
    if (hull.size() <= 2) {
      approximation = hull;
    } else {
      cv::approxPolyDP(hull, approximation, epsilon, true);
    }
    if (approximation.empty() || approximation.size() > hull.size() ||
        (approximation.size() >= 3 &&
         !cv::isContourConvex(approximation))) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }

    std::vector<unsigned char> seen(impl_->value.total(), 0);
    for (const auto& point : approximation) {
      if (point.x < 0 || point.x >= impl_->value.cols || point.y < 0 ||
          point.y >= impl_->value.rows ||
          impl_->value.at<unsigned char>(point.y, point.x) == 0) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      const auto index = static_cast<std::size_t>(point.y) *
                             static_cast<std::size_t>(impl_->value.cols) +
                         static_cast<std::size_t>(point.x);
      if (seen[index] != 0) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      seen[index] = 1;
    }

    rotate_topmost_leftmost(approximation);
    result->reserve(approximation.size() * 2);
    for (const auto& point : approximation) {
      result->push_back(point.x);
      result->push_back(point.y);
    }
    return result;
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::shape_match_scores_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !valid_u8_grayscale_image(impl_->value) ||
      !valid_u8_grayscale_image(other.impl_->value) ||
      !validation::valid_shape_match(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          other.impl_->value.rows, other.impl_->value.cols,
          other.impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, &other] {
    cv::Mat first_binary;
    cv::Mat second_binary;
    cv::compare(impl_->value, cv::Scalar::all(0), first_binary, cv::CMP_NE);
    cv::compare(other.impl_->value, cv::Scalar::all(0), second_binary,
                cv::CMP_NE);
    if (cv::countNonZero(first_binary) == 0 ||
        cv::countNonZero(second_binary) == 0) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<std::vector<double>>{};
    }

    std::array<double, 3> scores{};
    if (!normalized_shape_match_scores(first_binary, second_binary, scores)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(3);
    result->insert(result->end(), scores.begin(), scores.end());
    return result;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::match_template_sqdiff_owned(const OpenCvMat& pattern) const noexcept {
  failure::clear();
  if (impl_ == nullptr || pattern.impl_ == nullptr ||
      impl_->value.depth() != CV_8U ||
      impl_->value.type() != pattern.impl_->value.type() ||
      !validation::valid_template_match(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          pattern.impl_->value.rows, pattern.impl_->value.cols,
          pattern.impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, &pattern] {
    cv::Mat result;
    cv::matchTemplate(impl_->value, pattern.impl_->value, result,
                      cv::TM_SQDIFF);
    const int expected_rows = impl_->value.rows - pattern.impl_->value.rows + 1;
    const int expected_cols = impl_->value.cols - pattern.impl_->value.cols + 1;
    auto scores = copy_finite_f32_values(result, expected_rows, expected_cols);
    if (scores == nullptr) {
      return std::unique_ptr<std::vector<double>>{};
    }
    for (double& score : *scores) {
      if (score < -1.0e-3) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      score = score < 0.0 ? 0.0 : score;
    }
    return scores;
  });
}

}  // namespace terlan_opencv
