#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace terlan_opencv {

std::unique_ptr<std::vector<double>> OpenCvMat::detect_circles_owned(
    double minimum_distance, double edge_threshold, double center_threshold,
    std::int64_t minimum_radius, std::int64_t maximum_radius) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_circle_detection(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          minimum_distance, edge_threshold, center_threshold, minimum_radius,
          maximum_radius)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [this, minimum_distance, edge_threshold, center_threshold,
       minimum_radius, maximum_radius] {
        std::vector<cv::Vec3f> detected;
        cv::HoughCircles(impl_->value, detected, cv::HOUGH_GRADIENT, 1.0,
                         minimum_distance, edge_threshold, center_threshold,
                         static_cast<int>(minimum_radius),
                         static_cast<int>(maximum_radius));
        if (detected.size() > validation::kMaxDetectedCircles) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<double>>{};
        }
        for (const auto& circle : detected) {
          const double x = circle[0];
          const double y = circle[1];
          const double radius = circle[2];
          if (!std::isfinite(x) || !std::isfinite(y) ||
              !std::isfinite(radius) || x < 0.0 || y < 0.0 ||
              x >= impl_->value.cols || y >= impl_->value.rows ||
              radius + 1.0 < minimum_radius ||
              radius > maximum_radius + 1.0) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<std::vector<double>>{};
          }
        }
        std::sort(detected.begin(), detected.end(), [](const cv::Vec3f& first,
                                                       const cv::Vec3f& second) {
          if (first[1] != second[1]) {
            return first[1] < second[1];
          }
          if (first[0] != second[0]) {
            return first[0] < second[0];
          }
          return first[2] < second[2];
        });
        auto circles = std::make_unique<std::vector<double>>();
        circles->reserve(detected.size() * 3);
        for (const auto& circle : detected) {
          circles->push_back(circle[0]);
          circles->push_back(circle[1]);
          circles->push_back(circle[2]);
        }
        return circles;
      });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::detect_lines_owned(std::int64_t vote_threshold) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_line_detection(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          vote_threshold)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this, vote_threshold] {
    cv::Mat binary;
    cv::compare(impl_->value, 0, binary, cv::CMP_NE);
    std::vector<cv::Vec2f> detected;
    cv::HoughLines(binary, detected, 1.0, CV_PI / 180.0,
                   static_cast<int>(vote_threshold));
    if (detected.size() > validation::kMaxDetectedLines) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<double>>{};
    }

    const double maximum_rho =
        std::hypot(static_cast<double>(impl_->value.cols - 1),
                   static_cast<double>(impl_->value.rows - 1));
    for (const auto& line : detected) {
      const double rho = line[0];
      const double theta = line[1];
      if (!std::isfinite(rho) || !std::isfinite(theta) ||
          std::abs(rho) > maximum_rho + 1.0 || theta < 0.0 ||
          theta >= CV_PI) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
    }
    std::sort(detected.begin(), detected.end(), [](const cv::Vec2f& first,
                                                   const cv::Vec2f& second) {
      if (first[1] != second[1]) {
        return first[1] < second[1];
      }
      return first[0] < second[0];
    });

    auto result = std::make_unique<std::vector<double>>();
    result->reserve(detected.size() * 2);
    for (const auto& line : detected) {
      result->push_back(line[0]);
      result->push_back(line[1]);
    }
    return result;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMat::line_segments_owned(std::int64_t vote_threshold,
                               std::int64_t minimum_length,
                               std::int64_t maximum_gap) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_line_segment_detection(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          vote_threshold, minimum_length, maximum_gap)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this, vote_threshold, minimum_length, maximum_gap] {
        cv::Mat binary;
        cv::compare(impl_->value, 0, binary, cv::CMP_NE);
        std::vector<cv::Vec4i> detected;
        cv::HoughLinesP(binary, detected, 1.0, CV_PI / 180.0,
                        static_cast<int>(vote_threshold),
                        static_cast<double>(minimum_length),
                        static_cast<double>(maximum_gap));
        if (detected.size() > validation::kMaxDetectedLineSegments) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<std::int64_t>>{};
        }

        std::vector<std::array<int, 4>> normalized;
        normalized.reserve(detected.size());
        for (const auto& line : detected) {
          int x1 = line[0];
          int y1 = line[1];
          int x2 = line[2];
          int y2 = line[3];
          if (x2 < x1 || (x2 == x1 && y2 < y1)) {
            std::swap(x1, x2);
            std::swap(y1, y2);
          }
          const double length = std::hypot(static_cast<double>(x2 - x1),
                                           static_cast<double>(y2 - y1));
          if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 ||
              x1 >= impl_->value.cols || x2 >= impl_->value.cols ||
              y1 >= impl_->value.rows || y2 >= impl_->value.rows ||
              (x1 == x2 && y1 == y2) || !std::isfinite(length) ||
              length + 1.0e-9 < static_cast<double>(minimum_length)) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<std::vector<std::int64_t>>{};
          }
          normalized.push_back({x1, y1, x2, y2});
        }
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(
            std::unique(normalized.begin(), normalized.end()),
            normalized.end());

        auto result = std::make_unique<std::vector<std::int64_t>>();
        result->reserve(normalized.size() * 4);
        for (const auto& line : normalized) {
          result->insert(result->end(), line.begin(), line.end());
        }
        return result;
      });
}

}  // namespace terlan_opencv
