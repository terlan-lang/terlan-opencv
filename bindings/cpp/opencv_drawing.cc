#include "opencv_adapter.hpp"
#include "opencv_drawing_validation.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"

#include <opencv2/imgproc.hpp>

#include <utility>

namespace terlan_opencv {

std::unique_ptr<OpenCvMat> OpenCvMat::draw_line_owned(
    std::int64_t start_x, std::int64_t start_y, std::int64_t end_x,
    std::int64_t end_y, std::int64_t value,
    std::int64_t thickness) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_line_drawing(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          start_x, start_y, end_x, end_y, value, thickness)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, start_x, start_y, end_x, end_y, value, thickness] {
        cv::Mat result = impl_->value.clone();
        cv::line(result,
                 cv::Point(static_cast<int>(start_x),
                           static_cast<int>(start_y)),
                 cv::Point(static_cast<int>(end_x),
                           static_cast<int>(end_y)),
                 cv::Scalar(static_cast<double>(value)),
                 static_cast<int>(thickness), cv::LINE_8, 0);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::draw_rectangle_owned(
    std::int64_t x, std::int64_t y, std::int64_t width,
    std::int64_t height, std::int64_t value,
    std::int64_t thickness) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_rectangle_drawing(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(), x, y,
          width, height, value, thickness)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, x, y, width, height, value, thickness] {
        cv::Mat result = impl_->value.clone();
        cv::rectangle(result,
                      cv::Rect(static_cast<int>(x), static_cast<int>(y),
                               static_cast<int>(width),
                               static_cast<int>(height)),
                      cv::Scalar(static_cast<double>(value)),
                      static_cast<int>(thickness), cv::LINE_8, 0);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::draw_circle_owned(
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius,
    std::int64_t value, std::int64_t thickness) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_circle_drawing(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          center_x, center_y, radius, value, thickness)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, center_x, center_y, radius, value, thickness] {
        cv::Mat result = impl_->value.clone();
        cv::circle(result,
                   cv::Point(static_cast<int>(center_x),
                             static_cast<int>(center_y)),
                   static_cast<int>(radius),
                   cv::Scalar(static_cast<double>(value)),
                   static_cast<int>(thickness), cv::LINE_8, 0);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::fill_rectangle_owned(
    std::int64_t x, std::int64_t y, std::int64_t width,
    std::int64_t height, std::int64_t value) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_rectangle_fill(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(), x, y,
          width, height, value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, x, y, width, height, value] {
    cv::Mat result = impl_->value.clone();
    cv::rectangle(result,
                  cv::Rect(static_cast<int>(x), static_cast<int>(y),
                           static_cast<int>(width), static_cast<int>(height)),
                  cv::Scalar(static_cast<double>(value)), cv::FILLED,
                  cv::LINE_8, 0);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::fill_circle_owned(
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius,
    std::int64_t value) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_circle_fill(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          center_x, center_y, radius, value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, center_x, center_y, radius, value] {
        cv::Mat result = impl_->value.clone();
        cv::circle(result,
                   cv::Point(static_cast<int>(center_x),
                             static_cast<int>(center_y)),
                   static_cast<int>(radius),
                   cv::Scalar(static_cast<double>(value)), cv::FILLED,
                   cv::LINE_8, 0);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::draw_cross_owned(
    std::int64_t center_x, std::int64_t center_y, std::int64_t size,
    std::int64_t value, std::int64_t thickness) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_cross_drawing(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          center_x, center_y, size, value, thickness)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, center_x, center_y, size, value, thickness] {
        cv::Mat result = impl_->value.clone();
        cv::drawMarker(result,
                       cv::Point(static_cast<int>(center_x),
                                 static_cast<int>(center_y)),
                       cv::Scalar(static_cast<double>(value)), cv::MARKER_CROSS,
                       static_cast<int>(size), static_cast<int>(thickness),
                       cv::LINE_8);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::draw_ellipse_owned(
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x,
    std::int64_t radius_y, std::int64_t value,
    std::int64_t thickness) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_ellipse_drawing(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          center_x, center_y, radius_x, radius_y, value, thickness)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, center_x, center_y, radius_x, radius_y, value, thickness] {
        cv::Mat result = impl_->value.clone();
        cv::ellipse(result,
                    cv::Point(static_cast<int>(center_x),
                              static_cast<int>(center_y)),
                    cv::Size(static_cast<int>(radius_x),
                             static_cast<int>(radius_y)),
                    0.0, 0.0, 360.0, cv::Scalar(static_cast<double>(value)),
                    static_cast<int>(thickness), cv::LINE_8, 0);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::fill_ellipse_owned(
    std::int64_t center_x, std::int64_t center_y, std::int64_t radius_x,
    std::int64_t radius_y, std::int64_t value) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_ellipse_fill(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          center_x, center_y, radius_x, radius_y, value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, center_x, center_y, radius_x, radius_y, value] {
        cv::Mat result = impl_->value.clone();
        cv::ellipse(result,
                    cv::Point(static_cast<int>(center_x),
                              static_cast<int>(center_y)),
                    cv::Size(static_cast<int>(radius_x),
                             static_cast<int>(radius_y)),
                    0.0, 0.0, 360.0, cv::Scalar(static_cast<double>(value)),
                    cv::FILLED, cv::LINE_8, 0);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

}  // namespace terlan_opencv
