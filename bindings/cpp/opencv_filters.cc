#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <cmath>
#include <cstring>

#include <opencv2/imgproc.hpp>

namespace terlan_opencv {
namespace {

/// Lowers one validated symbolic binary threshold mode to OpenCV.
int binary_threshold_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(ThresholdMode::Binary):
      return cv::THRESH_BINARY;
    case static_cast<std::int64_t>(ThresholdMode::BinaryInverse):
      return cv::THRESH_BINARY_INV;
    default:
      return -1;
  }
}

/// Applies one validated OpenCV automatic threshold into fresh ownership.
std::unique_ptr<OpenCvMat> automatic_threshold_result(
    const cv::Mat& source, double maximum_value, int threshold_code,
    int algorithm_code) {
  cv::Mat result;
  const double selected = cv::threshold(source, result, 0.0, maximum_value,
                                        threshold_code | algorithm_code);
  if (!validation::valid_threshold_value(selected)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return own_matching_mat(std::move(result), source);
}

int morphology_code(std::int64_t operation) noexcept {
  switch (operation) {
    case static_cast<std::int64_t>(MorphologyOperation::Open):
      return cv::MORPH_OPEN;
    case static_cast<std::int64_t>(MorphologyOperation::Close):
      return cv::MORPH_CLOSE;
    case static_cast<std::int64_t>(MorphologyOperation::Gradient):
      return cv::MORPH_GRADIENT;
    case static_cast<std::int64_t>(MorphologyOperation::TopHat):
      return cv::MORPH_TOPHAT;
    case static_cast<std::int64_t>(MorphologyOperation::BlackHat):
      return cv::MORPH_BLACKHAT;
    default:
      return -1;
  }
}

}  // namespace

std::unique_ptr<OpenCvMat> OpenCvMat::median_blur_owned(
    std::int64_t kernel_size) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_median_blur(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          kernel_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, kernel_size] {
    cv::Mat result;
    cv::medianBlur(impl_->value, result, static_cast<int>(kernel_size));
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::bilateral_filter_owned(
    std::int64_t diameter, double sigma_color, double sigma_space) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_bilateral_filter(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          diameter, sigma_color, sigma_space)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, diameter, sigma_color, sigma_space] {
        cv::Mat result;
        cv::bilateralFilter(impl_->value, result, static_cast<int>(diameter),
                            sigma_color, sigma_space, cv::BORDER_REPLICATE);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::adaptive_threshold_owned(
    double maximum_value, std::int64_t adaptive_method,
    std::int64_t threshold_mode, std::int64_t block_size,
    double constant) const noexcept {
  failure::clear();
  const int adaptive_code =
      adaptive_method == static_cast<std::int64_t>(AdaptiveThresholdMethod::Mean)
          ? cv::ADAPTIVE_THRESH_MEAN_C
          : adaptive_method ==
                    static_cast<std::int64_t>(AdaptiveThresholdMethod::Gaussian)
                ? cv::ADAPTIVE_THRESH_GAUSSIAN_C
                : -1;
  const int threshold_code = binary_threshold_code(threshold_mode);
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      adaptive_code < 0 || threshold_code < 0 ||
      !validation::valid_adaptive_threshold(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          maximum_value, adaptive_method, threshold_mode, block_size,
          constant)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, maximum_value, adaptive_code, threshold_code, block_size,
       constant] {
        cv::Mat result;
        cv::adaptiveThreshold(impl_->value, result, maximum_value,
                              adaptive_code, threshold_code,
                              static_cast<int>(block_size), constant);
        return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::otsu_threshold_owned(
    double maximum_value, std::int64_t threshold_mode) const noexcept {
  failure::clear();
  const int threshold_code = binary_threshold_code(threshold_mode);
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      threshold_code < 0 ||
      !validation::valid_automatic_threshold(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          maximum_value, threshold_mode)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, maximum_value, threshold_code] {
        return automatic_threshold_result(impl_->value, maximum_value,
                                          threshold_code, cv::THRESH_OTSU);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::triangle_threshold_owned(
    double maximum_value, std::int64_t threshold_mode) const noexcept {
  failure::clear();
  const int threshold_code = binary_threshold_code(threshold_mode);
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      threshold_code < 0 ||
      !validation::valid_automatic_threshold(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          maximum_value, threshold_mode)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, maximum_value, threshold_code] {
        return automatic_threshold_result(impl_->value, maximum_value,
                                          threshold_code,
                                          cv::THRESH_TRIANGLE);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::box_blur_owned(
    std::int64_t kernel_width, std::int64_t kernel_height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_box_blur(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          kernel_width, kernel_height)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, kernel_width, kernel_height] {
    cv::Mat result;
    cv::blur(impl_->value, result,
             cv::Size(static_cast<int>(kernel_width),
                      static_cast<int>(kernel_height)),
             cv::Point(-1, -1), cv::BORDER_REPLICATE);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::stack_blur_owned(
    std::int64_t kernel_width, std::int64_t kernel_height) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_stack_blur(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          kernel_width, kernel_height)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, kernel_width, kernel_height] {
    cv::Mat result;
    cv::stackBlur(impl_->value, result,
                  cv::Size(static_cast<int>(kernel_width),
                           static_cast<int>(kernel_height)));
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::mean_shift_filter_owned(
    double spatial_radius, double color_radius,
    std::int64_t max_level) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_mean_shift_filter(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          spatial_radius, color_radius, max_level)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, spatial_radius, color_radius, max_level] {
        cv::Mat result;
        cv::pyrMeanShiftFiltering(impl_->value, result, spatial_radius,
                                  color_radius,
                                  static_cast<int>(max_level));
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::apply_lut_owned(
    rust::Slice<const std::uint8_t> table) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_lookup_table(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          table.size())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, table] {
    cv::Mat lookup(1, 256, CV_8UC1);
    std::memcpy(lookup.data, table.data(), table.size());
    cv::Mat result;
    cv::LUT(impl_->value, lookup, result);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::sharpen_owned(
    double amount) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_sharpen(impl_->value.rows, impl_->value.cols,
                                 impl_->value.channels(), amount)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, amount] {
    const cv::Mat kernel =
        (cv::Mat_<double>(3, 3) << 0.0, -amount, 0.0, -amount,
         1.0 + (4.0 * amount), -amount, 0.0, -amount, 0.0);
    cv::Mat result;
    cv::filter2D(impl_->value, result, -1, kernel, cv::Point(-1, -1), 0.0,
                 cv::BORDER_REPLICATE);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::laplacian_edges_owned(
    std::int64_t kernel_size) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_laplacian_edges(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          kernel_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, kernel_size] {
    cv::Mat signed_response;
    cv::Laplacian(impl_->value, signed_response, CV_16S,
                  static_cast<int>(kernel_size), 1.0, 0.0,
                  cv::BORDER_REPLICATE);
    cv::Mat absolute_response;
    cv::convertScaleAbs(signed_response, absolute_response);
    return own_matching_mat(std::move(absolute_response), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::sobel_edges_owned(
    std::int64_t x_order, std::int64_t y_order,
    std::int64_t kernel_size) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_sobel_edges(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          x_order, y_order, kernel_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, x_order, y_order, kernel_size] {
        cv::Mat signed_response;
        cv::Sobel(impl_->value, signed_response, CV_16S,
                  static_cast<int>(x_order), static_cast<int>(y_order),
                  static_cast<int>(kernel_size), 1.0, 0.0,
                  cv::BORDER_REPLICATE);
        cv::Mat absolute_response;
        cv::convertScaleAbs(signed_response, absolute_response);
        return own_matching_mat(std::move(absolute_response), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::scharr_edges_owned(
    std::int64_t x_order, std::int64_t y_order) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_scharr_edges(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          x_order, y_order)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, x_order, y_order] {
    cv::Mat signed_response;
    cv::Scharr(impl_->value, signed_response, CV_16S,
               static_cast<int>(x_order), static_cast<int>(y_order), 1.0, 0.0,
               cv::BORDER_REPLICATE);
    cv::Mat absolute_response;
    cv::convertScaleAbs(signed_response, absolute_response);
    return own_matching_mat(std::move(absolute_response), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::morphology_owned(
    std::int64_t operation, std::int64_t kernel_width,
    std::int64_t kernel_height, std::int64_t iterations) const noexcept {
  failure::clear();
  const int operation_code = morphology_code(operation);
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      operation_code < 0 ||
      !validation::valid_compound_morphology(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          operation, kernel_width, kernel_height, iterations)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, operation_code, kernel_width, kernel_height, iterations] {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(static_cast<int>(kernel_width),
                     static_cast<int>(kernel_height)));
        cv::Mat result;
        cv::morphologyEx(impl_->value, result, operation_code, kernel,
                         cv::Point(-1, -1), static_cast<int>(iterations),
                         cv::BORDER_REPLICATE);
        return own_matching_mat(std::move(result), impl_->value);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::pyr_down_owned() const noexcept {
  failure::clear();
  const auto shape =
      impl_ == nullptr
          ? validation::ImageShape{false, 0, 0}
          : validation::pyramid_down_shape(
                impl_->value.rows, impl_->value.cols, impl_->value.channels());
  if (!shape.valid || impl_->value.depth() != CV_8U) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const auto rows = (static_cast<std::int64_t>(impl_->value.rows) + 1) / 2;
  const auto cols = (static_cast<std::int64_t>(impl_->value.cols) + 1) / 2;
  return failure::contain<OpenCvMat>([this, rows, cols] {
    cv::Mat result;
    cv::pyrDown(impl_->value, result,
                cv::Size(static_cast<int>(cols), static_cast<int>(rows)),
                cv::BORDER_DEFAULT);
    return own_resized_mat(std::move(result), impl_->value, rows, cols);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::pyr_up_owned() const noexcept {
  failure::clear();
  const auto shape =
      impl_ == nullptr
          ? validation::ImageShape{false, 0, 0}
          : validation::pyramid_up_shape(
                impl_->value.rows, impl_->value.cols, impl_->value.channels());
  if (!shape.valid || impl_->value.depth() != CV_8U) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const auto rows = static_cast<std::int64_t>(impl_->value.rows) * 2;
  const auto cols = static_cast<std::int64_t>(impl_->value.cols) * 2;
  return failure::contain<OpenCvMat>([this, rows, cols] {
    cv::Mat result;
    cv::pyrUp(impl_->value, result,
              cv::Size(static_cast<int>(cols), static_cast<int>(rows)),
              cv::BORDER_DEFAULT);
    return own_resized_mat(std::move(result), impl_->value, rows, cols);
  });
}

std::unique_ptr<OpenCvMat>
OpenCvMat::manhattan_distance_transform_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    cv::Mat result;
    cv::distanceTransform(impl_->value, result, cv::DIST_L1,
                          cv::DIST_MASK_3, CV_8U);
    return own_grayscale_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<std::vector<double>>
OpenCvMat::euclidean_distance_transform_values_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels())) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    if (cv::countNonZero(impl_->value) ==
        static_cast<int>(impl_->value.total())) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<std::vector<double>>{};
    }
    cv::Mat distances;
    cv::distanceTransform(impl_->value, distances, cv::DIST_L2,
                          cv::DIST_MASK_PRECISE, CV_32F);
    auto result = copy_finite_f32_values(distances, impl_->value.rows,
                                         impl_->value.cols);
    if (result == nullptr) {
      return std::unique_ptr<std::vector<double>>{};
    }
    const double maximum = std::hypot(
        static_cast<double>(impl_->value.rows - 1),
        static_cast<double>(impl_->value.cols - 1));
    for (const double value : *result) {
      if (value < 0.0 || value > maximum + 1.0e-4) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
    }
    return result;
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::flood_fill_owned(
    std::int64_t seed_x, std::int64_t seed_y, std::int64_t fill_value,
    std::int64_t connectivity) const noexcept {
  failure::clear();
  if (impl_ == nullptr || impl_->value.depth() != CV_8U ||
      !validation::valid_flood_fill(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          seed_x, seed_y, fill_value, connectivity)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, seed_x, seed_y, fill_value, connectivity] {
        cv::Mat result = impl_->value.clone();
        const int filled = cv::floodFill(
            result, cv::Point(static_cast<int>(seed_x),
                              static_cast<int>(seed_y)),
            cv::Scalar::all(static_cast<double>(fill_value)), nullptr,
            cv::Scalar(), cv::Scalar(), static_cast<int>(connectivity));
        if (filled <= 0) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMat>{};
        }
        return own_matching_mat(std::move(result), impl_->value);
      });
}

}  // namespace terlan_opencv
