#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cmath>
#include <cstring>
#include <utility>

namespace terlan_opencv {
namespace {

std::atomic<std::int64_t> live_mats{0};
std::atomic<std::int64_t> live_component_analyses{0};

int mat_type(std::int64_t layout) noexcept {
  switch (layout) {
    case static_cast<std::int64_t>(PixelLayout::Gray):
      return CV_8UC1;
    case static_cast<std::int64_t>(PixelLayout::Bgr):
      return CV_8UC3;
    case static_cast<std::int64_t>(PixelLayout::Bgra):
      return CV_8UC4;
    default:
      return -1;
  }
}

int interpolation_code(std::int64_t interpolation) noexcept {
  switch (interpolation) {
    case static_cast<std::int64_t>(Interpolation::Nearest):
      return cv::INTER_NEAREST;
    case static_cast<std::int64_t>(Interpolation::Linear):
      return cv::INTER_LINEAR;
    case static_cast<std::int64_t>(Interpolation::Area):
      return cv::INTER_AREA;
    case static_cast<std::int64_t>(Interpolation::Cubic):
      return cv::INTER_CUBIC;
    default:
      return -1;
  }
}

int color_conversion_code(std::int64_t conversion) noexcept {
  switch (conversion) {
    case static_cast<std::int64_t>(ColorConversion::BgrToGray):
      return cv::COLOR_BGR2GRAY;
    case static_cast<std::int64_t>(ColorConversion::GrayToBgr):
      return cv::COLOR_GRAY2BGR;
    case static_cast<std::int64_t>(ColorConversion::BgrToBgra):
      return cv::COLOR_BGR2BGRA;
    case static_cast<std::int64_t>(ColorConversion::BgraToBgr):
      return cv::COLOR_BGRA2BGR;
    default:
      return -1;
  }
}

int decode_mode_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(DecodeMode::Unchanged):
      return cv::IMREAD_UNCHANGED;
    case static_cast<std::int64_t>(DecodeMode::Grayscale):
      return cv::IMREAD_GRAYSCALE;
    case static_cast<std::int64_t>(DecodeMode::Color):
      return cv::IMREAD_COLOR;
    default:
      return 0;
  }
}

const char* image_extension(std::int64_t format) noexcept {
  return format == static_cast<std::int64_t>(ImageFormat::Bmp) ? ".bmp"
                                                               : nullptr;
}

int border_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(BorderMode::Constant):
      return cv::BORDER_CONSTANT;
    case static_cast<std::int64_t>(BorderMode::Replicate):
      return cv::BORDER_REPLICATE;
    case static_cast<std::int64_t>(BorderMode::Reflect):
      return cv::BORDER_REFLECT;
    default:
      return -1;
  }
}

int flip_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(FlipMode::Both):
      return -1;
    case static_cast<std::int64_t>(FlipMode::Vertical):
      return 0;
    case static_cast<std::int64_t>(FlipMode::Horizontal):
      return 1;
    default:
      return 2;
  }
}

int rotate_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(RotateMode::Clockwise90):
      return cv::ROTATE_90_CLOCKWISE;
    case static_cast<std::int64_t>(RotateMode::Rotate180):
      return cv::ROTATE_180;
    case static_cast<std::int64_t>(RotateMode::CounterClockwise90):
      return cv::ROTATE_90_COUNTERCLOCKWISE;
    default:
      return -1;
  }
}

int threshold_code(std::int64_t mode) noexcept {
  switch (mode) {
    case static_cast<std::int64_t>(ThresholdMode::Binary):
      return cv::THRESH_BINARY;
    case static_cast<std::int64_t>(ThresholdMode::BinaryInverse):
      return cv::THRESH_BINARY_INV;
    case static_cast<std::int64_t>(ThresholdMode::Truncate):
      return cv::THRESH_TRUNC;
    case static_cast<std::int64_t>(ThresholdMode::ToZero):
      return cv::THRESH_TOZERO;
    case static_cast<std::int64_t>(ThresholdMode::ToZeroInverse):
      return cv::THRESH_TOZERO_INV;
    default:
      return -1;
  }
}

}  // namespace

struct OpenCvComponents::Impl final {
  Impl(cv::Mat labels, cv::Mat stats, cv::Mat centroids,
       std::int64_t component_count)
      : labels(std::move(labels)),
        stats(std::move(stats)),
        centroids(std::move(centroids)),
        component_count(component_count) {}

  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  std::int64_t component_count;
};

namespace {

std::unique_ptr<OpenCvMat> own(cv::Mat mat) {
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(mat)));
}

std::unique_ptr<OpenCvComponents> own_components(
    cv::Mat labels, cv::Mat stats, cv::Mat centroids,
    std::int64_t component_count) {
  return std::make_unique<OpenCvComponents>(
      std::make_unique<OpenCvComponents::Impl>(
          std::move(labels), std::move(stats), std::move(centroids),
          component_count));
}

bool valid_component_result(const cv::Mat& labels, const cv::Mat& stats,
                            const cv::Mat& centroids,
                            std::int64_t component_count,
                            std::int64_t rows,
                            std::int64_t cols) {
  if (component_count < 1 || labels.type() != CV_32SC1 ||
      labels.rows != rows || labels.cols != cols || !labels.isContinuous() ||
      stats.type() != CV_32SC1 || stats.rows != component_count ||
      stats.cols != cv::CC_STAT_MAX || centroids.type() != CV_64FC1 ||
      centroids.rows != component_count || centroids.cols != 2) {
    return false;
  }
  auto observed = std::vector<std::int64_t>(
      static_cast<std::size_t>(component_count), 0);
  auto sum_x = std::vector<std::int64_t>(
      static_cast<std::size_t>(component_count), 0);
  auto sum_y = std::vector<std::int64_t>(
      static_cast<std::size_t>(component_count), 0);
  for (int row = 0; row < labels.rows; ++row) {
    const int* values = labels.ptr<int>(row);
    for (int col = 0; col < labels.cols; ++col) {
      const int label = values[col];
      if (label < 0 || label >= component_count) {
        return false;
      }
      const auto index = static_cast<std::size_t>(label);
      ++observed[index];
      sum_x[index] += col;
      sum_y[index] += row;
    }
  }
  for (std::int64_t label = 1; label < component_count; ++label) {
    const int x = stats.at<int>(static_cast<int>(label), cv::CC_STAT_LEFT);
    const int y = stats.at<int>(static_cast<int>(label), cv::CC_STAT_TOP);
    const int width =
        stats.at<int>(static_cast<int>(label), cv::CC_STAT_WIDTH);
    const int height =
        stats.at<int>(static_cast<int>(label), cv::CC_STAT_HEIGHT);
    const int area = stats.at<int>(static_cast<int>(label), cv::CC_STAT_AREA);
    const double centroid_x =
        centroids.at<double>(static_cast<int>(label), 0);
    const double centroid_y =
        centroids.at<double>(static_cast<int>(label), 1);
    const auto index = static_cast<std::size_t>(label);
    if (observed[index] <= 0) {
      return false;
    }
    const double expected_x = static_cast<double>(sum_x[index]) /
                              static_cast<double>(observed[index]);
    const double expected_y = static_cast<double>(sum_y[index]) /
                              static_cast<double>(observed[index]);
    if (x < 0 || y < 0 || width <= 0 || height <= 0 || x > cols - width ||
        y > rows - height || area <= 0 ||
        area != observed[index] || !std::isfinite(centroid_x) ||
        !std::isfinite(centroid_y) || centroid_x < x ||
        centroid_x > x + width - 1 || centroid_y < y ||
        centroid_y > y + height - 1 ||
        std::abs(centroid_x - expected_x) > 1.0e-9 ||
        std::abs(centroid_y - expected_y) > 1.0e-9) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<OpenCvMat> apply_morphology(
    const cv::Mat& input, std::int64_t kernel_width,
    std::int64_t kernel_height, std::int64_t iterations,
    bool erosion) {
  const auto kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(static_cast<int>(kernel_width),
               static_cast<int>(kernel_height)));
  cv::Mat result;
  if (erosion) {
    cv::erode(input, result, kernel, cv::Point(-1, -1),
              static_cast<int>(iterations), cv::BORDER_REPLICATE);
  } else {
    cv::dilate(input, result, kernel, cv::Point(-1, -1),
               static_cast<int>(iterations), cv::BORDER_REPLICATE);
  }
  return own(std::move(result));
}

}  // namespace

OpenCvMat::OpenCvMat(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  ++live_mats;
}

OpenCvMat::~OpenCvMat() noexcept { --live_mats; }

std::int64_t OpenCvMat::rows() const noexcept {
  return impl_ == nullptr ? 0 : impl_->value.rows;
}

std::int64_t OpenCvMat::cols() const noexcept {
  return impl_ == nullptr ? 0 : impl_->value.cols;
}

std::int64_t OpenCvMat::channels() const noexcept {
  return impl_ == nullptr ? 0 : impl_->value.channels();
}

bool OpenCvMat::is_contiguous() const noexcept {
  return impl_ != nullptr && impl_->value.isContinuous();
}

std::unique_ptr<OpenCvMat> OpenCvMat::clone_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this] { return own(impl_->value.clone()); });
}

std::unique_ptr<std::vector<std::uint8_t>> OpenCvMat::to_u8() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  if (!impl_->value.isContinuous()) {
    failure::set(failure::NativeFailure::NonContiguous);
    return nullptr;
  }
  return failure::contain<std::vector<std::uint8_t>>([this] {
    const auto byte_count = impl_->value.total() * impl_->value.elemSize();
    return std::make_unique<std::vector<std::uint8_t>>(
        impl_->value.data, impl_->value.data + byte_count);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::resize_owned(
    std::int64_t width, std::int64_t height,
    std::int64_t interpolation) const noexcept {
  failure::clear();
  const int code = interpolation_code(interpolation);
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::image_shape(height, width,
                                                   impl_->value.channels());
  if (!shape.valid || code < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, width, height, code] {
    cv::Mat result;
    cv::resize(impl_->value, result,
               cv::Size(static_cast<int>(width), static_cast<int>(height)), 0.0,
               0.0, code);
    return own(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::cvt_color_owned(
    std::int64_t conversion) const noexcept {
  failure::clear();
  const int code = color_conversion_code(conversion);
  const auto shape =
      impl_ == nullptr
          ? validation::ColorConversionShape{false, 0, 0}
          : validation::color_conversion_shape(
                impl_->value.rows, impl_->value.cols, impl_->value.channels(),
                conversion);
  if (!shape.valid || code < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, code] {
    cv::Mat result;
    cv::cvtColor(impl_->value, result, code);
    return own(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::gaussian_blur_owned(
    std::int64_t kernel_width, std::int64_t kernel_height,
    double sigma_x) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_kernel_dimension(kernel_width) ||
      !validation::valid_kernel_dimension(kernel_height) ||
      !validation::valid_sigma(sigma_x)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, kernel_width, kernel_height, sigma_x] {
        cv::Mat result;
        cv::GaussianBlur(
            impl_->value, result,
            cv::Size(static_cast<int>(kernel_width),
                     static_cast<int>(kernel_height)),
            sigma_x);
        return own(std::move(result));
      });
}

std::unique_ptr<std::vector<std::uint8_t>> OpenCvMat::encode_owned(
    std::int64_t format) const noexcept {
  failure::clear();
  const char* extension = image_extension(format);
  if (impl_ == nullptr || extension == nullptr ||
      !validation::valid_image_format(format)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::uint8_t>>([this, extension] {
    auto encoded = std::make_unique<std::vector<std::uint8_t>>();
    if (!cv::imencode(extension, impl_->value, *encoded)) {
      failure::set(failure::NativeFailure::CodecUnavailable);
      return std::unique_ptr<std::vector<std::uint8_t>>{};
    }
    if (!validation::valid_encoded_output_size(encoded->size())) {
      failure::set(failure::NativeFailure::EncodedImageTooLarge);
      return std::unique_ptr<std::vector<std::uint8_t>>{};
    }
    return encoded;
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::crop_owned(
    std::int64_t x, std::int64_t y, std::int64_t width,
    std::int64_t height) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_crop_region(impl_->value.rows, impl_->value.cols, x,
                                     y, width, height) ||
      !validation::image_shape(height, width, impl_->value.channels()).valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, x, y, width, height] {
    const cv::Rect region(static_cast<int>(x), static_cast<int>(y),
                          static_cast<int>(width), static_cast<int>(height));
    return own(impl_->value(region).clone());
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::pad_owned(
    std::int64_t top, std::int64_t bottom, std::int64_t left,
    std::int64_t right, std::int64_t border_mode) const noexcept {
  failure::clear();
  const int code = border_code(border_mode);
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::padded_image_shape(
                               impl_->value.rows, impl_->value.cols,
                               impl_->value.channels(), top, bottom, left,
                               right);
  if (!shape.valid || code < 0 ||
      !validation::valid_border_mode(border_mode)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, top, bottom, left, right, code] {
        cv::Mat result;
        cv::copyMakeBorder(
            impl_->value, result, static_cast<int>(top),
            static_cast<int>(bottom), static_cast<int>(left),
            static_cast<int>(right), code, cv::Scalar::all(0));
        return own(std::move(result));
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::flip_owned(
    std::int64_t mode) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !validation::valid_flip_mode(mode) ||
      !validation::image_shape(impl_->value.rows, impl_->value.cols,
                               impl_->value.channels())
           .valid) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const int code = flip_code(mode);
  return failure::contain<OpenCvMat>([this, code] {
    cv::Mat result;
    cv::flip(impl_->value, result, code);
    return own(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::rotate_owned(
    std::int64_t mode) const noexcept {
  failure::clear();
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::rotated_image_shape(
                               impl_->value.rows, impl_->value.cols,
                               impl_->value.channels(), mode);
  const int code = rotate_code(mode);
  if (!shape.valid || code < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, code] {
    cv::Mat result;
    cv::rotate(impl_->value, result, code);
    return own(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::warp_affine_owned(
    double m00, double m01, double m02, double m10, double m11,
    double m12, std::int64_t width, std::int64_t height,
    std::int64_t interpolation, std::int64_t border_mode) const noexcept {
  failure::clear();
  const int interpolation_value = interpolation_code(interpolation);
  const int border_value = border_code(border_mode);
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::image_shape(
                               height, width, impl_->value.channels());
  if (!shape.valid || interpolation_value < 0 || border_value < 0 ||
      !validation::valid_affine_transform(m00, m01, m02, m10, m11, m12)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, m00, m01, m02, m10, m11, m12, width, height,
       interpolation_value, border_value] {
        const cv::Matx23d transform(m00, m01, m02, m10, m11, m12);
        cv::Mat result;
        cv::warpAffine(
            impl_->value, result, transform,
            cv::Size(static_cast<int>(width), static_cast<int>(height)),
            interpolation_value, border_value, cv::Scalar::all(0));
        return own(std::move(result));
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::warp_perspective_owned(
    double m00, double m01, double m02, double m10, double m11,
    double m12, double m20, double m21, double m22, std::int64_t width,
    std::int64_t height, std::int64_t interpolation,
    std::int64_t border_mode) const noexcept {
  failure::clear();
  const int interpolation_value = interpolation_code(interpolation);
  const int border_value = border_code(border_mode);
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::image_shape(
                               height, width, impl_->value.channels());
  if (!shape.valid || interpolation_value < 0 || border_value < 0 ||
      !validation::valid_perspective_transform(
          m00, m01, m02, m10, m11, m12, m20, m21, m22)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, m00, m01, m02, m10, m11, m12, m20, m21, m22, width, height,
       interpolation_value, border_value] {
        const cv::Matx33d transform(
            m00, m01, m02, m10, m11, m12, m20, m21, m22);
        cv::Mat result;
        cv::warpPerspective(
            impl_->value, result, transform,
            cv::Size(static_cast<int>(width), static_cast<int>(height)),
            interpolation_value, border_value, cv::Scalar::all(0));
        return own(std::move(result));
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::threshold_owned(
    double threshold_value, double maximum_value,
    std::int64_t mode) const noexcept {
  failure::clear();
  const int code = threshold_code(mode);
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::image_shape(
                               impl_->value.rows, impl_->value.cols,
                               impl_->value.channels());
  if (!shape.valid || code < 0 ||
      !validation::valid_threshold_mode(mode) ||
      !validation::valid_threshold_value(threshold_value) ||
      !validation::valid_threshold_value(maximum_value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, threshold_value, maximum_value, code] {
        cv::Mat result;
        cv::threshold(impl_->value, result, threshold_value, maximum_value,
                      code);
        return own(std::move(result));
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::erode_owned(
    std::int64_t kernel_width, std::int64_t kernel_height,
    std::int64_t iterations) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_morphology(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          kernel_width, kernel_height, iterations)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, kernel_width, kernel_height,
                                      iterations] {
    return apply_morphology(impl_->value, kernel_width, kernel_height,
                            iterations, true);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::dilate_owned(
    std::int64_t kernel_width, std::int64_t kernel_height,
    std::int64_t iterations) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_morphology(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          kernel_width, kernel_height, iterations)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, kernel_width, kernel_height,
                                      iterations] {
    return apply_morphology(impl_->value, kernel_width, kernel_height,
                            iterations, false);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::canny_owned(
    double low_threshold, double high_threshold,
    std::int64_t aperture_size, bool l2_gradient) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_canny(
          impl_->value.rows, impl_->value.cols, impl_->value.channels(),
          low_threshold, high_threshold, aperture_size)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, low_threshold, high_threshold, aperture_size, l2_gradient] {
        cv::Mat result;
        cv::Canny(impl_->value, result, low_threshold, high_threshold,
                  static_cast<int>(aperture_size), l2_gradient);
        return own(std::move(result));
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::normalize_min_max_owned(
    double minimum, double maximum) const noexcept {
  failure::clear();
  const auto shape = impl_ == nullptr
                         ? validation::ImageShape{false, 0, 0}
                         : validation::image_shape(
                               impl_->value.rows, impl_->value.cols,
                               impl_->value.channels());
  if (!shape.valid ||
      !validation::valid_normalization_range(minimum, maximum)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, minimum, maximum] {
    cv::Mat result;
    cv::normalize(impl_->value, result, minimum, maximum, cv::NORM_MINMAX, -1);
    return own(std::move(result));
  });
}

std::unique_ptr<OpenCvComponents> OpenCvMat::connected_components_owned(
    std::int64_t connectivity) const noexcept {
  failure::clear();
  if (impl_ == nullptr ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      !validation::valid_connectivity(connectivity)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvComponents>([this, connectivity] {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        impl_->value, labels, stats, centroids,
        static_cast<int>(connectivity), CV_32S, cv::CCL_DEFAULT);
    if (!valid_component_result(labels, stats, centroids, component_count,
                                impl_->value.rows, impl_->value.cols)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvComponents>{};
    }
    return own_components(
        std::move(labels), std::move(stats), std::move(centroids),
        component_count);
  });
}

OpenCvComponents::OpenCvComponents(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  ++live_component_analyses;
}

OpenCvComponents::~OpenCvComponents() noexcept {
  --live_component_analyses;
}

std::int64_t OpenCvComponents::foreground_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->component_count - 1;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvComponents::labels_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto labels = std::make_unique<std::vector<std::int64_t>>();
    labels->reserve(impl_->labels.total());
    for (int row = 0; row < impl_->labels.rows; ++row) {
      const int* values = impl_->labels.ptr<int>(row);
      for (int col = 0; col < impl_->labels.cols; ++col) {
        labels->push_back(values[col]);
      }
    }
    return labels;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvComponents::areas_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto areas = std::make_unique<std::vector<std::int64_t>>();
    areas->reserve(static_cast<std::size_t>(impl_->component_count - 1));
    for (std::int64_t label = 1; label < impl_->component_count; ++label) {
      areas->push_back(
          impl_->stats.at<int>(static_cast<int>(label), cv::CC_STAT_AREA));
    }
    return areas;
  });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvComponents::bounds_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this] {
    auto bounds = std::make_unique<std::vector<std::int64_t>>();
    bounds->reserve(
        static_cast<std::size_t>((impl_->component_count - 1) * 4));
    for (std::int64_t label = 1; label < impl_->component_count; ++label) {
      const int row = static_cast<int>(label);
      bounds->push_back(impl_->stats.at<int>(row, cv::CC_STAT_LEFT));
      bounds->push_back(impl_->stats.at<int>(row, cv::CC_STAT_TOP));
      bounds->push_back(impl_->stats.at<int>(row, cv::CC_STAT_WIDTH));
      bounds->push_back(impl_->stats.at<int>(row, cv::CC_STAT_HEIGHT));
    }
    return bounds;
  });
}

std::unique_ptr<std::vector<double>>
OpenCvComponents::centroids_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    auto centroids = std::make_unique<std::vector<double>>();
    centroids->reserve(
        static_cast<std::size_t>((impl_->component_count - 1) * 2));
    for (std::int64_t label = 1; label < impl_->component_count; ++label) {
      const int row = static_cast<int>(label);
      centroids->push_back(impl_->centroids.at<double>(row, 0));
      centroids->push_back(impl_->centroids.at<double>(row, 1));
    }
    return centroids;
  });
}

std::unique_ptr<OpenCvMat> mat_from_u8(
    rust::Slice<const std::uint8_t> bytes, std::int64_t rows,
    std::int64_t cols, std::int64_t layout) noexcept {
  failure::clear();
  const auto shape = validation::image_shape_for_layout(rows, cols, layout);
  const int type = mat_type(layout);
  if (!validation::valid_buffer_size(shape, bytes.size()) || type < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([bytes, rows, cols, type, shape] {
    cv::Mat result(static_cast<int>(rows), static_cast<int>(cols), type);
    std::memcpy(result.data, bytes.data(), shape.byte_count);
    return own(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> mat_zeros(
    std::int64_t rows, std::int64_t cols, std::int64_t layout) noexcept {
  failure::clear();
  const auto shape = validation::image_shape_for_layout(rows, cols, layout);
  const int type = mat_type(layout);
  if (!shape.valid || type < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([rows, cols, type] {
    return own(cv::Mat::zeros(static_cast<int>(rows), static_cast<int>(cols),
                              type));
  });
}

std::unique_ptr<OpenCvMat> decode_image(
    rust::Slice<const std::uint8_t> bytes, std::int64_t mode) noexcept {
  failure::clear();
  if (!validation::valid_decode_mode(mode)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  if (!validation::valid_bmp_input(bytes.data(), bytes.size())) {
    failure::set(bytes.size() > validation::kMaxEncodedImageBytes
                     ? failure::NativeFailure::EncodedImageTooLarge
                     : failure::NativeFailure::InvalidEncodedImage);
    return nullptr;
  }
  const int native_mode = decode_mode_code(mode);
  return failure::contain<OpenCvMat>([bytes, native_mode] {
    const std::vector<std::uint8_t> encoded(bytes.data(),
                                            bytes.data() + bytes.size());
    cv::Mat decoded = cv::imdecode(encoded, native_mode);
    if (decoded.empty()) {
      failure::set(failure::NativeFailure::InvalidEncodedImage);
      return std::unique_ptr<OpenCvMat>{};
    }
    if (!validation::valid_decoded_storage(decoded.total(),
                                           decoded.elemSize())) {
      failure::set(failure::NativeFailure::DecodedImageTooLarge);
      return std::unique_ptr<OpenCvMat>{};
    }
    return own(std::move(decoded));
  });
}

std::int64_t live_mat_count() noexcept { return live_mats.load(); }

std::int64_t live_component_analysis_count() noexcept {
  return live_component_analyses.load();
}

std::int64_t take_last_failure() noexcept { return failure::take(); }

}  // namespace terlan_opencv
