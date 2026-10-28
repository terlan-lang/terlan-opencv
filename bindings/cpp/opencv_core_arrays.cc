#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace terlan_opencv {
namespace {

/// Converts one reduction dimension to its OpenCV axis.
int reduction_dimension_code(std::int64_t dimension) noexcept {
  switch (dimension) {
    case static_cast<std::int64_t>(ReductionDimension::AcrossRows):
      return 0;
    case static_cast<std::int64_t>(ReductionDimension::AcrossColumns):
      return 1;
    default:
      return -1;
  }
}

/// Converts one reduction operation to its OpenCV value.
int reduction_operation_code(std::int64_t operation) noexcept {
  switch (operation) {
    case static_cast<std::int64_t>(ReductionOperation::Sum):
      return cv::REDUCE_SUM;
    case static_cast<std::int64_t>(ReductionOperation::Average):
      return cv::REDUCE_AVG;
    case static_cast<std::int64_t>(ReductionOperation::Maximum):
      return cv::REDUCE_MAX;
    case static_cast<std::int64_t>(ReductionOperation::Minimum):
      return cv::REDUCE_MIN;
    default:
      return -1;
  }
}

/// Converts one sort dimension and order to OpenCV flags.
int sort_flags(std::int64_t dimension, std::int64_t order) noexcept {
  int flags = 0;
  switch (dimension) {
    case static_cast<std::int64_t>(SortDimension::EachRow):
      flags = cv::SORT_EVERY_ROW;
      break;
    case static_cast<std::int64_t>(SortDimension::EachColumn):
      flags = cv::SORT_EVERY_COLUMN;
      break;
    default:
      return -1;
  }
  switch (order) {
    case static_cast<std::int64_t>(SortOrder::Ascending):
      return flags | cv::SORT_ASCENDING;
    case static_cast<std::int64_t>(SortOrder::Descending):
      return flags | cv::SORT_DESCENDING;
    default:
      return -1;
  }
}

/// Reports whether one matrix is a supported finite numeric transform input.
bool valid_float_matrix(const cv::Mat& matrix) noexcept {
  return valid_owned_matrix(matrix) &&
         (matrix.depth() == CV_32F || matrix.depth() == CV_64F) &&
         cv::checkRange(matrix, true, nullptr);
}

/// Copies one finite OpenCV scalar through the matrix channel count.
std::unique_ptr<std::vector<double>> copy_finite_channels(
    const cv::Scalar& scalar, int channels) {
  if (channels <= 0 || channels > 4) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  auto values = std::make_unique<std::vector<double>>();
  values->reserve(static_cast<std::size_t>(channels));
  for (int channel = 0; channel < channels; ++channel) {
    if (!std::isfinite(scalar[channel])) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    values->push_back(scalar[channel]);
  }
  return values;
}

/// Wraps one checked matrix result in fresh package ownership.
std::unique_ptr<OpenCvMat> own_valid_matrix(cv::Mat matrix) {
  if (!valid_owned_matrix(matrix)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(matrix)));
}

/// Applies one finite floating-point transform into fresh ownership.
template <typename Transform>
std::unique_ptr<OpenCvMat> transform_float_matrix(
    const cv::Mat& source, Transform transform) {
  cv::Mat result;
  transform(source, result);
  if (!valid_float_matrix(result)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return own_matching_mat(std::move(result), source);
}

}  // namespace

std::unique_ptr<OpenCvMat> mat_random_normal(
    std::int64_t rows, std::int64_t cols, std::int64_t depth, double mean,
    double standard_deviation, std::int64_t seed) noexcept {
  failure::clear();
  const bool valid_depth = depth == static_cast<std::int64_t>(MatrixDepth::F32) ||
                           depth == static_cast<std::int64_t>(MatrixDepth::F64);
  const auto shape = validation::matrix_shape(
      rows, cols, 1, depth == static_cast<std::int64_t>(MatrixDepth::F32) ? 4 : 8);
  if (!valid_depth || !shape.valid || !std::isfinite(mean) ||
      !std::isfinite(standard_deviation) || standard_deviation <= 0.0 ||
      seed < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [rows, cols, depth, mean, standard_deviation, seed] {
        cv::Mat result(static_cast<int>(rows), static_cast<int>(cols),
                       static_cast<int>(depth));
        cv::RNG random(static_cast<std::uint64_t>(seed));
        random.fill(result, cv::RNG::NORMAL, mean, standard_deviation);
        if (!valid_float_matrix(result)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMat>{};
        }
        return own_valid_matrix(std::move(result));
      });
}

std::unique_ptr<std::vector<double>> OpenCvMat::sum_values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      impl_->value.channels() > 4) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    return copy_finite_channels(cv::sum(impl_->value),
                                impl_->value.channels());
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::mean_values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      impl_->value.channels() > 4) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    return copy_finite_channels(cv::mean(impl_->value),
                                impl_->value.channels());
  });
}

std::unique_ptr<std::vector<double>> OpenCvMat::standard_deviation_values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      impl_->value.channels() > 4) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    cv::Scalar mean;
    cv::Scalar standard_deviation;
    cv::meanStdDev(impl_->value, mean, standard_deviation);
    return copy_finite_channels(standard_deviation,
                                impl_->value.channels());
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::normalize_norm_owned(
    double target, std::int64_t norm) const noexcept {
  failure::clear();
  int norm_type = -1;
  switch (norm) {
    case static_cast<std::int64_t>(MatrixNorm::L1):
      norm_type = cv::NORM_L1;
      break;
    case static_cast<std::int64_t>(MatrixNorm::L2):
      norm_type = cv::NORM_L2;
      break;
    case static_cast<std::int64_t>(MatrixNorm::Infinity):
      norm_type = cv::NORM_INF;
      break;
    default:
      break;
  }
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      !std::isfinite(target) || target <= 0.0 || norm_type < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, target, norm_type] {
    cv::Mat result;
    cv::normalize(impl_->value, result, target, 0.0, norm_type);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::reduce_owned(
    std::int64_t dimension, std::int64_t operation) const noexcept {
  failure::clear();
  const int checked_dimension = reduction_dimension_code(dimension);
  const int checked_operation = reduction_operation_code(operation);
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      checked_dimension < 0 || checked_operation < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, checked_dimension, checked_operation] {
        cv::Mat result;
        const int output_depth =
            checked_operation == cv::REDUCE_SUM ||
                    checked_operation == cv::REDUCE_AVG
                ? CV_64F
                : -1;
        cv::reduce(impl_->value, result, checked_dimension,
                   checked_operation, output_depth);
        return own_valid_matrix(std::move(result));
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::sort_owned(
    std::int64_t dimension, std::int64_t order) const noexcept {
  failure::clear();
  const int flags = sort_flags(dimension, order);
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      impl_->value.channels() != 1 || flags < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, flags] {
    cv::Mat result;
    cv::sort(impl_->value, result, flags);
    return own_matching_mat(std::move(result), impl_->value);
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::sort_indices_owned(
    std::int64_t dimension, std::int64_t order) const noexcept {
  failure::clear();
  const int flags = sort_flags(dimension, order);
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      impl_->value.channels() != 1 || flags < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, flags] {
    cv::Mat result;
    cv::sortIdx(impl_->value, result, flags);
    if (result.rows != impl_->value.rows ||
        result.cols != impl_->value.cols || result.type() != CV_32SC1) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<OpenCvMat>{};
    }
    return own_valid_matrix(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::exponential_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_float_matrix(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    return transform_float_matrix(
        impl_->value, [](const cv::Mat& source, cv::Mat& result) {
          cv::exp(source, result);
        });
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::logarithm_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_float_matrix(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  cv::Mat flattened = impl_->value.reshape(1);
  double minimum = 0.0;
  cv::minMaxLoc(flattened, &minimum, nullptr);
  if (minimum <= 0.0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    return transform_float_matrix(
        impl_->value, [](const cv::Mat& source, cv::Mat& result) {
          cv::log(source, result);
        });
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::power_owned(
    double exponent) const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_float_matrix(impl_->value) ||
      !std::isfinite(exponent)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, exponent] {
    return transform_float_matrix(
        impl_->value, [exponent](const cv::Mat& source, cv::Mat& result) {
          cv::pow(source, exponent, result);
        });
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::square_root_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_float_matrix(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  cv::Mat flattened = impl_->value.reshape(1);
  double minimum = 0.0;
  cv::minMaxLoc(flattened, &minimum, nullptr);
  if (minimum < 0.0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this] {
    return transform_float_matrix(
        impl_->value, [](const cv::Mat& source, cv::Mat& result) {
          cv::sqrt(source, result);
        });
  });
}

}  // namespace terlan_opencv
