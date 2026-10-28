#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace terlan_opencv {
namespace {

/// Maximum accepted or produced portable matrix document size.
constexpr std::size_t kMaxMatrixDocumentBytes = 64U * 1024U * 1024U;

/// Converts one package decomposition policy to its OpenCV value.
int decomposition_code(std::int64_t method) noexcept {
  switch (method) {
    case static_cast<std::int64_t>(DecompositionMethod::Lu):
      return cv::DECOMP_LU;
    case static_cast<std::int64_t>(DecompositionMethod::Svd):
      return cv::DECOMP_SVD;
    case static_cast<std::int64_t>(DecompositionMethod::Eigen):
      return cv::DECOMP_EIG;
    case static_cast<std::int64_t>(DecompositionMethod::Cholesky):
      return cv::DECOMP_CHOLESKY;
    case static_cast<std::int64_t>(DecompositionMethod::Qr):
      return cv::DECOMP_QR;
    case static_cast<std::int64_t>(DecompositionMethod::NormalLu):
      return cv::DECOMP_NORMAL | cv::DECOMP_LU;
    case static_cast<std::int64_t>(DecompositionMethod::NormalSvd):
      return cv::DECOMP_NORMAL | cv::DECOMP_SVD;
    default:
      return -1;
  }
}

/// Converts one package norm policy to its OpenCV value.
int norm_code(std::int64_t norm) noexcept {
  switch (norm) {
    case static_cast<std::int64_t>(MatrixNorm::L1):
      return cv::NORM_L1;
    case static_cast<std::int64_t>(MatrixNorm::L2):
      return cv::NORM_L2;
    case static_cast<std::int64_t>(MatrixNorm::Infinity):
      return cv::NORM_INF;
    default:
      return -1;
  }
}

/// Converts one package serialization format to OpenCV FileStorage flags.
int serialization_code(std::int64_t format) noexcept {
  switch (format) {
    case static_cast<std::int64_t>(SerializationFormat::Yaml):
      return cv::FileStorage::FORMAT_YAML;
    case static_cast<std::int64_t>(SerializationFormat::Xml):
      return cv::FileStorage::FORMAT_XML;
    case static_cast<std::int64_t>(SerializationFormat::Json):
      return cv::FileStorage::FORMAT_JSON;
    default:
      return -1;
  }
}

/// Reports whether one matrix is a supported dense linear-algebra operand.
bool valid_linear_matrix(const cv::Mat& matrix) noexcept {
  return valid_owned_matrix(matrix) && matrix.channels() == 1 &&
         (matrix.depth() == CV_32F || matrix.depth() == CV_64F);
}

/// Reports whether one matrix can be inverted or have a determinant.
bool valid_square_linear_matrix(const cv::Mat& matrix) noexcept {
  return valid_linear_matrix(matrix) && matrix.rows == matrix.cols;
}

/// Reports whether a decomposition requires a square coefficient matrix.
bool requires_square_coefficients(std::int64_t method) noexcept {
  return method != static_cast<std::int64_t>(DecompositionMethod::Svd) &&
         method != static_cast<std::int64_t>(DecompositionMethod::NormalLu) &&
         method != static_cast<std::int64_t>(DecompositionMethod::NormalSvd);
}

/// Wraps one validated matrix in package ownership.
std::unique_ptr<OpenCvMat> own_linear_matrix(cv::Mat matrix) {
  if (!valid_linear_matrix(matrix)) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<OpenCvMat>(
      std::make_unique<OpenCvMat::Impl>(std::move(matrix)));
}

/// Creates a typed matrix after validating public storage policy.
template <typename Factory>
std::unique_ptr<OpenCvMat> create_typed_matrix(
    std::int64_t rows, std::int64_t cols, std::int64_t channels,
    std::int64_t depth, Factory factory) noexcept {
  failure::clear();
  const auto scalar_size =
      depth == 0 || depth == 1   ? 1
      : depth == 2 || depth == 3 || depth == 7 ? 2
      : depth == 4 || depth == 5 ? 4
      : depth == 6               ? 8
                                 : 0;
  const auto shape =
      validation::matrix_shape(rows, cols, channels, scalar_size);
  if (!shape.valid || depth < 0 || depth > 7) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [rows, cols, channels, depth, factory] {
        cv::Mat result = factory(
            static_cast<int>(rows), static_cast<int>(cols),
            CV_MAKETYPE(static_cast<int>(depth), static_cast<int>(channels)));
        if (!valid_owned_matrix(result)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMat>{};
        }
        return std::make_unique<OpenCvMat>(
            std::make_unique<OpenCvMat::Impl>(std::move(result)));
      });
}

}  // namespace

std::unique_ptr<OpenCvMat> mat_ones_typed(
    std::int64_t rows, std::int64_t cols, std::int64_t channels,
    std::int64_t depth) noexcept {
  return create_typed_matrix(
      rows, cols, channels, depth,
      [](int checked_rows, int checked_cols, int type) {
        return cv::Mat(checked_rows, checked_cols, type, cv::Scalar::all(1));
      });
}

std::unique_ptr<OpenCvMat> mat_identity_typed(
    std::int64_t rows, std::int64_t cols, std::int64_t depth) noexcept {
  return create_typed_matrix(
      rows, cols, 1, depth, [](int checked_rows, int checked_cols, int type) {
        return cv::Mat::eye(checked_rows, checked_cols, type);
      });
}

std::unique_ptr<OpenCvMat> mat_random_uniform(
    std::int64_t rows, std::int64_t cols, std::int64_t depth, double lower,
    double upper, std::int64_t seed) noexcept {
  failure::clear();
  const bool valid_depth = depth == static_cast<std::int64_t>(MatrixDepth::F32) ||
                           depth == static_cast<std::int64_t>(MatrixDepth::F64);
  const auto shape = validation::matrix_shape(
      rows, cols, 1, depth == static_cast<std::int64_t>(MatrixDepth::F32) ? 4 : 8);
  if (!valid_depth || !shape.valid || !std::isfinite(lower) ||
      !std::isfinite(upper) || lower >= upper || seed < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([rows, cols, depth, lower, upper, seed] {
    cv::Mat result(static_cast<int>(rows), static_cast<int>(cols),
                   static_cast<int>(depth));
    cv::RNG random(static_cast<std::uint64_t>(seed));
    random.fill(result, cv::RNG::UNIFORM, lower, upper);
    return own_linear_matrix(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> mat_deserialize(
    rust::Str document, std::int64_t format) noexcept {
  failure::clear();
  const int checked_format = serialization_code(format);
  if (document.empty() || document.size() > kMaxMatrixDocumentBytes ||
      checked_format < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([document, checked_format] {
    try {
      const std::string copied(document.data(), document.size());
      cv::FileStorage storage(
          copied, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                      checked_format);
      if (!storage.isOpened()) {
        failure::set(failure::NativeFailure::InvalidArgument);
        return std::unique_ptr<OpenCvMat>{};
      }
      cv::Mat result;
      storage["matrix"] >> result;
      if (!valid_owned_matrix(result)) {
        failure::set(failure::NativeFailure::InvalidArgument);
        return std::unique_ptr<OpenCvMat>{};
      }
      return std::make_unique<OpenCvMat>(
          std::make_unique<OpenCvMat::Impl>(std::move(result)));
    } catch (const cv::Exception&) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
  });
}

double OpenCvMat::dot_value(const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !compatible_matrices(impl_->value, other.impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return 0.0;
  }
  return failure::contain_value<double>(
      [this, &other] {
        const double result = impl_->value.dot(other.impl_->value);
        if (!std::isfinite(result)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return 0.0;
        }
        return result;
      },
      0.0);
}

std::unique_ptr<OpenCvMat> OpenCvMat::matrix_multiply_owned(
    const OpenCvMat& other) const noexcept {
  failure::clear();
  if (impl_ == nullptr || other.impl_ == nullptr ||
      !valid_linear_matrix(impl_->value) ||
      !valid_linear_matrix(other.impl_->value) ||
      impl_->value.depth() != other.impl_->value.depth() ||
      impl_->value.cols != other.impl_->value.rows) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, &other] {
    cv::Mat result;
    cv::gemm(impl_->value, other.impl_->value, 1.0, cv::noArray(), 0.0,
             result);
    return own_linear_matrix(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::matrix_inverse_owned(
    std::int64_t method) const noexcept {
  failure::clear();
  const int checked_method = decomposition_code(method);
  if (impl_ == nullptr || !valid_square_linear_matrix(impl_->value) ||
      checked_method < 0 ||
      method == static_cast<std::int64_t>(DecompositionMethod::NormalLu) ||
      method == static_cast<std::int64_t>(DecompositionMethod::NormalSvd)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>([this, checked_method, method] {
    cv::Mat result;
    const double status = cv::invert(impl_->value, result, checked_method);
    if (!std::isfinite(status) ||
        (status == 0.0 &&
         method != static_cast<std::int64_t>(DecompositionMethod::Svd))) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    return own_linear_matrix(std::move(result));
  });
}

std::unique_ptr<OpenCvMat> OpenCvMat::matrix_solve_owned(
    const OpenCvMat& right_hand_side, std::int64_t method) const noexcept {
  failure::clear();
  const int checked_method = decomposition_code(method);
  if (impl_ == nullptr || right_hand_side.impl_ == nullptr ||
      !valid_linear_matrix(impl_->value) ||
      !valid_linear_matrix(right_hand_side.impl_->value) ||
      impl_->value.depth() != right_hand_side.impl_->value.depth() ||
      impl_->value.rows != right_hand_side.impl_->value.rows ||
      checked_method < 0 ||
      (requires_square_coefficients(method) &&
       impl_->value.rows != impl_->value.cols)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, &right_hand_side, checked_method] {
        cv::Mat result;
        if (!cv::solve(impl_->value, right_hand_side.impl_->value, result,
                       checked_method)) {
          failure::set(failure::NativeFailure::InvalidArgument);
          return std::unique_ptr<OpenCvMat>{};
        }
        return own_linear_matrix(std::move(result));
      });
}

double OpenCvMat::determinant_value() const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_square_linear_matrix(impl_->value)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return 0.0;
  }
  return failure::contain_value<double>(
      [this] {
        const double result = cv::determinant(impl_->value);
        if (!std::isfinite(result)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return 0.0;
        }
        return result;
      },
      0.0);
}

double OpenCvMat::norm_value(std::int64_t norm) const noexcept {
  failure::clear();
  const int checked_norm = norm_code(norm);
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      checked_norm < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return 0.0;
  }
  return failure::contain_value<double>(
      [this, checked_norm] {
        const double result = cv::norm(impl_->value, checked_norm);
        if (!std::isfinite(result)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return 0.0;
        }
        return result;
      },
      0.0);
}

std::unique_ptr<std::vector<double>> OpenCvMat::trace_values_owned()
    const noexcept {
  failure::clear();
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      impl_->value.channels() > 4) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>([this] {
    const cv::Scalar trace = cv::trace(impl_->value);
    auto result = std::make_unique<std::vector<double>>();
    result->reserve(static_cast<std::size_t>(impl_->value.channels()));
    for (int channel = 0; channel < impl_->value.channels(); ++channel) {
      if (!std::isfinite(trace[channel])) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<double>>{};
      }
      result->push_back(trace[channel]);
    }
    return result;
  });
}

std::unique_ptr<std::string> OpenCvMat::serialize_owned(
    std::int64_t format) const noexcept {
  failure::clear();
  const int checked_format = serialization_code(format);
  if (impl_ == nullptr || !valid_owned_matrix(impl_->value) ||
      checked_format < 0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::string>([this, checked_format] {
    cv::FileStorage storage(
        "", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                checked_format);
    storage << "matrix" << impl_->value;
    std::string document = storage.releaseAndGetString();
    if (document.empty() || document.size() > kMaxMatrixDocumentBytes) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::string>{};
    }
    return std::make_unique<std::string>(std::move(document));
  });
}

}  // namespace terlan_opencv
