#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_ml_internal.hpp"

#include <opencv2/ml.hpp>

#include <atomic>
#include <cmath>
#include <utility>

namespace terlan_opencv {
using namespace ml_internal;

namespace {

constexpr std::int64_t kMaxEmTrainingSamples = 16384;
constexpr std::int64_t kMaxEmClusters = 64;
std::atomic<std::int64_t> live_gaussian_mixture_models{0};

/// Reports whether one matrix is a finite continuous double matrix.
bool valid_double_matrix(const cv::Mat& matrix, std::int64_t rows,
                         std::int64_t columns) noexcept {
  if (matrix.type() != CV_64FC1 || matrix.rows != static_cast<int>(rows) ||
      matrix.cols != static_cast<int>(columns) || !matrix.isContinuous()) {
    return false;
  }
  const double* values = matrix.ptr<double>();
  for (std::size_t index = 0; index < matrix.total(); ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

/// Reports whether every row is a finite normalized probability distribution.
bool valid_probability_matrix(const cv::Mat& matrix, std::int64_t rows,
                              std::int64_t clusters) noexcept {
  if (!valid_double_matrix(matrix, rows, clusters)) {
    return false;
  }
  for (std::int64_t row = 0; row < rows; ++row) {
    const double* probabilities = matrix.ptr<double>(static_cast<int>(row));
    double total = 0.0;
    for (std::int64_t cluster = 0; cluster < clusters; ++cluster) {
      if (probabilities[cluster] < 0.0 || probabilities[cluster] > 1.0) {
        return false;
      }
      total += probabilities[cluster];
    }
    if (std::abs(total - 1.0) > 1e-6) {
      return false;
    }
  }
  return true;
}

/// Reports whether copied EM labels identify only configured clusters.
bool valid_cluster_labels(const cv::Mat& labels, std::int64_t rows,
                          std::int64_t clusters) noexcept {
  if (labels.type() != CV_32SC1 || labels.rows != static_cast<int>(rows) ||
      labels.cols != 1 || !labels.isContinuous()) {
    return false;
  }
  for (std::int64_t row = 0; row < rows; ++row) {
    const int label = labels.at<int>(static_cast<int>(row), 0);
    if (label < 0 || label >= clusters) {
      return false;
    }
  }
  return true;
}

/// Reports whether all learned diagonal covariance matrices are finite.
bool valid_covariances(const std::vector<cv::Mat>& covariances,
                       std::int64_t clusters,
                       std::int64_t features) noexcept {
  if (covariances.size() != static_cast<std::size_t>(clusters)) {
    return false;
  }
  for (const cv::Mat& covariance : covariances) {
    if (!valid_double_matrix(covariance, features, features)) {
      return false;
    }
  }
  return true;
}

}  // namespace

struct OpenCvGaussianMixtureModel::Impl final
    : ClassifierState<cv::ml::EM> {
  /// Constructs state around one trained mixture and immutable cluster count.
  Impl(cv::Ptr<cv::ml::EM> model, std::int64_t sample_count,
       std::int64_t feature_count, std::int64_t cluster_count)
      : ClassifierState(std::move(model), sample_count, feature_count),
        cluster_count(cluster_count) {}

  /// Number of model-local Gaussian cluster indexes.
  std::int64_t cluster_count;
};

OpenCvGaussianMixtureModel::OpenCvGaussianMixtureModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_gaussian_mixture_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvGaussianMixtureModel::~OpenCvGaussianMixtureModel() noexcept {
  live_gaussian_mixture_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvGaussianMixtureModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvGaussianMixtureModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::int64_t OpenCvGaussianMixtureModel::cluster_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->cluster_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvGaussianMixtureModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  failure::clear();
  const std::int64_t rows =
      impl_ == nullptr
          ? 0
          : valid_sample_rows(samples, impl_->feature_count, kMaxQuerySamples);
  if (rows == 0 || impl_->model.empty() || !impl_->model->isTrained() ||
      impl_->cluster_count < 2) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this, samples, rows] {
    const cv::Mat input = copy_samples(samples, rows, impl_->feature_count);
    auto result = std::make_unique<std::vector<std::int64_t>>();
    result->reserve(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) {
      cv::Mat probabilities;
      const cv::Vec2d prediction = impl_->model->predict2(
          input.row(static_cast<int>(row)), probabilities);
      if (!std::isfinite(prediction[0]) || !std::isfinite(prediction[1]) ||
          std::trunc(prediction[1]) != prediction[1] || prediction[1] < 0.0 ||
          prediction[1] >= static_cast<double>(impl_->cluster_count) ||
          !valid_probability_matrix(probabilities, 1,
                                    impl_->cluster_count)) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      result->push_back(static_cast<std::int64_t>(prediction[1]));
    }
    return result;
  });
}

std::unique_ptr<OpenCvGaussianMixtureModel>
train_gaussian_mixture_owned(rust::Slice<const double> samples,
                             std::int64_t feature_count,
                             std::int64_t cluster_count) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxEmTrainingSamples);
  if (rows == 0 || cluster_count < 2 || cluster_count > kMaxEmClusters ||
      rows < cluster_count * 2) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvGaussianMixtureModel>(
      [samples, rows, feature_count, cluster_count] {
        constexpr std::uint64_t kTrainingSeed = 0x7465726c616eULL;
        const cv::TermCriteria criteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-6);
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        cv::Ptr<cv::ml::EM> model = cv::ml::EM::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvGaussianMixtureModel>{};
        }
        model->setClustersNumber(static_cast<int>(cluster_count));
        model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
        model->setTermCriteria(criteria);
        cv::Mat likelihoods;
        cv::Mat labels;
        cv::Mat probabilities;
        bool trained = false;
        {
          const ScopedRandomSeed deterministic_rng(kTrainingSeed);
          trained = model->trainEM(training, likelihoods, labels,
                                   probabilities);
        }
        const cv::TermCriteria observed = model->getTermCriteria();
        std::vector<cv::Mat> covariances;
        model->getCovs(covariances);
        if (!trained || !model->isTrained() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getClustersNumber() != static_cast<int>(cluster_count) ||
            model->getCovarianceMatrixType() != cv::ml::EM::COV_MAT_DIAGONAL ||
            observed.type != criteria.type ||
            observed.maxCount != criteria.maxCount ||
            observed.epsilon != criteria.epsilon ||
            !valid_double_matrix(likelihoods, rows, 1) ||
            !valid_cluster_labels(labels, rows, cluster_count) ||
            !valid_probability_matrix(probabilities, rows, cluster_count) ||
            !valid_double_matrix(model->getMeans(), cluster_count,
                                 feature_count) ||
            !valid_probability_matrix(model->getWeights(), 1, cluster_count) ||
            !valid_covariances(covariances, cluster_count, feature_count)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvGaussianMixtureModel>{};
        }
        return std::make_unique<OpenCvGaussianMixtureModel>(
            std::make_unique<OpenCvGaussianMixtureModel::Impl>(
                std::move(model), rows, feature_count, cluster_count));
      });
}

std::int64_t live_gaussian_mixture_model_count() noexcept {
  return live_gaussian_mixture_models.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
