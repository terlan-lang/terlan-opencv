#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_ml_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/ml.hpp>

#include <atomic>
#include <cmath>
#include <utility>

namespace terlan_opencv {
using namespace ml_internal;

namespace {

constexpr std::int64_t kMaxTrainingSamples = 65536;
constexpr std::int64_t kMaxSvmTrainingSamples = 16384;
constexpr std::int64_t kMaxLogisticTrainingSamples = 16384;
constexpr std::int64_t kMaxDecisionTreeTrainingSamples = 16384;
constexpr std::int64_t kMaxBoostTrainingSamples = 16384;
constexpr std::int64_t kMaxSgdSvmTrainingSamples = 16384;
constexpr std::int64_t kMaxRandomForestTrainingSamples = 16384;
std::atomic<std::int64_t> live_knn_models{0};
std::atomic<std::int64_t> live_linear_svm_models{0};
std::atomic<std::int64_t> live_normal_bayes_models{0};
std::atomic<std::int64_t> live_logistic_models{0};
std::atomic<std::int64_t> live_decision_tree_models{0};
std::atomic<std::int64_t> live_boost_models{0};
std::atomic<std::int64_t> live_sgd_svm_models{0};
std::atomic<std::int64_t> live_random_forest_models{0};

}  // namespace

struct OpenCvKnnModel::Impl final
    : ClassifierState<cv::ml::KNearest> {
  using ClassifierState<cv::ml::KNearest>::ClassifierState;
};

OpenCvKnnModel::OpenCvKnnModel(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_knn_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvKnnModel::~OpenCvKnnModel() noexcept {
  live_knn_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvKnnModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvKnnModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>> OpenCvKnnModel::predict_owned(
    rust::Slice<const double> samples, std::int64_t k) const noexcept {
  failure::clear();
  const std::int64_t rows =
      impl_ == nullptr
          ? 0
          : valid_sample_rows(samples, impl_->feature_count, kMaxQuerySamples);
  if (rows == 0 || k < 1 || k > impl_->sample_count || k > 4096 ||
      impl_->model.empty() || !impl_->model->isTrained() ||
      !impl_->model->isClassifier()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this, samples, rows, k] {
    const cv::Mat input = copy_samples(samples, rows, impl_->feature_count);
    cv::Mat predictions;
    cv::Mat neighbors;
    cv::Mat distances;
    const float first = impl_->model->findNearest(
        input, static_cast<int>(k), predictions, neighbors, distances);
    if (!std::isfinite(first) ||
        !valid_float_matrix(predictions, rows, 1) ||
        !valid_float_matrix(neighbors, rows, k) ||
        !valid_float_matrix(distances, rows, k)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    for (std::int64_t row = 0; row < rows; ++row) {
      float previous_distance = -1.0F;
      for (std::int64_t column = 0; column < k; ++column) {
        const float neighbor =
            neighbors.at<float>(static_cast<int>(row), static_cast<int>(column));
        const float distance =
            distances.at<float>(static_cast<int>(row), static_cast<int>(column));
        if (!std::isfinite(neighbor) || std::trunc(neighbor) != neighbor ||
            !valid_label(static_cast<std::int64_t>(neighbor)) ||
            !std::isfinite(distance) || distance < previous_distance) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<std::int64_t>>{};
        }
        previous_distance = distance;
      }
    }
    return copy_prediction_labels(predictions, rows);
  });
}

std::unique_ptr<OpenCvKnnModel> train_knn_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels, std::int64_t feature_count,
    std::int64_t default_k) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxTrainingSamples);
  if (rows == 0 || !valid_labels(labels, rows, false) ||
      default_k < 1 || default_k > rows || default_k > 4096) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvKnnModel>(
      [samples, labels, rows, feature_count, default_k] {
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses = copy_labels<int, CV_32SC1>(labels, rows);
        cv::Ptr<cv::ml::KNearest> model = cv::ml::KNearest::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvKnnModel>{};
        }
        model->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
        model->setIsClassifier(true);
        model->setDefaultK(static_cast<int>(default_k));
        if (!model->train(training, cv::ml::ROW_SAMPLE, responses) ||
            !model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getAlgorithmType() != cv::ml::KNearest::BRUTE_FORCE ||
            model->getDefaultK() != static_cast<int>(default_k)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvKnnModel>{};
        }
        return std::make_unique<OpenCvKnnModel>(
            std::make_unique<OpenCvKnnModel::Impl>(std::move(model), rows,
                                                   feature_count));
      });
}

std::int64_t live_knn_model_count() noexcept {
  return live_knn_models.load(std::memory_order_relaxed);
}

struct OpenCvLinearSvmModel::Impl final : ClassifierState<cv::ml::SVM> {
  using ClassifierState<cv::ml::SVM>::ClassifierState;
};

OpenCvLinearSvmModel::OpenCvLinearSvmModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_linear_svm_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvLinearSvmModel::~OpenCvLinearSvmModel() noexcept {
  live_linear_svm_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvLinearSvmModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvLinearSvmModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvLinearSvmModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  return predict_stat_model(impl_.get(), samples);
}

std::unique_ptr<OpenCvLinearSvmModel> train_linear_svm_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxSvmTrainingSamples);
  if (rows == 0 || !valid_labels(labels, rows, true)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvLinearSvmModel>(
      [samples, labels, rows, feature_count] {
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses = copy_labels<int, CV_32SC1>(labels, rows);
        cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvLinearSvmModel>{};
        }
        const cv::TermCriteria criteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6);
        model->setType(cv::ml::SVM::C_SVC);
        model->setKernel(cv::ml::SVM::LINEAR);
        model->setC(1.0);
        model->setTermCriteria(criteria);
        if (!model->train(training, cv::ml::ROW_SAMPLE, responses)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvLinearSvmModel>{};
        }
        const cv::TermCriteria observed = model->getTermCriteria();
        if (!model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getType() != cv::ml::SVM::C_SVC ||
            model->getKernelType() != cv::ml::SVM::LINEAR ||
            model->getC() != 1.0 || observed.type != criteria.type ||
            observed.maxCount != criteria.maxCount ||
            observed.epsilon != criteria.epsilon) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvLinearSvmModel>{};
        }
        return std::make_unique<OpenCvLinearSvmModel>(
            std::make_unique<OpenCvLinearSvmModel::Impl>(
                std::move(model), rows, feature_count));
      });
}

std::int64_t live_linear_svm_model_count() noexcept {
  return live_linear_svm_models.load(std::memory_order_relaxed);
}

struct OpenCvNormalBayesModel::Impl final
    : ClassifierState<cv::ml::NormalBayesClassifier> {
  using ClassifierState<cv::ml::NormalBayesClassifier>::ClassifierState;
};

OpenCvNormalBayesModel::OpenCvNormalBayesModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_normal_bayes_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvNormalBayesModel::~OpenCvNormalBayesModel() noexcept {
  live_normal_bayes_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvNormalBayesModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvNormalBayesModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvNormalBayesModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  return predict_stat_model(impl_.get(), samples);
}

std::unique_ptr<OpenCvNormalBayesModel> train_normal_bayes_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxTrainingSamples);
  if (rows == 0 || !valid_labels(labels, rows, true)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvNormalBayesModel>(
      [samples, labels, rows, feature_count] {
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses = copy_labels<int, CV_32SC1>(labels, rows);
        cv::Ptr<cv::ml::NormalBayesClassifier> model =
            cv::ml::NormalBayesClassifier::create();
        if (model.empty() ||
            !model->train(training, cv::ml::ROW_SAMPLE, responses) ||
            !model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvNormalBayesModel>{};
        }
        return std::make_unique<OpenCvNormalBayesModel>(
            std::make_unique<OpenCvNormalBayesModel::Impl>(
                std::move(model), rows, feature_count));
      });
}

std::int64_t live_normal_bayes_model_count() noexcept {
  return live_normal_bayes_models.load(std::memory_order_relaxed);
}

struct OpenCvLogisticRegressionModel::Impl final
    : ClassifierState<cv::ml::LogisticRegression> {
  /// Constructs state with the original sorted categorical labels.
  Impl(cv::Ptr<cv::ml::LogisticRegression> model, std::int64_t sample_count,
       std::int64_t feature_count, std::int64_t negative_label,
       std::int64_t positive_label)
      : ClassifierState<cv::ml::LogisticRegression>(
            std::move(model), sample_count, feature_count),
        negative_label(negative_label),
        positive_label(positive_label) {}

  /// Public label represented by OpenCV class zero.
  std::int64_t negative_label;
  /// Public label represented by OpenCV class one.
  std::int64_t positive_label;
};

OpenCvLogisticRegressionModel::OpenCvLogisticRegressionModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_logistic_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvLogisticRegressionModel::~OpenCvLogisticRegressionModel() noexcept {
  live_logistic_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvLogisticRegressionModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvLogisticRegressionModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvLogisticRegressionModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  failure::clear();
  const std::int64_t rows =
      impl_ == nullptr
          ? 0
          : valid_sample_rows(samples, impl_->feature_count, kMaxQuerySamples);
  if (rows == 0 || impl_->model.empty() || !impl_->model->isTrained() ||
      !impl_->model->isClassifier()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this, samples, rows] {
    const cv::Mat input = copy_samples(samples, rows, impl_->feature_count);
    cv::Mat predictions;
    const float first = impl_->model->predict(input, predictions, 0);
    if (!std::isfinite(first) || predictions.rows != static_cast<int>(rows) ||
        predictions.cols != 1 || !predictions.isContinuous() ||
        (predictions.type() != CV_32FC1 && predictions.type() != CV_32SC1)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto result = std::make_unique<std::vector<std::int64_t>>();
    result->reserve(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) {
      const std::int64_t prediction =
          predictions.type() == CV_32FC1
              ? static_cast<std::int64_t>(
                    predictions.at<float>(static_cast<int>(row), 0))
              : static_cast<std::int64_t>(
                    predictions.at<int>(static_cast<int>(row), 0));
      if (prediction != 0 && prediction != 1) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      result->push_back(prediction == 1 ? impl_->positive_label
                                        : impl_->negative_label);
    }
    return result;
  });
}

std::unique_ptr<OpenCvLogisticRegressionModel>
train_logistic_classifier_owned(rust::Slice<const double> samples,
                                rust::Slice<const std::int64_t> labels,
                                std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows = valid_sample_rows(
      samples, feature_count, kMaxLogisticTrainingSamples);
  const auto classes = binary_label_pair(labels, rows);
  if (rows == 0 || classes.first == classes.second) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvLogisticRegressionModel>(
      [samples, labels, rows, feature_count, classes] {
        constexpr double kLearningRate = 1.0;
        constexpr int kIterations = 10001;
        const cv::TermCriteria criteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kIterations,
            kLearningRate);
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses =
            copy_binary_index_labels(labels, rows, classes);
        cv::Ptr<cv::ml::LogisticRegression> model =
            cv::ml::LogisticRegression::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvLogisticRegressionModel>{};
        }
        model->setLearningRate(kLearningRate);
        model->setIterations(kIterations);
        model->setRegularization(cv::ml::LogisticRegression::REG_L2);
        model->setTrainMethod(cv::ml::LogisticRegression::BATCH);
        model->setMiniBatchSize(1);
        model->setTermCriteria(criteria);
        if (!model->train(training, cv::ml::ROW_SAMPLE, responses)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvLogisticRegressionModel>{};
        }
        const cv::TermCriteria observed = model->getTermCriteria();
        if (!model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count + 1) ||
            model->getLearningRate() != kLearningRate ||
            model->getIterations() != kIterations ||
            model->getRegularization() != cv::ml::LogisticRegression::REG_L2 ||
            model->getTrainMethod() != cv::ml::LogisticRegression::BATCH ||
            model->getMiniBatchSize() != 1 || observed.type != criteria.type ||
            observed.maxCount != criteria.maxCount ||
            observed.epsilon != criteria.epsilon) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvLogisticRegressionModel>{};
        }
        return std::make_unique<OpenCvLogisticRegressionModel>(
            std::make_unique<OpenCvLogisticRegressionModel::Impl>(
                std::move(model), rows, feature_count, classes.first,
                classes.second));
      });
}

std::int64_t live_logistic_model_count() noexcept {
  return live_logistic_models.load(std::memory_order_relaxed);
}

struct OpenCvDecisionTreeModel::Impl final
    : ClassifierState<cv::ml::DTrees> {
  using ClassifierState<cv::ml::DTrees>::ClassifierState;
};

OpenCvDecisionTreeModel::OpenCvDecisionTreeModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_decision_tree_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvDecisionTreeModel::~OpenCvDecisionTreeModel() noexcept {
  live_decision_tree_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvDecisionTreeModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvDecisionTreeModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvDecisionTreeModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  return predict_stat_model(impl_.get(), samples);
}

std::unique_ptr<OpenCvDecisionTreeModel>
train_decision_tree_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows = valid_sample_rows(
      samples, feature_count, kMaxDecisionTreeTrainingSamples);
  if (rows == 0 || !valid_labels(labels, rows, true)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvDecisionTreeModel>(
      [samples, labels, rows, feature_count] {
        constexpr int kMaxCategories = 16;
        constexpr int kMaxDepth = 8;
        constexpr int kMinSampleCount = 2;
        constexpr float kRegressionAccuracy = 0.01F;
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses = copy_labels<int, CV_32SC1>(labels, rows);
        cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDecisionTreeModel>{};
        }
        model->setMaxCategories(kMaxCategories);
        model->setMaxDepth(kMaxDepth);
        model->setMinSampleCount(kMinSampleCount);
        model->setCVFolds(0);
        model->setUseSurrogates(false);
        model->setUse1SERule(false);
        model->setTruncatePrunedTree(true);
        model->setRegressionAccuracy(kRegressionAccuracy);
        model->setPriors(cv::Mat());
        if (!model->train(training, cv::ml::ROW_SAMPLE, responses) ||
            !model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getMaxDepth() != kMaxDepth ||
            model->getMinSampleCount() != kMinSampleCount ||
            model->getCVFolds() != 0 || model->getUseSurrogates() ||
            model->getUse1SERule() || !model->getTruncatePrunedTree() ||
            model->getRegressionAccuracy() != kRegressionAccuracy ||
            !model->getPriors().empty() || model->getRoots().size() != 1 ||
            model->getNodes().empty() || model->getSplits().empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvDecisionTreeModel>{};
        }
        return std::make_unique<OpenCvDecisionTreeModel>(
            std::make_unique<OpenCvDecisionTreeModel::Impl>(
                std::move(model), rows, feature_count));
      });
}

std::int64_t live_decision_tree_model_count() noexcept {
  return live_decision_tree_models.load(std::memory_order_relaxed);
}

struct OpenCvBoostModel::Impl final : ClassifierState<cv::ml::Boost> {
  using ClassifierState<cv::ml::Boost>::ClassifierState;
};

OpenCvBoostModel::OpenCvBoostModel(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_boost_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvBoostModel::~OpenCvBoostModel() noexcept {
  live_boost_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvBoostModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvBoostModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>> OpenCvBoostModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  return predict_stat_model(impl_.get(), samples);
}

std::unique_ptr<OpenCvBoostModel> train_boost_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxBoostTrainingSamples);
  const auto classes = binary_label_pair(labels, rows);
  if (rows == 0 || classes.first == classes.second) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvBoostModel>(
      [samples, labels, rows, feature_count] {
        constexpr int kWeakCount = 32;
        constexpr double kWeightTrimRate = 0.95;
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses = copy_labels<int, CV_32SC1>(labels, rows);
        cv::Ptr<cv::ml::Boost> model = cv::ml::Boost::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvBoostModel>{};
        }
        model->setBoostType(cv::ml::Boost::DISCRETE);
        model->setWeakCount(kWeakCount);
        model->setWeightTrimRate(kWeightTrimRate);
        model->setMaxCategories(16);
        model->setMaxDepth(1);
        model->setMinSampleCount(2);
        model->setCVFolds(0);
        model->setUseSurrogates(false);
        model->setUse1SERule(false);
        model->setTruncatePrunedTree(true);
        model->setRegressionAccuracy(0.01F);
        model->setPriors(cv::Mat());
        const bool trained =
            model->train(training, cv::ml::ROW_SAMPLE, responses);
        if (!trained ||
            !model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getBoostType() != cv::ml::Boost::DISCRETE ||
            model->getWeakCount() != kWeakCount ||
            model->getWeightTrimRate() != kWeightTrimRate ||
            model->getMaxDepth() != 1 || model->getMinSampleCount() != 2 ||
            model->getCVFolds() != 0 ||
            model->getUseSurrogates() || model->getUse1SERule() ||
            !model->getTruncatePrunedTree() ||
            model->getRegressionAccuracy() != 0.01F ||
            !model->getPriors().empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvBoostModel>{};
        }
        return std::make_unique<OpenCvBoostModel>(
            std::make_unique<OpenCvBoostModel::Impl>(
                std::move(model), rows, feature_count));
      });
}

std::int64_t live_boost_model_count() noexcept {
  return live_boost_models.load(std::memory_order_relaxed);
}

struct OpenCvSgdSvmModel::Impl final : ClassifierState<cv::ml::SVMSGD> {
  /// Constructs state with the original sorted categorical labels.
  Impl(cv::Ptr<cv::ml::SVMSGD> model, std::int64_t sample_count,
       std::int64_t feature_count, std::int64_t negative_label,
       std::int64_t positive_label)
      : ClassifierState<cv::ml::SVMSGD>(std::move(model), sample_count,
                                        feature_count),
        negative_label(negative_label),
        positive_label(positive_label) {}

  std::int64_t negative_label;
  std::int64_t positive_label;
};

OpenCvSgdSvmModel::OpenCvSgdSvmModel(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_sgd_svm_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvSgdSvmModel::~OpenCvSgdSvmModel() noexcept {
  live_sgd_svm_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvSgdSvmModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvSgdSvmModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvSgdSvmModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  failure::clear();
  const std::int64_t rows =
      impl_ == nullptr
          ? 0
          : valid_sample_rows(samples, impl_->feature_count, kMaxQuerySamples);
  if (rows == 0 || impl_->model.empty() || !impl_->model->isTrained() ||
      !impl_->model->isClassifier()) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this, samples, rows] {
    const cv::Mat input = copy_samples(samples, rows, impl_->feature_count);
    cv::Mat predictions;
    const float first = impl_->model->predict(input, predictions, 0);
    if (!std::isfinite(first) || !valid_float_matrix(predictions, rows, 1)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto result = std::make_unique<std::vector<std::int64_t>>();
    result->reserve(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) {
      const float prediction =
          predictions.at<float>(static_cast<int>(row), 0);
      if (prediction != -1.0F && prediction != 1.0F) {
        failure::set(failure::NativeFailure::NativeFailure);
        return std::unique_ptr<std::vector<std::int64_t>>{};
      }
      result->push_back(prediction > 0.0F ? impl_->positive_label
                                          : impl_->negative_label);
    }
    return result;
  });
}

std::unique_ptr<OpenCvSgdSvmModel> train_sgd_svm_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxSgdSvmTrainingSamples);
  const auto classes = binary_label_pair(labels, rows);
  if (rows == 0 || classes.first == classes.second) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvSgdSvmModel>(
      [samples, labels, rows, feature_count, classes] {
        const cv::TermCriteria criteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100000,
            0.00001);
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses =
            copy_binary_sign_labels(labels, rows, classes);
        cv::Ptr<cv::ml::SVMSGD> model = cv::ml::SVMSGD::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSgdSvmModel>{};
        }
        model->setOptimalParameters(cv::ml::SVMSGD::ASGD,
                                    cv::ml::SVMSGD::SOFT_MARGIN);
        model->setTermCriteria(criteria);
        if (!model->train(training, cv::ml::ROW_SAMPLE, responses) ||
            !model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getSvmsgdType() != cv::ml::SVMSGD::ASGD ||
            model->getMarginType() != cv::ml::SVMSGD::SOFT_MARGIN ||
            model->getMarginRegularization() != 0.00001F ||
            model->getInitialStepSize() != 0.05F ||
            model->getStepDecreasingPower() != 0.75F) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSgdSvmModel>{};
        }
        const cv::TermCriteria observed = model->getTermCriteria();
        const cv::Mat weights = model->getWeights();
        const float shift = model->getShift();
        if (observed.type != criteria.type ||
            observed.maxCount != criteria.maxCount ||
            observed.epsilon != criteria.epsilon ||
            !valid_float_matrix(weights, 1, feature_count) ||
            !std::isfinite(shift)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvSgdSvmModel>{};
        }
        for (std::int64_t column = 0; column < feature_count; ++column) {
          if (!std::isfinite(weights.at<float>(0, static_cast<int>(column)))) {
            failure::set(failure::NativeFailure::NativeFailure);
            return std::unique_ptr<OpenCvSgdSvmModel>{};
          }
        }
        return std::make_unique<OpenCvSgdSvmModel>(
            std::make_unique<OpenCvSgdSvmModel::Impl>(
                std::move(model), rows, feature_count, classes.first,
                classes.second));
      });
}

std::int64_t live_sgd_svm_model_count() noexcept {
  return live_sgd_svm_models.load(std::memory_order_relaxed);
}

struct OpenCvRandomForestModel::Impl final
    : ClassifierState<cv::ml::RTrees> {
  using ClassifierState<cv::ml::RTrees>::ClassifierState;
};

OpenCvRandomForestModel::OpenCvRandomForestModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_random_forest_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvRandomForestModel::~OpenCvRandomForestModel() noexcept {
  live_random_forest_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvRandomForestModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvRandomForestModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvRandomForestModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  return predict_stat_model(impl_.get(), samples);
}

std::unique_ptr<OpenCvRandomForestModel>
train_random_forest_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows = valid_sample_rows(
      samples, feature_count, kMaxRandomForestTrainingSamples);
  if (rows == 0 || !valid_labels(labels, rows, true)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvRandomForestModel>(
      [samples, labels, rows, feature_count] {
        constexpr int kMaxCategories = 16;
        constexpr int kMaxDepth = 8;
        constexpr int kMinSampleCount = 2;
        constexpr float kRegressionAccuracy = 0.01F;
        constexpr int kTreeCount = 32;
        constexpr std::uint64_t kTrainingSeed = 0x7465726c616eULL;
        const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER,
                                        kTreeCount, 0.0);
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat responses = copy_labels<int, CV_32SC1>(labels, rows);
        cv::Ptr<cv::ml::RTrees> model = cv::ml::RTrees::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvRandomForestModel>{};
        }
        model->setMaxCategories(kMaxCategories);
        model->setMaxDepth(kMaxDepth);
        model->setMinSampleCount(kMinSampleCount);
        model->setCVFolds(0);
        model->setUseSurrogates(false);
        model->setUse1SERule(false);
        model->setTruncatePrunedTree(false);
        model->setRegressionAccuracy(kRegressionAccuracy);
        model->setPriors(cv::Mat());
        model->setCalculateVarImportance(false);
        model->setActiveVarCount(0);
        model->setTermCriteria(criteria);
        bool trained = false;
        {
          const ScopedRandomSeed deterministic_rng(kTrainingSeed);
          trained = model->train(training, cv::ml::ROW_SAMPLE, responses);
        }
        const cv::TermCriteria observed = model->getTermCriteria();
        if (!trained || !model->isTrained() || !model->isClassifier() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getMaxDepth() != kMaxDepth ||
            model->getMinSampleCount() != kMinSampleCount ||
            model->getCVFolds() != 0 || model->getUseSurrogates() ||
            model->getUse1SERule() || model->getTruncatePrunedTree() ||
            model->getRegressionAccuracy() != kRegressionAccuracy ||
            !model->getPriors().empty() ||
            model->getCalculateVarImportance() ||
            model->getActiveVarCount() != 0 ||
            observed.type != criteria.type ||
            observed.maxCount != criteria.maxCount ||
            observed.epsilon != criteria.epsilon ||
            model->getRoots().size() != static_cast<std::size_t>(kTreeCount) ||
            model->getNodes().empty() || model->getSplits().empty() ||
            !model->getVarImportance().empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvRandomForestModel>{};
        }
        return std::make_unique<OpenCvRandomForestModel>(
            std::make_unique<OpenCvRandomForestModel::Impl>(
                std::move(model), rows, feature_count));
      });
}

std::int64_t live_random_forest_model_count() noexcept {
  return live_random_forest_models.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
