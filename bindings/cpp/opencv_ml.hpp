#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#if defined(TERLAN_OPENCV_METADATA)
#include "opencv_metadata_rust.hpp"
#else
#include "rust/cxx.h"
#endif

namespace terlan_opencv {

/// Uniquely owns one trained process-local K-nearest-neighbor classifier.
class OpenCvKnnModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvKnnModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvKnnModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples, std::int64_t k) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded brute-force K-nearest-neighbor classifier.
std::unique_ptr<OpenCvKnnModel> train_knn_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels, std::int64_t feature_count,
    std::int64_t default_k) noexcept;

/// Returns the number of live K-nearest-neighbor model owners.
std::int64_t live_knn_model_count() noexcept;

/// Uniquely owns one trained process-local linear support-vector classifier.
class OpenCvLinearSvmModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvLinearSvmModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvLinearSvmModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded fixed-policy linear support-vector classifier.
std::unique_ptr<OpenCvLinearSvmModel> train_linear_svm_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live linear support-vector model owners.
std::int64_t live_linear_svm_model_count() noexcept;

/// Uniquely owns one trained process-local normal Bayes classifier.
class OpenCvNormalBayesModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvNormalBayesModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvNormalBayesModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded in-memory normal Bayes classifier.
std::unique_ptr<OpenCvNormalBayesModel> train_normal_bayes_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live normal Bayes model owners.
std::int64_t live_normal_bayes_model_count() noexcept;

/// Uniquely owns one trained process-local logistic-regression classifier.
class OpenCvLogisticRegressionModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvLogisticRegressionModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvLogisticRegressionModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded fixed-policy logistic-regression classifier.
std::unique_ptr<OpenCvLogisticRegressionModel>
train_logistic_classifier_owned(rust::Slice<const double> samples,
                                rust::Slice<const std::int64_t> labels,
                                std::int64_t feature_count) noexcept;

/// Returns the number of live logistic-regression model owners.
std::int64_t live_logistic_model_count() noexcept;

/// Uniquely owns one trained process-local fixed-policy decision tree.
class OpenCvDecisionTreeModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvDecisionTreeModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvDecisionTreeModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded fixed-policy decision-tree classifier.
std::unique_ptr<OpenCvDecisionTreeModel> train_decision_tree_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live decision-tree model owners.
std::int64_t live_decision_tree_model_count() noexcept;

/// Uniquely owns one trained process-local fixed-policy boosted classifier.
class OpenCvBoostModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvBoostModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvBoostModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded fixed-policy binary boosted classifier.
std::unique_ptr<OpenCvBoostModel> train_boost_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live boosted-classifier model owners.
std::int64_t live_boost_model_count() noexcept;

/// Uniquely owns one trained process-local fixed-policy SGD-SVM classifier.
class OpenCvSgdSvmModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvSgdSvmModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvSgdSvmModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded fixed-policy binary average-SGD SVM classifier.
std::unique_ptr<OpenCvSgdSvmModel> train_sgd_svm_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live SGD-SVM model owners.
std::int64_t live_sgd_svm_model_count() noexcept;

/// Uniquely owns one trained process-local fixed-policy random forest.
class OpenCvRandomForestModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvRandomForestModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvRandomForestModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded deterministic fixed-policy random-forest classifier.
std::unique_ptr<OpenCvRandomForestModel>
train_random_forest_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live random-forest model owners.
std::int64_t live_random_forest_model_count() noexcept;

/// Uniquely owns one trained process-local fixed-policy MLP classifier.
class OpenCvMlpClassifierModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvMlpClassifierModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvMlpClassifierModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Predicts original integer labels for bounded row-major query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded deterministic fixed-policy MLP classifier.
std::unique_ptr<OpenCvMlpClassifierModel> train_mlp_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept;

/// Returns the number of live MLP classifier owners.
std::int64_t live_mlp_model_count() noexcept;

/// Uniquely owns one trained process-local fixed-policy Gaussian mixture.
class OpenCvGaussianMixtureModel final {
 public:
  struct Impl;
  /// Constructs a model from its private OpenCV representation.
  explicit OpenCvGaussianMixtureModel(std::unique_ptr<Impl> impl) noexcept;
  /// Destroys the trained model.
  ~OpenCvGaussianMixtureModel() noexcept;
  /// Returns the copied training sample count.
  std::int64_t sample_count() const noexcept;
  /// Returns the copied feature count per sample.
  std::int64_t feature_count() const noexcept;
  /// Returns the immutable number of model-local clusters.
  std::int64_t cluster_count() const noexcept;
  /// Predicts model-local cluster indexes for bounded query samples.
  std::unique_ptr<std::vector<std::int64_t>> predict_owned(
      rust::Slice<const double> samples) const noexcept;

 private:
  std::unique_ptr<Impl> impl_;
};

/// Trains one bounded deterministic fixed-policy Gaussian mixture.
std::unique_ptr<OpenCvGaussianMixtureModel> train_gaussian_mixture_owned(
    rust::Slice<const double> samples, std::int64_t feature_count,
    std::int64_t cluster_count) noexcept;

/// Returns the number of live Gaussian-mixture model owners.
std::int64_t live_gaussian_mixture_model_count() noexcept;

}  // namespace terlan_opencv
