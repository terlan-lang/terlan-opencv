#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_ml_internal.hpp"

#include <opencv2/ml.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <utility>

namespace terlan_opencv {
using namespace ml_internal;

namespace {

constexpr std::int64_t kMaxMlpTrainingSamples = 16384;
constexpr std::int64_t kMaxMlpClasses = 64;
std::atomic<std::int64_t> live_mlp_models{0};

/// Returns the fixed bounded hidden-layer width for one classifier shape.
int hidden_layer_width(std::int64_t feature_count,
                       std::int64_t class_count) noexcept {
  return static_cast<int>(
      std::min<std::int64_t>(64, std::max<std::int64_t>(4, feature_count +
                                                               class_count)));
}

/// Copies categorical labels into symmetric one-hot ANN target rows.
cv::Mat copy_mlp_targets(rust::Slice<const std::int64_t> labels,
                         std::int64_t rows,
                         const std::vector<std::int64_t>& classes) {
  cv::Mat targets(static_cast<int>(rows), static_cast<int>(classes.size()),
                  CV_32FC1, cv::Scalar(-1.0F));
  for (std::int64_t row = 0; row < rows; ++row) {
    const auto found =
        std::lower_bound(classes.begin(), classes.end(), labels[row]);
    targets.at<float>(static_cast<int>(row),
                      static_cast<int>(found - classes.begin())) = 1.0F;
  }
  return targets;
}

/// Reports whether OpenCV retained the exact fixed network topology.
bool valid_layer_sizes(const cv::Mat& layers, std::int64_t feature_count,
                       int hidden_count, std::int64_t class_count) noexcept {
  if (layers.type() != CV_32SC1 || !layers.isContinuous() ||
      layers.total() != 3) {
    return false;
  }
  const int* values = layers.ptr<int>();
  return values[0] == static_cast<int>(feature_count) &&
         values[1] == hidden_count &&
         values[2] == static_cast<int>(class_count);
}

}  // namespace

struct OpenCvMlpClassifierModel::Impl final
    : ClassifierState<cv::ml::ANN_MLP> {
  /// Constructs state around one trained network and copied class ordering.
  Impl(cv::Ptr<cv::ml::ANN_MLP> model, std::int64_t sample_count,
       std::int64_t feature_count, std::vector<std::int64_t> classes)
      : ClassifierState(std::move(model), sample_count, feature_count),
        classes(std::move(classes)) {}

  /// Sorted original class labels corresponding to network output columns.
  std::vector<std::int64_t> classes;
};

OpenCvMlpClassifierModel::OpenCvMlpClassifierModel(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_mlp_models.fetch_add(1, std::memory_order_relaxed);
}

OpenCvMlpClassifierModel::~OpenCvMlpClassifierModel() noexcept {
  live_mlp_models.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvMlpClassifierModel::sample_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->sample_count;
}

std::int64_t OpenCvMlpClassifierModel::feature_count() const noexcept {
  return impl_ == nullptr ? 0 : impl_->feature_count;
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMlpClassifierModel::predict_owned(
    rust::Slice<const double> samples) const noexcept {
  failure::clear();
  const std::int64_t rows =
      impl_ == nullptr
          ? 0
          : valid_sample_rows(samples, impl_->feature_count, kMaxQuerySamples);
  if (rows == 0 || impl_->model.empty() || !impl_->model->isTrained() ||
      impl_->classes.size() < 2) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>([this, samples, rows] {
    const cv::Mat input = copy_samples(samples, rows, impl_->feature_count);
    cv::Mat outputs;
    const float first = impl_->model->predict(input, outputs, 0);
    if (!std::isfinite(first) ||
        !valid_float_matrix(outputs, rows,
                            static_cast<std::int64_t>(impl_->classes.size()))) {
      failure::set(failure::NativeFailure::NativeFailure);
      return std::unique_ptr<std::vector<std::int64_t>>{};
    }
    auto result = std::make_unique<std::vector<std::int64_t>>();
    result->reserve(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) {
      const float* values = outputs.ptr<float>(static_cast<int>(row));
      std::size_t best = 0;
      for (std::size_t column = 0; column < impl_->classes.size(); ++column) {
        if (!std::isfinite(values[column])) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<std::vector<std::int64_t>>{};
        }
        if (values[column] > values[best]) {
          best = column;
        }
      }
      result->push_back(impl_->classes[best]);
    }
    return result;
  });
}

std::unique_ptr<OpenCvMlpClassifierModel> train_mlp_classifier_owned(
    rust::Slice<const double> samples,
    rust::Slice<const std::int64_t> labels,
    std::int64_t feature_count) noexcept {
  failure::clear();
  const std::int64_t rows =
      valid_sample_rows(samples, feature_count, kMaxMlpTrainingSamples);
  std::vector<std::int64_t> classes =
      sorted_unique_labels(labels, rows, kMaxMlpClasses);
  if (rows == 0 || classes.size() < 2) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMlpClassifierModel>(
      [samples, labels, rows, feature_count, classes = std::move(classes)]() mutable {
        constexpr double kBackpropWeightScale = 0.1;
        constexpr double kBackpropMomentumScale = 0.1;
        constexpr std::uint64_t kTrainingSeed = 0x7465726c616eULL;
        const int hidden_count =
            hidden_layer_width(feature_count, classes.size());
        const cv::TermCriteria criteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6);
        const cv::Mat training = copy_samples(samples, rows, feature_count);
        const cv::Mat targets = copy_mlp_targets(labels, rows, classes);
        const cv::Mat layers =
            (cv::Mat_<int>(1, 3) << static_cast<int>(feature_count),
             hidden_count, static_cast<int>(classes.size()));
        cv::Ptr<cv::ml::ANN_MLP> model = cv::ml::ANN_MLP::create();
        if (model.empty()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMlpClassifierModel>{};
        }
        model->setLayerSizes(layers);
        model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
        model->setTrainMethod(cv::ml::ANN_MLP::BACKPROP,
                              kBackpropWeightScale,
                              kBackpropMomentumScale);
        model->setTermCriteria(criteria);
        bool trained = false;
        {
          const ScopedRandomSeed deterministic_rng(kTrainingSeed);
          trained = model->train(training, cv::ml::ROW_SAMPLE, targets);
        }
        const cv::TermCriteria observed = model->getTermCriteria();
        if (!trained || !model->isTrained() ||
            model->getVarCount() != static_cast<int>(feature_count) ||
            model->getTrainMethod() != cv::ml::ANN_MLP::BACKPROP ||
            model->getBackpropWeightScale() != kBackpropWeightScale ||
            model->getBackpropMomentumScale() != kBackpropMomentumScale ||
            observed.type != criteria.type ||
            observed.maxCount != criteria.maxCount ||
            observed.epsilon != criteria.epsilon ||
            !valid_layer_sizes(model->getLayerSizes(), feature_count,
                               hidden_count, classes.size())) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMlpClassifierModel>{};
        }
        return std::make_unique<OpenCvMlpClassifierModel>(
            std::make_unique<OpenCvMlpClassifierModel::Impl>(
                std::move(model), rows, feature_count, std::move(classes)));
      });
}

std::int64_t live_mlp_model_count() noexcept {
  return live_mlp_models.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
