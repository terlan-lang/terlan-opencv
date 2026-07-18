#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

int main() {
  cv::Mat source(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat cloned = source.clone();
  cv::Mat resized;
  cv::resize(cloned, resized, cv::Size(1, 1), 0.0, 0.0, cv::INTER_NEAREST);
  cv::Mat gray;
  cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0.0);
  const cv::Mat blob = cv::dnn::blobFromImage(
      source, 1.0 / 255.0, cv::Size(2, 2), cv::Scalar(), true, false,
      CV_32F);
  if (blob.dims != 4 || blob.size[0] != 1 || blob.size[1] != 3 ||
      blob.size[2] != 2 || blob.size[3] != 2 || blob.type() != CV_32FC1) {
    return 1;
  }
  const cv::Mat training =
      (cv::Mat_<float>(4, 2) << 0.0F, 0.0F, 0.0F, 1.0F, 10.0F, 10.0F,
       10.0F, 11.0F);
  const cv::Mat labels = (cv::Mat_<int>(4, 1) << 0, 0, 1, 1);
  const auto knn = cv::ml::KNearest::create();
  knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
  knn->setIsClassifier(true);
  knn->setDefaultK(3);
  const bool knn_trained = knn->train(training, cv::ml::ROW_SAMPLE, labels);
  const cv::Mat queries =
      (cv::Mat_<float>(2, 2) << 0.0F, 0.25F, 10.0F, 10.25F);
  cv::Mat knn_predictions;
  cv::Mat knn_neighbors;
  cv::Mat knn_distances;
  knn->findNearest(queries, 3, knn_predictions, knn_neighbors, knn_distances);
  const auto svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::LINEAR);
  svm->setC(1.0);
  svm->setTermCriteria(cv::TermCriteria(
      cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));
  const bool svm_trained = svm->train(training, cv::ml::ROW_SAMPLE, labels);
  cv::Mat svm_predictions;
  svm->predict(queries, svm_predictions);
  const cv::Mat normal_bayes_training =
      (cv::Mat_<float>(8, 2) << 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F,
       1.0F, 1.0F, 10.0F, 10.0F, 10.0F, 11.0F, 11.0F, 10.0F, 11.0F,
       11.0F);
  const cv::Mat normal_bayes_labels =
      (cv::Mat_<int>(8, 1) << 0, 0, 0, 0, 1, 1, 1, 1);
  const auto normal_bayes = cv::ml::NormalBayesClassifier::create();
  const bool normal_bayes_trained =
      normal_bayes->train(normal_bayes_training, cv::ml::ROW_SAMPLE,
                          normal_bayes_labels);
  cv::Mat normal_bayes_predictions;
  normal_bayes->predict(queries, normal_bayes_predictions);
  const cv::Mat logistic_labels =
      (cv::Mat_<float>(8, 1) << -3.0F, -3.0F, -3.0F, -3.0F, 7.0F, 7.0F,
       7.0F, 7.0F);
  const auto logistic = cv::ml::LogisticRegression::create();
  const cv::TermCriteria logistic_criteria(
      cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 0.001);
  logistic->setLearningRate(0.001);
  logistic->setIterations(1000);
  logistic->setRegularization(cv::ml::LogisticRegression::REG_L2);
  logistic->setTrainMethod(cv::ml::LogisticRegression::BATCH);
  logistic->setMiniBatchSize(1);
  logistic->setTermCriteria(logistic_criteria);
  const bool logistic_trained = logistic->train(
      normal_bayes_training, cv::ml::ROW_SAMPLE, logistic_labels);
  cv::Mat logistic_predictions;
  logistic->predict(queries, logistic_predictions);
  const auto decision_tree = cv::ml::DTrees::create();
  decision_tree->setMaxCategories(16);
  decision_tree->setMaxDepth(8);
  decision_tree->setMinSampleCount(2);
  decision_tree->setCVFolds(0);
  decision_tree->setUseSurrogates(false);
  decision_tree->setUse1SERule(false);
  decision_tree->setTruncatePrunedTree(true);
  decision_tree->setRegressionAccuracy(0.01F);
  decision_tree->setPriors(cv::Mat());
  const bool decision_tree_trained = decision_tree->train(
      normal_bayes_training, cv::ml::ROW_SAMPLE, normal_bayes_labels);
  cv::Mat decision_tree_predictions;
  decision_tree->predict(queries, decision_tree_predictions);
  const auto boost = cv::ml::Boost::create();
  boost->setBoostType(cv::ml::Boost::REAL);
  boost->setWeakCount(32);
  boost->setWeightTrimRate(0.95);
  boost->setMaxCategories(16);
  boost->setMaxDepth(1);
  boost->setMinSampleCount(2);
  boost->setCVFolds(0);
  boost->setUseSurrogates(false);
  boost->setUse1SERule(false);
  boost->setTruncatePrunedTree(true);
  boost->setRegressionAccuracy(0.01F);
  boost->setPriors(cv::Mat());
  const bool boost_trained = boost->train(
      normal_bayes_training, cv::ml::ROW_SAMPLE, normal_bayes_labels);
  cv::Mat boost_predictions;
  boost->predict(queries, boost_predictions);
  const cv::Mat sgd_svm_labels =
      (cv::Mat_<float>(8, 1) << -1.0F, -1.0F, -1.0F, -1.0F, 1.0F, 1.0F,
       1.0F, 1.0F);
  const auto sgd_svm = cv::ml::SVMSGD::create();
  const cv::TermCriteria sgd_svm_criteria(
      cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100000, 0.00001);
  sgd_svm->setOptimalParameters(cv::ml::SVMSGD::ASGD,
                                cv::ml::SVMSGD::SOFT_MARGIN);
  sgd_svm->setTermCriteria(sgd_svm_criteria);
  const bool sgd_svm_trained = sgd_svm->train(
      normal_bayes_training, cv::ml::ROW_SAMPLE, sgd_svm_labels);
  const cv::Mat sgd_svm_weights = sgd_svm->getWeights();
  const float sgd_svm_shift = sgd_svm->getShift();
  cv::Mat sgd_svm_predictions;
  sgd_svm->predict(queries, sgd_svm_predictions);
  const auto random_forest = cv::ml::RTrees::create();
  const cv::TermCriteria random_forest_criteria(
      cv::TermCriteria::MAX_ITER, 32, 0.0);
  random_forest->setMaxCategories(16);
  random_forest->setMaxDepth(8);
  random_forest->setMinSampleCount(2);
  random_forest->setCVFolds(0);
  random_forest->setUseSurrogates(false);
  random_forest->setUse1SERule(false);
  random_forest->setTruncatePrunedTree(false);
  random_forest->setRegressionAccuracy(0.01F);
  random_forest->setPriors(cv::Mat());
  random_forest->setCalculateVarImportance(false);
  random_forest->setActiveVarCount(0);
  random_forest->setTermCriteria(random_forest_criteria);
  const std::uint64_t previous_rng_state = cv::theRNG().state;
  cv::theRNG().state = 0x7465726c616eULL;
  const bool random_forest_trained = random_forest->train(
      normal_bayes_training, cv::ml::ROW_SAMPLE, normal_bayes_labels);
  cv::theRNG().state = previous_rng_state;
  cv::Mat random_forest_predictions;
  random_forest->predict(queries, random_forest_predictions);
  const cv::Mat mlp_training =
      (cv::Mat_<float>(8, 2) << -1.0F, -1.0F, -1.0F, -0.8F, -0.8F, -1.0F,
       -0.8F, -0.8F, 0.8F, 0.8F, 0.8F, 1.0F, 1.0F, 0.8F, 1.0F, 1.0F);
  const cv::Mat mlp_targets =
      (cv::Mat_<float>(8, 2) << 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F,
       1.0F, -1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F,
       1.0F);
  const cv::Mat mlp_layers = (cv::Mat_<int>(1, 3) << 2, 4, 2);
  const cv::TermCriteria mlp_criteria(
      cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6);
  const auto mlp = cv::ml::ANN_MLP::create();
  mlp->setLayerSizes(mlp_layers);
  mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
  mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.1, 0.1);
  mlp->setTermCriteria(mlp_criteria);
  const std::uint64_t previous_mlp_rng_state = cv::theRNG().state;
  cv::theRNG().state = 0x7465726c616eULL;
  const bool mlp_trained =
      mlp->train(mlp_training, cv::ml::ROW_SAMPLE, mlp_targets);
  cv::theRNG().state = previous_mlp_rng_state;
  const cv::Mat mlp_queries =
      (cv::Mat_<float>(2, 2) << -0.9F, -0.9F, 0.9F, 0.9F);
  cv::Mat mlp_predictions;
  mlp->predict(mlp_queries, mlp_predictions);
  const cv::Mat observed_mlp_layers = mlp->getLayerSizes();
  const cv::TermCriteria mixture_criteria(
      cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-6);
  const auto mixture = cv::ml::EM::create();
  mixture->setClustersNumber(2);
  mixture->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
  mixture->setTermCriteria(mixture_criteria);
  cv::Mat mixture_likelihoods;
  cv::Mat mixture_labels;
  cv::Mat mixture_probabilities;
  const std::uint64_t previous_mixture_rng_state = cv::theRNG().state;
  cv::theRNG().state = 0x7465726c616eULL;
  const bool mixture_trained = mixture->trainEM(
      mlp_training, mixture_likelihoods, mixture_labels,
      mixture_probabilities);
  cv::theRNG().state = previous_mixture_rng_state;
  std::vector<cv::Mat> mixture_covariances;
  mixture->getCovs(mixture_covariances);
  cv::Mat first_mixture_probabilities;
  cv::Mat second_mixture_probabilities;
  const cv::Vec2d first_mixture_prediction = mixture->predict2(
      mlp_queries.row(0), first_mixture_probabilities);
  const cv::Vec2d second_mixture_prediction = mixture->predict2(
      mlp_queries.row(1), second_mixture_probabilities);
  const cv::Mat camera =
      (cv::Mat_<double>(3, 3) << 2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0,
       1.0);
  const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
  cv::Rect valid_pixels;
  const cv::Mat optimal = cv::getOptimalNewCameraMatrix(
      camera, distortion, gray.size(), 0.5, gray.size(), &valid_pixels, true);
  cv::Mat undistorted;
  cv::undistort(gray, undistorted, camera, distortion, optimal);
  const std::vector<cv::Point2d> image_points{{0.5, 0.5}};
  std::vector<cv::Point2d> corrected_points;
  cv::undistortPoints(image_points, corrected_points, camera, distortion,
                      cv::noArray(), camera);
  std::vector<cv::Point2d> transformed_points;
  cv::perspectiveTransform(image_points, transformed_points,
                           cv::Matx33d::eye());
  std::vector<cv::Point2d> affine_points;
  cv::transform(image_points, affine_points,
                cv::Matx23d(1.0, 0.0, 0.0, 0.0, 1.0, 0.0));
  const std::vector<cv::Point2f> perspective_source{
      {0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}};
  const cv::Mat estimated_perspective =
      cv::getPerspectiveTransform(perspective_source, perspective_source);
  const std::vector<cv::Point2f> affine_source{
      {0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}};
  const cv::Mat estimated_affine =
      cv::getAffineTransform(affine_source, affine_source);
  const std::vector<cv::Point3d> object_points{{0.0, 0.0, 2.0}};
  std::vector<cv::Point2d> projected_points;
  cv::projectPoints(object_points, cv::Vec3d::all(0.0), cv::Vec3d::all(0.0),
                    camera, distortion, projected_points);
  const std::vector<cv::Point3d> pose_objects{
      {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {-1.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},   {-0.5, -0.5, 1.0}, {0.5, -0.5, 1.0},
      {-0.5, 0.5, 1.0},  {0.5, 0.5, 1.0}};
  std::vector<cv::Point2d> pose_images;
  cv::projectPoints(pose_objects, cv::Vec3d(0.1, -0.2, 0.05),
                    cv::Vec3d(0.25, -0.4, 8.0), camera, distortion,
                    pose_images);
  cv::Vec3d solved_rotation;
  cv::Vec3d solved_translation;
  const bool pose_solved = cv::solvePnP(
      pose_objects, pose_images, camera, distortion, solved_rotation,
      solved_translation, false, cv::SOLVEPNP_ITERATIVE);
  cv::Vec3d robust_rotation;
  cv::Vec3d robust_translation;
  cv::Mat robust_inliers;
  const bool robust_pose_solved = cv::solvePnPRansac(
      pose_objects, pose_images, camera, distortion, robust_rotation,
      robust_translation, false, 100, 8.0F, 0.99, robust_inliers,
      cv::SOLVEPNP_ITERATIVE);
  const std::vector<cv::Point3d> minimal_objects{
      {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {-1.0, 1.0, 0.0},
      {0.0, 0.0, 1.0}};
  std::vector<cv::Point2d> minimal_images;
  cv::projectPoints(minimal_objects, cv::Vec3d(0.1, -0.2, 0.05),
                    cv::Vec3d(0.25, -0.4, 8.0), camera, distortion,
                    minimal_images);
  std::vector<cv::Mat> minimal_rotations;
  std::vector<cv::Mat> minimal_translations;
  const int minimal_pose_count = cv::solveP3P(
      minimal_objects, minimal_images, camera, distortion, minimal_rotations,
      minimal_translations, cv::SOLVEPNP_AP3P);
  const std::vector<cv::Point3d> planar_objects{
      {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}, {-1.0, 1.0, 0.0},
      {1.0, 1.0, 0.0},   {-0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}};
  std::vector<cv::Point2d> planar_images;
  cv::projectPoints(planar_objects, cv::Vec3d(0.1, -0.2, 0.05),
                    cv::Vec3d(0.25, -0.4, 8.0), camera, distortion,
                    planar_images);
  std::vector<cv::Mat> planar_rotations;
  std::vector<cv::Mat> planar_translations;
  cv::Mat planar_errors;
  const int planar_pose_count = cv::solvePnPGeneric(
      planar_objects, planar_images, camera, distortion, planar_rotations,
      planar_translations, false, cv::SOLVEPNP_IPPE, cv::noArray(),
      cv::noArray(), planar_errors);
  const std::vector<cv::Point3d> square_objects{
      {-1.0, 1.0, 0.0}, {1.0, 1.0, 0.0},
      {1.0, -1.0, 0.0}, {-1.0, -1.0, 0.0}};
  std::vector<cv::Point2d> square_images;
  cv::projectPoints(square_objects, cv::Vec3d(0.1, -0.2, 0.05),
                    cv::Vec3d(0.25, -0.4, 8.0), camera, distortion,
                    square_images);
  std::vector<cv::Mat> square_rotations;
  std::vector<cv::Mat> square_translations;
  cv::Mat square_errors;
  const int square_pose_count = cv::solvePnPGeneric(
      square_objects, square_images, camera, distortion, square_rotations,
      square_translations, false, cv::SOLVEPNP_IPPE_SQUARE, cv::noArray(),
      cv::noArray(), square_errors);
  cv::solvePnPRefineLM(
      pose_objects, pose_images, camera, distortion, solved_rotation,
      solved_translation,
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,
                       std::numeric_limits<float>::epsilon()));
  cv::solvePnPRefineVVS(
      pose_objects, pose_images, camera, distortion, solved_rotation,
      solved_translation,
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,
                       std::numeric_limits<float>::epsilon()),
      1.0);
  cv::Vec3d composed_rotation;
  cv::Vec3d composed_translation;
  cv::composeRT(solved_rotation, solved_translation, cv::Vec3d(0.0, 0.0, 0.0),
                cv::Vec3d(0.0, 0.0, 0.0), composed_rotation,
                composed_translation);
  if (!knn_trained || !knn->isTrained() || !knn->isClassifier() ||
      knn->getVarCount() != 2 || knn->getDefaultK() != 3 ||
      knn->getAlgorithmType() != cv::ml::KNearest::BRUTE_FORCE ||
      knn_predictions.rows != 2 || knn_predictions.cols != 1 ||
      knn_predictions.at<float>(0, 0) != 0.0F ||
      knn_predictions.at<float>(1, 0) != 1.0F ||
      knn_neighbors.rows != 2 || knn_neighbors.cols != 3 ||
      knn_distances.rows != 2 || knn_distances.cols != 3 ||
      !svm_trained || !svm->isTrained() || !svm->isClassifier() ||
      svm->getVarCount() != 2 || svm->getType() != cv::ml::SVM::C_SVC ||
      svm->getKernelType() != cv::ml::SVM::LINEAR || svm->getC() != 1.0 ||
      svm_predictions.rows != 2 || svm_predictions.cols != 1 ||
      svm_predictions.at<float>(0, 0) != 0.0F ||
      svm_predictions.at<float>(1, 0) != 1.0F ||
      !normal_bayes_trained || !normal_bayes->isTrained() ||
      !normal_bayes->isClassifier() || normal_bayes->getVarCount() != 2 ||
      normal_bayes_predictions.rows != 2 ||
      normal_bayes_predictions.cols != 1 ||
      normal_bayes_predictions.at<float>(0, 0) != 0.0F ||
      normal_bayes_predictions.at<float>(1, 0) != 1.0F ||
      !logistic_trained || !logistic->isTrained() ||
      !logistic->isClassifier() || logistic->getVarCount() != 3 ||
      logistic->getLearningRate() != 0.001 ||
      logistic->getIterations() != 1000 ||
      logistic->getRegularization() != cv::ml::LogisticRegression::REG_L2 ||
      logistic->getTrainMethod() != cv::ml::LogisticRegression::BATCH ||
      logistic->getMiniBatchSize() != 1 ||
      logistic->getTermCriteria().type != logistic_criteria.type ||
      logistic->getTermCriteria().maxCount != logistic_criteria.maxCount ||
      logistic->getTermCriteria().epsilon != logistic_criteria.epsilon ||
      logistic_predictions.type() != CV_32SC1 ||
      logistic_predictions.rows != 2 || logistic_predictions.cols != 1 ||
      logistic_predictions.at<int>(0, 0) != -3 ||
      logistic_predictions.at<int>(1, 0) != 7 ||
      !decision_tree_trained || !decision_tree->isTrained() ||
      !decision_tree->isClassifier() || decision_tree->getVarCount() != 2 ||
      decision_tree->getMaxCategories() != 16 ||
      decision_tree->getMaxDepth() != 8 ||
      decision_tree->getMinSampleCount() != 2 ||
      decision_tree->getCVFolds() != 0 || decision_tree->getUseSurrogates() ||
      decision_tree->getUse1SERule() ||
      !decision_tree->getTruncatePrunedTree() ||
      decision_tree->getRegressionAccuracy() != 0.01F ||
      !decision_tree->getPriors().empty() ||
      decision_tree->getRoots().size() != 1 ||
      decision_tree->getNodes().empty() ||
      decision_tree->getSplits().empty() ||
      decision_tree_predictions.type() != CV_32FC1 ||
      decision_tree_predictions.rows != 2 ||
      decision_tree_predictions.cols != 1 ||
      decision_tree_predictions.at<float>(0, 0) != 0.0F ||
      decision_tree_predictions.at<float>(1, 0) != 1.0F ||
      !boost_trained || !boost->isTrained() || !boost->isClassifier() ||
      boost->getVarCount() != 2 ||
      boost->getBoostType() != cv::ml::Boost::REAL ||
      boost->getWeakCount() != 32 || boost->getWeightTrimRate() != 0.95 ||
      boost->getMaxCategories() != 16 || boost->getMaxDepth() != 1 ||
      boost->getMinSampleCount() != 2 || boost->getCVFolds() != 0 ||
      boost->getUseSurrogates() || boost->getUse1SERule() ||
      !boost->getTruncatePrunedTree() ||
      boost->getRegressionAccuracy() != 0.01F ||
      !boost->getPriors().empty() || boost->getRoots().empty() ||
      boost->getNodes().empty() || boost->getSplits().empty() ||
      boost_predictions.type() != CV_32FC1 || boost_predictions.rows != 2 ||
      boost_predictions.cols != 1 ||
      boost_predictions.at<float>(0, 0) != 0.0F ||
      boost_predictions.at<float>(1, 0) != 1.0F ||
      !sgd_svm_trained || !sgd_svm->isTrained() ||
      !sgd_svm->isClassifier() || sgd_svm->getVarCount() != 2 ||
      sgd_svm->getSvmsgdType() != cv::ml::SVMSGD::ASGD ||
      sgd_svm->getMarginType() != cv::ml::SVMSGD::SOFT_MARGIN ||
      sgd_svm->getMarginRegularization() != 0.00001F ||
      sgd_svm->getInitialStepSize() != 0.05F ||
      sgd_svm->getStepDecreasingPower() != 0.75F ||
      sgd_svm->getTermCriteria().type != sgd_svm_criteria.type ||
      sgd_svm->getTermCriteria().maxCount != sgd_svm_criteria.maxCount ||
      sgd_svm->getTermCriteria().epsilon != sgd_svm_criteria.epsilon ||
      sgd_svm_weights.type() != CV_32FC1 || sgd_svm_weights.rows != 1 ||
      sgd_svm_weights.cols != 2 || !std::isfinite(sgd_svm_shift) ||
      sgd_svm_predictions.type() != CV_32FC1 ||
      sgd_svm_predictions.rows != 2 || sgd_svm_predictions.cols != 1 ||
      sgd_svm_predictions.at<float>(0, 0) != -1.0F ||
      sgd_svm_predictions.at<float>(1, 0) != 1.0F ||
      !random_forest_trained || !random_forest->isTrained() ||
      !random_forest->isClassifier() || random_forest->getVarCount() != 2 ||
      random_forest->getMaxCategories() != 16 ||
      random_forest->getMaxDepth() != 8 ||
      random_forest->getMinSampleCount() != 2 ||
      random_forest->getCVFolds() != 0 ||
      random_forest->getUseSurrogates() ||
      random_forest->getUse1SERule() ||
      random_forest->getTruncatePrunedTree() ||
      random_forest->getRegressionAccuracy() != 0.01F ||
      !random_forest->getPriors().empty() ||
      random_forest->getCalculateVarImportance() ||
      random_forest->getActiveVarCount() != 0 ||
      random_forest->getTermCriteria().type != random_forest_criteria.type ||
      random_forest->getTermCriteria().maxCount !=
          random_forest_criteria.maxCount ||
      random_forest->getTermCriteria().epsilon !=
          random_forest_criteria.epsilon ||
      random_forest->getRoots().size() != 32 ||
      random_forest->getNodes().empty() ||
      random_forest->getSplits().empty() ||
      !random_forest->getVarImportance().empty() ||
      random_forest_predictions.type() != CV_32FC1 ||
      random_forest_predictions.rows != 2 ||
      random_forest_predictions.cols != 1 ||
      random_forest_predictions.at<float>(0, 0) != 0.0F ||
      random_forest_predictions.at<float>(1, 0) != 1.0F ||
      !mlp_trained || !mlp->isTrained() || mlp->getVarCount() != 2 ||
      mlp->getTrainMethod() != cv::ml::ANN_MLP::BACKPROP ||
      mlp->getBackpropWeightScale() != 0.1 ||
      mlp->getBackpropMomentumScale() != 0.1 ||
      mlp->getTermCriteria().type != mlp_criteria.type ||
      mlp->getTermCriteria().maxCount != mlp_criteria.maxCount ||
      mlp->getTermCriteria().epsilon != mlp_criteria.epsilon ||
      observed_mlp_layers.type() != CV_32SC1 ||
      !observed_mlp_layers.isContinuous() || observed_mlp_layers.total() != 3 ||
      observed_mlp_layers.ptr<int>()[0] != 2 ||
      observed_mlp_layers.ptr<int>()[1] != 4 ||
      observed_mlp_layers.ptr<int>()[2] != 2 ||
      mlp_predictions.type() != CV_32FC1 || mlp_predictions.rows != 2 ||
      mlp_predictions.cols != 2 ||
      !std::isfinite(mlp_predictions.at<float>(0, 0)) ||
      !std::isfinite(mlp_predictions.at<float>(0, 1)) ||
      !std::isfinite(mlp_predictions.at<float>(1, 0)) ||
      !std::isfinite(mlp_predictions.at<float>(1, 1)) ||
      mlp_predictions.at<float>(0, 0) <= mlp_predictions.at<float>(0, 1) ||
      mlp_predictions.at<float>(1, 1) <= mlp_predictions.at<float>(1, 0) ||
      !mixture_trained || !mixture->isTrained() ||
      mixture->getVarCount() != 2 || mixture->getClustersNumber() != 2 ||
      mixture->getCovarianceMatrixType() != cv::ml::EM::COV_MAT_DIAGONAL ||
      mixture->getTermCriteria().type != mixture_criteria.type ||
      mixture->getTermCriteria().maxCount != mixture_criteria.maxCount ||
      mixture->getTermCriteria().epsilon != mixture_criteria.epsilon ||
      mixture_likelihoods.type() != CV_64FC1 ||
      mixture_likelihoods.rows != 8 || mixture_likelihoods.cols != 1 ||
      mixture_labels.type() != CV_32SC1 || mixture_labels.rows != 8 ||
      mixture_labels.cols != 1 || mixture_probabilities.type() != CV_64FC1 ||
      mixture_probabilities.rows != 8 || mixture_probabilities.cols != 2 ||
      mixture->getMeans().type() != CV_64FC1 ||
      mixture->getMeans().rows != 2 || mixture->getMeans().cols != 2 ||
      mixture->getWeights().type() != CV_64FC1 ||
      mixture->getWeights().rows != 1 || mixture->getWeights().cols != 2 ||
      mixture_covariances.size() != 2 ||
      mixture_covariances[0].type() != CV_64FC1 ||
      mixture_covariances[0].rows != 2 ||
      mixture_covariances[0].cols != 2 ||
      mixture_covariances[1].type() != CV_64FC1 ||
      mixture_covariances[1].rows != 2 ||
      mixture_covariances[1].cols != 2 ||
      !std::isfinite(first_mixture_prediction[0]) ||
      !std::isfinite(first_mixture_prediction[1]) ||
      !std::isfinite(second_mixture_prediction[0]) ||
      !std::isfinite(second_mixture_prediction[1]) ||
      first_mixture_prediction[1] == second_mixture_prediction[1] ||
      first_mixture_probabilities.type() != CV_64FC1 ||
      first_mixture_probabilities.rows != 1 ||
      first_mixture_probabilities.cols != 2 ||
      second_mixture_probabilities.type() != CV_64FC1 ||
      second_mixture_probabilities.rows != 1 ||
      second_mixture_probabilities.cols != 2 ||
      optimal.rows != 3 || optimal.cols != 3 ||
      undistorted.size() != gray.size() || undistorted.type() != gray.type()) {
    return 1;
  }
  if (corrected_points.size() != image_points.size()) {
    return 1;
  }
  if (transformed_points != image_points) {
    return 1;
  }
  if (affine_points != image_points) {
    return 1;
  }
  if (estimated_perspective.rows != 3 || estimated_perspective.cols != 3 ||
      estimated_perspective.type() != CV_64F) {
    return 1;
  }
  if (estimated_affine.rows != 2 || estimated_affine.cols != 3 ||
      estimated_affine.type() != CV_64F) {
    return 1;
  }
  if (projected_points.size() != object_points.size()) {
    return 1;
  }
  if (!pose_solved || !robust_pose_solved || robust_inliers.total() < 6 ||
      minimal_pose_count <= 0 || minimal_rotations.empty() ||
      minimal_rotations.size() != minimal_translations.size() ||
      planar_pose_count <= 0 || planar_rotations.empty() ||
      planar_rotations.size() != planar_translations.size() ||
      planar_errors.total() != planar_rotations.size() ||
      square_pose_count <= 0 || square_rotations.empty() ||
      square_rotations.size() != square_translations.size() ||
      square_errors.total() != square_rotations.size()) {
    return 1;
  }
  std::vector<cv::KeyPoint> keypoints;
  cv::FAST(gray, keypoints, 10, true, cv::FastFeatureDetector::TYPE_9_16);
  const auto agast = cv::AgastFeatureDetector::create(
      10, true, cv::AgastFeatureDetector::OAST_9_16);
  if (agast.empty() || agast->getThreshold() != 10 ||
      !agast->getNonmaxSuppression() ||
      agast->getType() != cv::AgastFeatureDetector::OAST_9_16) {
    return 1;
  }
  std::vector<cv::KeyPoint> agast_keypoints;
  agast->detect(gray, agast_keypoints);
  const auto orb = cv::ORB::create(8);
  if (orb.empty() || orb->descriptorSize() != 32) {
    return 1;
  }
  cv::Mat descriptors;
  orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors, false);
  const auto brisk = cv::BRISK::create(30, 3, 1.0F);
  if (brisk.empty() || brisk->descriptorSize() != 64 ||
      brisk->defaultNorm() != cv::NORM_HAMMING) {
    return 1;
  }
  std::vector<cv::KeyPoint> brisk_keypoints;
  cv::Mat brisk_descriptors;
  brisk->detectAndCompute(gray, cv::noArray(), brisk_keypoints,
                          brisk_descriptors, false);
  const auto akaze = cv::AKAZE::create(
      cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001F, 4, 4, cv::KAZE::DIFF_PM_G2,
      8);
  if (akaze.empty() || akaze->descriptorSize() != 61 ||
      akaze->defaultNorm() != cv::NORM_HAMMING ||
      akaze->getDescriptorType() != cv::AKAZE::DESCRIPTOR_MLDB ||
      akaze->getDescriptorSize() != 0 || akaze->getDescriptorChannels() != 3 ||
      akaze->getNOctaves() != 4 || akaze->getNOctaveLayers() != 4 ||
      akaze->getDiffusivity() != cv::KAZE::DIFF_PM_G2 ||
      akaze->getMaxPoints() != 8) {
    return 1;
  }
  std::vector<cv::KeyPoint> akaze_keypoints;
  cv::Mat akaze_descriptors;
  akaze->detectAndCompute(gray, cv::noArray(), akaze_keypoints,
                          akaze_descriptors, false);
  const auto sift = cv::SIFT::create(8, 3, 0.04, 10.0, 1.6, CV_32F, false);
  if (sift.empty() || sift->getNFeatures() != 8 ||
      sift->getNOctaveLayers() != 3 ||
      sift->getContrastThreshold() != 0.04 ||
      sift->getEdgeThreshold() != 10.0 || sift->getSigma() != 1.6 ||
      sift->descriptorSize() != 128 || sift->descriptorType() != CV_32F ||
      sift->defaultNorm() != cv::NORM_L2) {
    return 1;
  }
  std::vector<cv::KeyPoint> sift_keypoints;
  cv::Mat sift_descriptors;
  sift->detectAndCompute(gray, cv::noArray(), sift_keypoints,
                         sift_descriptors, false);
  const auto affine_sift = cv::AffineFeature::create(
      sift, 2, 0, 1.4142135623730951F, 72.0F);
  if (affine_sift.empty() || affine_sift->descriptorSize() != 128 ||
      affine_sift->descriptorType() != CV_32F ||
      affine_sift->defaultNorm() != cv::NORM_L2) {
    return 1;
  }
  std::vector<float> affine_tilts;
  std::vector<float> affine_rolls;
  affine_sift->getViewParams(affine_tilts, affine_rolls);
  if (affine_tilts.empty() || affine_tilts.size() != affine_rolls.size()) {
    return 1;
  }
  affine_sift->setViewParams(affine_tilts, affine_rolls);
  std::vector<float> configured_affine_tilts;
  std::vector<float> configured_affine_rolls;
  affine_sift->getViewParams(configured_affine_tilts,
                             configured_affine_rolls);
  if (configured_affine_tilts != affine_tilts ||
      configured_affine_rolls != affine_rolls) {
    return 1;
  }
  std::vector<cv::KeyPoint> affine_sift_keypoints;
  cv::Mat affine_sift_descriptors;
  affine_sift->detectAndCompute(gray, cv::noArray(), affine_sift_keypoints,
                                affine_sift_descriptors, false);
  const auto kaze = cv::KAZE::create(false, false, 0.001F, 4, 4,
                                     cv::KAZE::DIFF_PM_G2);
  if (kaze.empty() || kaze->getExtended() || kaze->getUpright() ||
      kaze->getThreshold() != 0.001F || kaze->getNOctaves() != 4 ||
      kaze->getNOctaveLayers() != 4 ||
      kaze->getDiffusivity() != cv::KAZE::DIFF_PM_G2 ||
      kaze->descriptorSize() != 64 || kaze->descriptorType() != CV_32F ||
      kaze->defaultNorm() != cv::NORM_L2) {
    return 1;
  }
  std::vector<cv::KeyPoint> kaze_keypoints;
  cv::Mat kaze_descriptors;
  kaze->detectAndCompute(gray, cv::noArray(), kaze_keypoints,
                         kaze_descriptors, false);
  const auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
  if (matcher.empty()) {
    return 1;
  }
  std::vector<cv::DMatch> matches;
  if (!descriptors.empty()) {
    matcher->match(descriptors, descriptors, matches);
  }
  std::vector<cv::DMatch> brisk_matches;
  if (!brisk_descriptors.empty()) {
    matcher->match(brisk_descriptors, brisk_descriptors, brisk_matches);
  }
  std::vector<cv::DMatch> akaze_matches;
  if (!akaze_descriptors.empty()) {
    matcher->match(akaze_descriptors, akaze_descriptors, akaze_matches);
  }
  const auto l2_matcher = cv::BFMatcher::create(cv::NORM_L2, true);
  if (l2_matcher.empty()) {
    return 1;
  }
  std::vector<cv::DMatch> sift_matches;
  if (!sift_descriptors.empty()) {
    l2_matcher->match(sift_descriptors, sift_descriptors, sift_matches);
  }
  std::vector<cv::DMatch> kaze_matches;
  if (!kaze_descriptors.empty()) {
    l2_matcher->match(kaze_descriptors, kaze_descriptors, kaze_matches);
  }
  const auto mser = cv::MSER::create(5, 4, 128, 0.25, 0.2, 200, 1.01,
                                     0.003, 5);
  if (mser.empty() || mser->getDelta() != 5 || mser->getMinArea() != 4 ||
      mser->getMaxArea() != 128 || mser->getMaxVariation() != 0.25 ||
      mser->getMinDiversity() != 0.2 || mser->getMaxEvolution() != 200 ||
      mser->getAreaThreshold() != 1.01 || mser->getMinMargin() != 0.003 ||
      mser->getEdgeBlurSize() != 5 || mser->getPass2Only()) {
    return 1;
  }
  cv::Mat mser_source(16, 16, CV_8UC1, cv::Scalar(0));
  cv::rectangle(mser_source, cv::Rect(4, 4, 8, 8), cv::Scalar(255),
                cv::FILLED);
  std::vector<std::vector<cv::Point>> mser_regions;
  std::vector<cv::Rect> mser_bounds;
  mser->detectRegions(mser_source, mser_regions, mser_bounds);
  if (mser_regions.size() != mser_bounds.size()) {
    return 1;
  }
  cv::SimpleBlobDetector::Params blob_params;
  blob_params.minThreshold = 0.0F;
  blob_params.maxThreshold = 256.0F;
  blob_params.thresholdStep = 10.0F;
  blob_params.minRepeatability = 2;
  blob_params.minDistBetweenBlobs = 1.0F;
  blob_params.filterByColor = true;
  blob_params.blobColor = 255;
  blob_params.filterByArea = true;
  blob_params.minArea = 4.0F;
  blob_params.maxArea = 128.0F;
  blob_params.filterByCircularity = false;
  blob_params.filterByInertia = false;
  blob_params.filterByConvexity = false;
  blob_params.collectContours = false;
  const auto blob_detector = cv::SimpleBlobDetector::create(blob_params);
  if (blob_detector.empty()) {
    return 1;
  }
  const auto configured_blobs = blob_detector->getParams();
  if (configured_blobs.minThreshold != 0.0F ||
      configured_blobs.maxThreshold != 256.0F ||
      configured_blobs.thresholdStep != 10.0F ||
      configured_blobs.minRepeatability != 2 ||
      configured_blobs.minDistBetweenBlobs != 1.0F ||
      !configured_blobs.filterByColor || configured_blobs.blobColor != 255 ||
      !configured_blobs.filterByArea || configured_blobs.minArea != 4.0F ||
      configured_blobs.maxArea != 128.0F ||
      configured_blobs.filterByCircularity || configured_blobs.filterByInertia ||
      configured_blobs.filterByConvexity || configured_blobs.collectContours) {
    return 1;
  }
  std::vector<cv::KeyPoint> blob_keypoints;
  blob_detector->detect(mser_source, blob_keypoints);
  std::vector<unsigned char> encoded;
  if (!cv::imencode(".bmp", blurred, encoded)) {
    return 1;
  }
  const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
  return decoded.empty() ? 1 : 0;
}
