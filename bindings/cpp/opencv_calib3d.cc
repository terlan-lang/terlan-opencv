#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <limits>

namespace terlan_opencv {
namespace {

constexpr double kMaximumDistortionCoefficient = 10.0;

/// Reports whether copied camera and distortion scalars are finite and bounded.
bool valid_camera_scalars(double fx, double fy, double cx, double cy,
                          double k1, double k2, double p1, double p2,
                          double k3) noexcept {
  const double coefficients[] = {k1, k2, p1, p2, k3};
  if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) ||
      !std::isfinite(cy) || fx <= 0.0 || fy <= 0.0 ||
      fx > validation::kMaxTransformCoefficient ||
      fy > validation::kMaxTransformCoefficient || cx < 0.0 || cy < 0.0 ||
      cx > validation::kMaxTransformCoefficient ||
      cy > validation::kMaxTransformCoefficient) {
    return false;
  }
  for (double coefficient : coefficients) {
    if (!std::isfinite(coefficient) ||
        std::abs(coefficient) > kMaximumDistortionCoefficient) {
      return false;
    }
  }
  return true;
}

/// Reports whether one finite camera model is bounded for this image.
bool valid_camera_model(const cv::Mat& image, double fx, double fy, double cx,
                        double cy, double k1, double k2, double p1, double p2,
                        double k3) noexcept {
  return valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3) &&
         cx < image.cols && cy < image.rows;
}

/// Reports whether one flattened point sequence is bounded and well formed.
bool valid_points(rust::Slice<const double> points) noexcept {
  return validation::valid_flattened_points(points.data(), points.size());
}

/// Reports whether one flattened 3D point sequence is bounded and well formed.
bool valid_object_points(rust::Slice<const double> points) noexcept {
  return validation::valid_flattened_object_points(points.data(), points.size());
}

/// Reports whether four 3D points contain no duplicate or collinear triple.
bool independent_minimal_object_points(
    rust::Slice<const double> points) noexcept {
  constexpr double kMinimumSquaredLength = 1.0e-18;
  for (std::size_t first = 0; first < 4; ++first) {
    for (std::size_t second = first + 1; second < 4; ++second) {
      const double dx = points[first * 3] - points[second * 3];
      const double dy = points[first * 3 + 1] - points[second * 3 + 1];
      const double dz = points[first * 3 + 2] - points[second * 3 + 2];
      if (dx * dx + dy * dy + dz * dz <= kMinimumSquaredLength) {
        return false;
      }
    }
  }
  for (std::size_t first = 0; first < 4; ++first) {
    for (std::size_t second = first + 1; second < 4; ++second) {
      for (std::size_t third = second + 1; third < 4; ++third) {
        const cv::Vec3d left(points[second * 3] - points[first * 3],
                             points[second * 3 + 1] - points[first * 3 + 1],
                             points[second * 3 + 2] - points[first * 3 + 2]);
        const cv::Vec3d right(points[third * 3] - points[first * 3],
                              points[third * 3 + 1] - points[first * 3 + 1],
                              points[third * 3 + 2] - points[first * 3 + 2]);
        const cv::Vec3d cross = left.cross(right);
        if (cross.dot(cross) <= kMinimumSquaredLength) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Reports whether bounded object points span exactly one nondegenerate plane.
bool coplanar_object_points(rust::Slice<const double> points) noexcept {
  constexpr double kMinimumSquaredLength = 1.0e-18;
  constexpr double kPlaneTolerance = 1.0e-8;
  const std::size_t count = points.size() / 3;
  cv::Vec3d anchor;
  cv::Vec3d normal;
  bool found_plane = false;
  for (std::size_t first = 0; first < count && !found_plane; ++first) {
    for (std::size_t second = first + 1; second < count && !found_plane;
         ++second) {
      for (std::size_t third = second + 1; third < count; ++third) {
        anchor = cv::Vec3d(points[first * 3], points[first * 3 + 1],
                           points[first * 3 + 2]);
        const cv::Vec3d left(
            points[second * 3] - anchor[0],
            points[second * 3 + 1] - anchor[1],
            points[second * 3 + 2] - anchor[2]);
        const cv::Vec3d right(points[third * 3] - anchor[0],
                              points[third * 3 + 1] - anchor[1],
                              points[third * 3 + 2] - anchor[2]);
        normal = left.cross(right);
        found_plane = normal.dot(normal) > kMinimumSquaredLength;
        if (found_plane) {
          break;
        }
      }
    }
  }
  if (!found_plane) {
    return false;
  }
  const double scaled_tolerance = kPlaneTolerance * std::sqrt(normal.dot(normal));
  for (std::size_t index = 0; index < count; ++index) {
    const cv::Vec3d offset(points[index * 3] - anchor[0],
                           points[index * 3 + 1] - anchor[1],
                           points[index * 3 + 2] - anchor[2]);
    if (std::abs(normal.dot(offset)) > scaled_tolerance) {
      return false;
    }
  }
  return true;
}

/// Reports whether copied image points contain one non-collinear triple.
bool spans_image_plane(rust::Slice<const double> points) noexcept {
  constexpr double kMinimumTriangleArea = 1.0e-9;
  const std::size_t count = points.size() / 2;
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = first + 1; second < count; ++second) {
      for (std::size_t third = second + 1; third < count; ++third) {
        const double left_x = points[second * 2] - points[first * 2];
        const double left_y = points[second * 2 + 1] - points[first * 2 + 1];
        const double right_x = points[third * 2] - points[first * 2];
        const double right_y = points[third * 2 + 1] - points[first * 2 + 1];
        if (std::abs(left_x * right_y - left_y * right_x) >
            kMinimumTriangleArea) {
          return true;
        }
      }
    }
  }
  return false;
}

/// Reports whether fixed correspondence points contain no collinear triple.
template <std::size_t Count>
bool independent_correspondence_points(
    const std::array<cv::Point2f, Count>& points) noexcept {
  constexpr double kMinimumTriangleArea = 1.0e-9;
  for (std::size_t first = 0; first < points.size(); ++first) {
    for (std::size_t second = first + 1; second < points.size(); ++second) {
      for (std::size_t third = second + 1; third < points.size(); ++third) {
        const cv::Point2f left = points[second] - points[first];
        const cv::Point2f right = points[third] - points[first];
        const double twice_area =
            static_cast<double>(left.x) * right.y -
            static_cast<double>(left.y) * right.x;
        if (std::abs(twice_area) <= kMinimumTriangleArea) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Copies fixed validated flattened points into OpenCV point values.
template <std::size_t Count>
std::array<cv::Point2f, Count> copy_correspondence_points(
    rust::Slice<const double> points) noexcept {
  std::array<cv::Point2f, Count> copied{};
  for (std::size_t index = 0; index < copied.size(); ++index) {
    copied[index] = cv::Point2f(static_cast<float>(points[index * 2]),
                               static_cast<float>(points[index * 2 + 1]));
  }
  return copied;
}

/// Copies one validated flattened point sequence into OpenCV point values.
std::vector<cv::Point2d> copy_points(rust::Slice<const double> points) {
  std::vector<cv::Point2d> copied;
  copied.reserve(points.size() / 2);
  for (std::size_t index = 0; index < points.size(); index += 2) {
    copied.emplace_back(points[index], points[index + 1]);
  }
  return copied;
}

/// Copies one validated flattened 3D point sequence into OpenCV point values.
std::vector<cv::Point3d> copy_object_points(
    rust::Slice<const double> points) {
  std::vector<cv::Point3d> copied;
  copied.reserve(points.size() / 3);
  for (std::size_t index = 0; index < points.size(); index += 3) {
    copied.emplace_back(points[index], points[index + 1], points[index + 2]);
  }
  return copied;
}

/// Flattens finite bounded OpenCV points into one copied result.
std::unique_ptr<std::vector<double>> flatten_points(
    const std::vector<cv::Point2d>& points) {
  auto flattened = std::make_unique<std::vector<double>>();
  flattened->reserve(points.size() * 2);
  for (const cv::Point2d& point : points) {
    const double coordinates[] = {point.x, point.y};
    if (!validation::valid_flattened_points(coordinates, 2)) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return nullptr;
    }
    flattened->push_back(point.x);
    flattened->push_back(point.y);
  }
  return flattened;
}

/// Applies one native transform while enforcing the shared copied-point result contract.
template <typename Operation>
std::unique_ptr<std::vector<double>> apply_point_transform(
    rust::Slice<const double> points, Operation operation) {
  const std::vector<cv::Point2d> input = copy_points(points);
  if (input.empty()) {
    return std::make_unique<std::vector<double>>();
  }
  std::vector<cv::Point2d> output;
  operation(input, output);
  if (output.size() != input.size()) {
    failure::set(failure::NativeFailure::NativeException);
    return nullptr;
  }
  return flatten_points(output);
}

/// Reports whether one source and camera model satisfy the bounded policy.
bool valid_undistortion(const OpenCvMat::Impl* source, double fx, double fy,
                        double cx, double cy, double k1, double k2, double p1,
                        double p2, double k3) noexcept {
  return source != nullptr && source->value.depth() == CV_8U &&
         validation::valid_u8_image(source->value.rows, source->value.cols,
                                    source->value.channels()) &&
         source->value.rows <=
             validation::kMaxCopiedAnalysisPixels / source->value.cols &&
         valid_camera_model(source->value, fx, fy, cx, cy, k1, k2, p1, p2,
                            k3);
}

/// Builds one private pinhole camera matrix from validated scalars.
cv::Mat camera_matrix(double fx, double fy, double cx, double cy) {
  return (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0,
          1.0);
}

/// Builds one private five-coefficient distortion matrix.
cv::Mat distortion_matrix(double k1, double k2, double p1, double p2,
                          double k3) {
  return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
}

/// Returns whether one copied Rodrigues rotation and translation are bounded.
bool valid_pose(double rx, double ry, double rz, double tx, double ty,
                double tz) noexcept {
  return validation::valid_vector3(rx, ry, rz) &&
         validation::valid_vector3(tx, ty, tz);
}

/// Copies one validated private OpenCV pose into its public value owner.
std::unique_ptr<CameraPose> own_pose(const cv::Vec3d& rotation,
                                    const cv::Vec3d& translation) {
  if (!valid_pose(rotation[0], rotation[1], rotation[2], translation[0],
                  translation[1], translation[2])) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return std::make_unique<CameraPose>(CameraPose{
      rotation[0], rotation[1], rotation[2], translation[0], translation[1],
      translation[2]});
}

/// Validates matched copied 3D and 2D pose correspondences within a range.
bool valid_pose_correspondences(rust::Slice<const double> object_points,
                                rust::Slice<const double> image_points,
                                std::size_t minimum,
                                std::size_t maximum) noexcept {
  const std::size_t object_count = object_points.size() / 3;
  const std::size_t image_count = image_points.size() / 2;
  return valid_object_points(object_points) && valid_points(image_points) &&
         object_count >= minimum && object_count <= maximum &&
         object_count == image_count;
}

/// Copies one private OpenCV three-element matrix into a native vector.
bool copy_pose_vector(const cv::Mat& source, cv::Vec3d& destination) {
  if (source.total() != 3 || source.channels() != 1 ||
      (source.depth() != CV_32F && source.depth() != CV_64F)) {
    return false;
  }
  cv::Mat values;
  source.reshape(1, 1).convertTo(values, CV_64F);
  destination = cv::Vec3d(values.at<double>(0, 0), values.at<double>(0, 1),
                          values.at<double>(0, 2));
  return true;
}

/// Corrects one source with reviewed private camera matrices.
std::unique_ptr<OpenCvMat> undistort_with_camera(const cv::Mat& source,
                                                 const cv::Mat& camera,
                                                 const cv::Mat& distortion,
                                                 const cv::Mat& output_camera) {
  cv::Mat result;
  cv::undistort(source, result, camera, distortion, output_camera);
  return own_matching_mat(std::move(result), source);
}

/// Selects one reviewed OpenCV pose-refinement algorithm.
enum class PoseRefinement { LevenbergMarquardt, VirtualVisualServoing };

/// Selects one reviewed OpenCV camera-pose solver policy.
enum class PoseSolver { Iterative, Robust };

/// Selects and copies the minimum-error pose from one generic solver call.
std::unique_ptr<CameraPose> solve_candidate_pose(
    const std::vector<cv::Point3d>& objects,
    const std::vector<cv::Point2d>& images, const cv::Mat& camera,
    const cv::Mat& distortion, cv::SolvePnPMethod method) {
  std::vector<cv::Mat> rotations;
  std::vector<cv::Mat> translations;
  cv::Mat errors;
  const int count = cv::solvePnPGeneric(
      objects, images, camera, distortion, rotations, translations, false,
      method, cv::noArray(), cv::noArray(), errors);
  if (count <= 0 || static_cast<std::size_t>(count) != rotations.size() ||
      rotations.size() != translations.size() ||
      errors.total() != rotations.size() || errors.channels() != 1 ||
      (errors.depth() != CV_32F && errors.depth() != CV_64F)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  cv::Mat copied_errors;
  errors.reshape(1, 1).convertTo(copied_errors, CV_64F);
  std::size_t best = 0;
  for (std::size_t index = 0; index < rotations.size(); ++index) {
    const double error = copied_errors.at<double>(0, index);
    if (!std::isfinite(error) || error < 0.0) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return nullptr;
    }
    if (error < copied_errors.at<double>(0, best)) {
      best = index;
    }
  }
  cv::Vec3d rotation;
  cv::Vec3d translation;
  if (!copy_pose_vector(rotations[best], rotation) ||
      !copy_pose_vector(translations[best], translation)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return own_pose(rotation, translation);
}

/// Estimates one validated copied pose through the selected solver policy.
std::unique_ptr<CameraPose> solve_pose_with(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2, double k3,
    PoseSolver method) noexcept {
  failure::clear();
  if (!valid_pose_correspondences(object_points, image_points, 6, 4096) ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<CameraPose>(
      [object_points, image_points, fx, fy, cx, cy, k1, k2, p1, p2, k3,
       method] {
        const std::vector<cv::Point3d> objects =
            copy_object_points(object_points);
        const std::vector<cv::Point2d> images = copy_points(image_points);
        cv::Vec3d rotation{};
        cv::Vec3d translation{};
        cv::Mat inliers;
        const bool solved = method == PoseSolver::Iterative
                                ? cv::solvePnP(
                                      objects, images,
                                      camera_matrix(fx, fy, cx, cy),
                                      distortion_matrix(k1, k2, p1, p2, k3),
                                      rotation, translation, false,
                                      cv::SOLVEPNP_ITERATIVE)
                                : cv::solvePnPRansac(
                                      objects, images,
                                      camera_matrix(fx, fy, cx, cy),
                                      distortion_matrix(k1, k2, p1, p2, k3),
                                      rotation, translation, false, 100, 8.0F,
                                      0.99, inliers, cv::SOLVEPNP_ITERATIVE);
        if (!solved ||
            (method == PoseSolver::Robust &&
             (inliers.total() < 6 || inliers.total() > objects.size()))) {
          failure::set(failure::NativeFailure::InvalidArgument);
          return std::unique_ptr<CameraPose>{};
        }
        return own_pose(rotation, translation);
      });
}

/// Refines one validated copied pose through the selected private algorithm.
std::unique_ptr<CameraPose> refine_pose_with(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3,
    PoseRefinement method) noexcept {
  failure::clear();
  if (!valid_pose_correspondences(object_points, image_points, 6, 4096) ||
      !valid_pose(rx, ry, rz, tx, ty, tz) ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<CameraPose>(
      [object_points, image_points, rx, ry, rz, tx, ty, tz, fx, fy, cx, cy,
       k1, k2, p1, p2, k3, method] {
        const std::vector<cv::Point3d> objects =
            copy_object_points(object_points);
        const std::vector<cv::Point2d> images = copy_points(image_points);
        cv::Vec3d rotation(rx, ry, rz);
        cv::Vec3d translation(tx, ty, tz);
        const cv::TermCriteria criteria(
            cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20,
            std::numeric_limits<float>::epsilon());
        if (method == PoseRefinement::LevenbergMarquardt) {
          cv::solvePnPRefineLM(objects, images, camera_matrix(fx, fy, cx, cy),
                               distortion_matrix(k1, k2, p1, p2, k3),
                               rotation, translation, criteria);
        } else {
          cv::solvePnPRefineVVS(objects, images, camera_matrix(fx, fy, cx, cy),
                                distortion_matrix(k1, k2, p1, p2, k3),
                                rotation, translation, criteria, 1.0);
        }
        return own_pose(rotation, translation);
      });
}

}  // namespace

std::unique_ptr<OpenCvMat> OpenCvMat::undistort_owned(
    double fx, double fy, double cx, double cy, double k1, double k2,
    double p1, double p2, double k3) const noexcept {
  failure::clear();
  if (!valid_undistortion(impl_.get(), fx, fy, cx, cy, k1, k2, p1, p2,
                          k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, fx, fy, cx, cy, k1, k2, p1, p2, k3] {
        const cv::Mat camera = camera_matrix(fx, fy, cx, cy);
        return undistort_with_camera(
            impl_->value, camera,
            distortion_matrix(k1, k2, p1, p2, k3), camera);
      });
}

std::unique_ptr<OpenCvMat> OpenCvMat::undistort_optimal_owned(
    double fx, double fy, double cx, double cy, double k1, double k2,
    double p1, double p2, double k3, double alpha,
    bool center_principal_point) const noexcept {
  failure::clear();
  if (!valid_undistortion(impl_.get(), fx, fy, cx, cy, k1, k2, p1, p2, k3) ||
      !std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMat>(
      [this, fx, fy, cx, cy, k1, k2, p1, p2, k3, alpha,
       center_principal_point] {
        const cv::Mat camera = camera_matrix(fx, fy, cx, cy);
        const cv::Mat distortion = distortion_matrix(k1, k2, p1, p2, k3);
        const cv::Size size(impl_->value.cols, impl_->value.rows);
        const cv::Mat optimal = cv::getOptimalNewCameraMatrix(
            camera, distortion, size, alpha, size, nullptr,
            center_principal_point);
        return undistort_with_camera(impl_->value, camera, distortion,
                                     optimal);
      });
}

std::unique_ptr<std::vector<double>> undistort_points_owned(
    rust::Slice<const double> points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept {
  failure::clear();
  if (!valid_points(points) ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [points, fx, fy, cx, cy, k1, k2, p1, p2, k3] {
        const cv::Mat camera = camera_matrix(fx, fy, cx, cy);
        const cv::Mat distortion = distortion_matrix(k1, k2, p1, p2, k3);
        return apply_point_transform(
            points, [&camera, &distortion](const auto& input, auto& output) {
              cv::undistortPoints(input, output, camera, distortion,
                                  cv::noArray(), camera);
            });
      });
}

std::unique_ptr<std::vector<double>> perspective_transform_points_owned(
    rust::Slice<const double> points, double m00, double m01, double m02,
    double m10, double m11, double m12, double m20, double m21,
    double m22) noexcept {
  failure::clear();
  if (!valid_points(points) ||
      !validation::valid_perspective_transform(
          m00, m01, m02, m10, m11, m12, m20, m21, m22)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [points, m00, m01, m02, m10, m11, m12, m20, m21, m22] {
        const cv::Matx33d transform(m00, m01, m02, m10, m11, m12, m20,
                                    m21, m22);
        return apply_point_transform(
            points, [&transform](const auto& input, auto& output) {
              cv::perspectiveTransform(input, output, transform);
            });
      });
}

std::unique_ptr<std::vector<double>> affine_transform_points_owned(
    rust::Slice<const double> points, double m00, double m01, double m02,
    double m10, double m11, double m12) noexcept {
  failure::clear();
  if (!valid_points(points) ||
      !validation::valid_affine_transform(m00, m01, m02, m10, m11, m12)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<double>>(
      [points, m00, m01, m02, m10, m11, m12] {
        const cv::Matx23d transform(m00, m01, m02, m10, m11, m12);
        return apply_point_transform(
            points, [&transform](const auto& input, auto& output) {
              cv::transform(input, output, transform);
            });
      });
}

std::unique_ptr<PerspectiveTransform> estimate_perspective_transform_owned(
    rust::Slice<const double> source_points,
    rust::Slice<const double> destination_points) noexcept {
  failure::clear();
  if (source_points.size() != 8 || destination_points.size() != 8 ||
      !valid_points(source_points) || !valid_points(destination_points)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const auto source = copy_correspondence_points<4>(source_points);
  const auto destination = copy_correspondence_points<4>(destination_points);
  if (!independent_correspondence_points(source) ||
      !independent_correspondence_points(destination)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<PerspectiveTransform>([&source, &destination] {
    const cv::Mat transform =
        cv::getPerspectiveTransform(source.data(), destination.data());
    if (transform.rows != 3 || transform.cols != 3 ||
        transform.type() != CV_64F) {
      failure::set(failure::NativeFailure::NativeException);
      return std::unique_ptr<PerspectiveTransform>{};
    }
    std::array<double, 9> values{};
    for (int row = 0; row < transform.rows; ++row) {
      for (int column = 0; column < transform.cols; ++column) {
        values[static_cast<std::size_t>(row * transform.cols + column)] =
            transform.at<double>(row, column);
      }
    }
    if (!validation::valid_perspective_transform(
            values[0], values[1], values[2], values[3], values[4], values[5],
            values[6], values[7], values[8]) ||
        std::abs(cv::determinant(transform)) <= 1.0e-12) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<PerspectiveTransform>{};
    }
    return std::make_unique<PerspectiveTransform>(PerspectiveTransform{
        values[0], values[1], values[2], values[3], values[4], values[5],
        values[6], values[7], values[8]});
  });
}

std::unique_ptr<AffineTransform> estimate_affine_transform_owned(
    rust::Slice<const double> source_points,
    rust::Slice<const double> destination_points) noexcept {
  failure::clear();
  if (source_points.size() != 6 || destination_points.size() != 6 ||
      !valid_points(source_points) || !valid_points(destination_points)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  const auto source = copy_correspondence_points<3>(source_points);
  const auto destination = copy_correspondence_points<3>(destination_points);
  if (!independent_correspondence_points(source) ||
      !independent_correspondence_points(destination)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<AffineTransform>([&source, &destination] {
    const cv::Mat transform =
        cv::getAffineTransform(source.data(), destination.data());
    if (transform.rows != 2 || transform.cols != 3 ||
        transform.type() != CV_64F) {
      failure::set(failure::NativeFailure::NativeException);
      return std::unique_ptr<AffineTransform>{};
    }
    std::array<double, 6> values{};
    for (int row = 0; row < transform.rows; ++row) {
      for (int column = 0; column < transform.cols; ++column) {
        values[static_cast<std::size_t>(row * transform.cols + column)] =
            transform.at<double>(row, column);
      }
    }
    const double determinant = values[0] * values[4] - values[1] * values[3];
    if (!validation::valid_affine_transform(values[0], values[1], values[2],
                                            values[3], values[4], values[5]) ||
        std::abs(determinant) <= 1.0e-12) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<AffineTransform>{};
    }
    return std::make_unique<AffineTransform>(AffineTransform{
        values[0], values[1], values[2], values[3], values[4], values[5]});
  });
}

std::unique_ptr<std::vector<double>> project_points_owned(
    rust::Slice<const double> object_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3) noexcept {
  failure::clear();
  if (!valid_object_points(object_points) ||
      !valid_pose(rx, ry, rz, tx, ty, tz) ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  if (object_points.empty()) {
    return std::make_unique<std::vector<double>>();
  }
  return failure::contain<std::vector<double>>(
      [object_points, rx, ry, rz, tx, ty, tz, fx, fy, cx, cy, k1, k2, p1,
       p2, k3] {
        const std::vector<cv::Point3d> input = copy_object_points(object_points);
        std::vector<cv::Point2d> output;
        cv::projectPoints(input, cv::Vec3d(rx, ry, rz), cv::Vec3d(tx, ty, tz),
                          camera_matrix(fx, fy, cx, cy),
                          distortion_matrix(k1, k2, p1, p2, k3), output);
        if (output.size() != input.size()) {
          failure::set(failure::NativeFailure::NativeException);
          return std::unique_ptr<std::vector<double>>{};
        }
        return flatten_points(output);
      });
}

std::unique_ptr<CameraPose> solve_pose_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept {
  return solve_pose_with(object_points, image_points, fx, fy, cx, cy, k1, k2,
                         p1, p2, k3, PoseSolver::Iterative);
}

std::unique_ptr<CameraPose> refine_pose_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3) noexcept {
  return refine_pose_with(object_points, image_points, rx, ry, rz, tx, ty, tz,
                          fx, fy, cx, cy, k1, k2, p1, p2, k3,
                          PoseRefinement::LevenbergMarquardt);
}

std::unique_ptr<CameraPose> refine_pose_vvs_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double rx, double ry, double rz,
    double tx, double ty, double tz, double fx, double fy, double cx, double cy,
    double k1, double k2, double p1, double p2, double k3) noexcept {
  return refine_pose_with(object_points, image_points, rx, ry, rz, tx, ty, tz,
                          fx, fy, cx, cy, k1, k2, p1, p2, k3,
                          PoseRefinement::VirtualVisualServoing);
}

std::unique_ptr<CameraPose> compose_pose_owned(
    double first_rx, double first_ry, double first_rz, double first_tx,
    double first_ty, double first_tz, double second_rx, double second_ry,
    double second_rz, double second_tx, double second_ty,
    double second_tz) noexcept {
  failure::clear();
  if (!valid_pose(first_rx, first_ry, first_rz, first_tx, first_ty,
                  first_tz) ||
      !valid_pose(second_rx, second_ry, second_rz, second_tx, second_ty,
                  second_tz)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<CameraPose>(
      [first_rx, first_ry, first_rz, first_tx, first_ty, first_tz, second_rx,
       second_ry, second_rz, second_tx, second_ty, second_tz] {
        cv::Vec3d rotation;
        cv::Vec3d translation;
        cv::composeRT(cv::Vec3d(first_rx, first_ry, first_rz),
                      cv::Vec3d(first_tx, first_ty, first_tz),
                      cv::Vec3d(second_rx, second_ry, second_rz),
                      cv::Vec3d(second_tx, second_ty, second_tz), rotation,
                      translation);
        return own_pose(rotation, translation);
      });
}

std::unique_ptr<CameraPose> solve_pose_robust_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept {
  return solve_pose_with(object_points, image_points, fx, fy, cx, cy, k1, k2,
                         p1, p2, k3, PoseSolver::Robust);
}

std::unique_ptr<CameraPose> solve_pose_minimal_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept {
  failure::clear();
  if (!valid_pose_correspondences(object_points, image_points, 4, 4) ||
      !independent_minimal_object_points(object_points) ||
      !independent_correspondence_points(
          copy_correspondence_points<4>(image_points)) ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<CameraPose>(
      [object_points, image_points, fx, fy, cx, cy, k1, k2, p1, p2, k3] {
        const std::vector<cv::Point3d> objects =
            copy_object_points(object_points);
        const std::vector<cv::Point2d> images = copy_points(image_points);
        std::vector<cv::Mat> rotations;
        std::vector<cv::Mat> translations;
        const int count = cv::solveP3P(
            objects, images, camera_matrix(fx, fy, cx, cy),
            distortion_matrix(k1, k2, p1, p2, k3), rotations, translations,
            cv::SOLVEPNP_AP3P);
        cv::Vec3d rotation;
        cv::Vec3d translation;
        if (count <= 0 || static_cast<std::size_t>(count) != rotations.size() ||
            rotations.size() != translations.size() ||
            !copy_pose_vector(rotations.front(), rotation) ||
            !copy_pose_vector(translations.front(), translation)) {
          failure::set(failure::NativeFailure::InvalidArgument);
          return std::unique_ptr<CameraPose>{};
        }
        return own_pose(rotation, translation);
      });
}

std::unique_ptr<CameraPose> solve_pose_planar_owned(
    rust::Slice<const double> object_points,
    rust::Slice<const double> image_points, double fx, double fy, double cx,
    double cy, double k1, double k2, double p1, double p2,
    double k3) noexcept {
  failure::clear();
  if (!valid_pose_correspondences(object_points, image_points, 4, 4096) ||
      !coplanar_object_points(object_points) || !spans_image_plane(image_points) ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<CameraPose>(
      [object_points, image_points, fx, fy, cx, cy, k1, k2, p1, p2, k3] {
        const std::vector<cv::Point3d> objects =
            copy_object_points(object_points);
        const std::vector<cv::Point2d> images = copy_points(image_points);
        return solve_candidate_pose(
            objects, images, camera_matrix(fx, fy, cx, cy),
            distortion_matrix(k1, k2, p1, p2, k3), cv::SOLVEPNP_IPPE);
      });
}

std::unique_ptr<CameraPose> solve_square_marker_pose_owned(
    rust::Slice<const double> image_points, double marker_size, double fx,
    double fy, double cx, double cy, double k1, double k2, double p1,
    double p2, double k3) noexcept {
  failure::clear();
  if (!valid_points(image_points) || image_points.size() != 8 ||
      !independent_correspondence_points(
          copy_correspondence_points<4>(image_points)) ||
      !std::isfinite(marker_size) || marker_size <= 0.0 ||
      marker_size > validation::kMaxTransformCoefficient ||
      !valid_camera_scalars(fx, fy, cx, cy, k1, k2, p1, p2, k3)) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<CameraPose>(
      [image_points, marker_size, fx, fy, cx, cy, k1, k2, p1, p2, k3] {
        const double half = marker_size / 2.0;
        const std::vector<cv::Point3d> objects{
            {-half, half, 0.0}, {half, half, 0.0},
            {half, -half, 0.0}, {-half, -half, 0.0}};
        return solve_candidate_pose(
            objects, copy_points(image_points), camera_matrix(fx, fy, cx, cy),
            distortion_matrix(k1, k2, p1, p2, k3),
            cv::SOLVEPNP_IPPE_SQUARE);
      });
}

}  // namespace terlan_opencv
