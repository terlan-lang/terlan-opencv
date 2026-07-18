#ifndef TERLAN_OPENCV_FEATURE_INTERNAL_HPP
#define TERLAN_OPENCV_FEATURE_INTERNAL_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace terlan_opencv {
namespace feature {

/// Canonical keypoints and their row-aligned descriptor matrix.
struct Data final {
  /// Constructs one canonical feature set.
  Data(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors,
       int descriptor_width, int descriptor_type);

  /// Canonically ordered keypoints.
  std::vector<cv::KeyPoint> keypoints;
  /// Continuous row-aligned descriptor storage.
  cv::Mat descriptors;
  /// Number of descriptor elements in each row.
  int descriptor_width;
  /// OpenCV scalar type used by each descriptor element.
  int descriptor_type;
};

/// One canonical cross-checked descriptor match.
struct Match final {
  std::int64_t query_index;  ///< Query descriptor row.
  std::int64_t train_index;  ///< Train descriptor row.
  double distance;           ///< Validated descriptor distance.
  double query_x;            ///< Query keypoint x coordinate.
  double query_y;            ///< Query keypoint y coordinate.
  double train_x;            ///< Train keypoint x coordinate.
  double train_y;            ///< Train keypoint y coordinate.
};

/// Checks one bounded feature keypoint and its orientation policy.
bool valid_keypoint(const cv::KeyPoint& keypoint, int rows, int cols,
                    int maximum_octaves, bool allow_unoriented) noexcept;

/// Orders two keypoints canonically for descriptor-row projection.
bool keypoint_less(const cv::KeyPoint& left,
                   const cv::KeyPoint& right) noexcept;

/// Retains the strongest keypoints and their aligned descriptor rows.
bool truncate_by_response(std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors,
                          std::int64_t maximum);

/// Validates and canonically aligns keypoints with descriptor rows.
bool canonicalize(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                  std::int64_t maximum, int rows, int cols,
                  int descriptor_width, int descriptor_type,
                  int maximum_octaves, bool allow_unoriented);

/// Copies flattened keypoint coordinates from one canonical feature set.
std::unique_ptr<std::vector<double>> copy_keypoints(const Data& data);

/// Copies unsigned-byte descriptor elements in canonical row order.
std::unique_ptr<std::vector<std::uint8_t>> copy_u8_descriptors(
    const Data& data);

/// Copies floating-point descriptor elements in canonical row order.
std::unique_ptr<std::vector<double>> copy_f32_descriptors(const Data& data);

/// Cross-checks compatible descriptors with one fixed OpenCV norm.
bool match(const Data& query, const Data& train, std::int64_t maximum,
           int norm_type, double maximum_distance, bool integral_distance,
           std::vector<Match>& result);

/// Copies flattened query and train descriptor-row indices.
std::unique_ptr<std::vector<std::int64_t>> copy_match_indices(
    const std::vector<Match>& matches);

/// Copies validated integral match distances.
std::unique_ptr<std::vector<std::int64_t>> copy_integral_match_distances(
    const std::vector<Match>& matches);

/// Copies floating-point match distances.
std::unique_ptr<std::vector<double>> copy_match_distances(
    const std::vector<Match>& matches);

/// Copies query and train endpoint coordinates.
std::unique_ptr<std::vector<double>> copy_match_coordinates(
    const std::vector<Match>& matches);

}  // namespace feature
}  // namespace terlan_opencv

#endif
