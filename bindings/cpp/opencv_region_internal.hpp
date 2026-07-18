#ifndef TERLAN_OPENCV_REGION_INTERNAL_HPP
#define TERLAN_OPENCV_REGION_INTERNAL_HPP

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace terlan_opencv {
namespace region {

/// Copies flattened x/y coordinates from canonically ordered point regions.
std::unique_ptr<std::vector<std::int64_t>> copy_points(
    const std::vector<std::vector<cv::Point>>& regions);

/// Copies cumulative point offsets, including the initial zero.
std::unique_ptr<std::vector<std::int64_t>> copy_offsets(
    const std::vector<std::vector<cv::Point>>& regions);

/// Copies flattened x/y/width/height values from aligned region bounds.
std::unique_ptr<std::vector<std::int64_t>> copy_bounds(
    const std::vector<cv::Rect>& bounds);

}  // namespace region
}  // namespace terlan_opencv

#endif
