#include "opencv_region_internal.hpp"

#include "opencv_failure.hpp"
#include "opencv_validation.hpp"

#include <limits>

namespace terlan_opencv {
namespace region {

std::unique_ptr<std::vector<std::int64_t>> copy_points(
    const std::vector<std::vector<cv::Point>>& regions) {
  auto points = std::make_unique<std::vector<std::int64_t>>();
  std::size_t count = 0;
  for (const auto& item : regions) {
    if (item.size() >
        static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) -
            count) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    count += item.size();
  }
  points->reserve(count * 2);
  for (const auto& item : regions) {
    for (const auto& point : item) {
      points->push_back(point.x);
      points->push_back(point.y);
    }
  }
  return points;
}

std::unique_ptr<std::vector<std::int64_t>> copy_offsets(
    const std::vector<std::vector<cv::Point>>& regions) {
  auto offsets = std::make_unique<std::vector<std::int64_t>>();
  offsets->reserve(regions.size() + 1);
  std::int64_t offset = 0;
  offsets->push_back(offset);
  for (const auto& item : regions) {
    if (item.size() > static_cast<std::size_t>(
                          validation::kMaxCopiedAnalysisPixels - offset)) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    offset += static_cast<std::int64_t>(item.size());
    offsets->push_back(offset);
  }
  return offsets;
}

std::unique_ptr<std::vector<std::int64_t>> copy_bounds(
    const std::vector<cv::Rect>& bounds) {
  if (bounds.size() > std::numeric_limits<std::size_t>::max() / 4) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  auto result = std::make_unique<std::vector<std::int64_t>>();
  result->reserve(bounds.size() * 4);
  for (const auto& bound : bounds) {
    result->push_back(bound.x);
    result->push_back(bound.y);
    result->push_back(bound.width);
    result->push_back(bound.height);
  }
  return result;
}

}  // namespace region
}  // namespace terlan_opencv
