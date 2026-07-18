#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"
#include "opencv_region_internal.hpp"
#include "opencv_validation.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <utility>

namespace terlan_opencv {
namespace {

constexpr double kMserMaximumVariation = 0.25;
constexpr double kMserMinimumDiversity = 0.2;
constexpr int kMserMaximumEvolution = 200;
constexpr double kMserAreaThreshold = 1.01;
constexpr double kMserMinimumMargin = 0.003;
constexpr int kMserEdgeBlurSize = 5;
constexpr std::int64_t kMaximumMserRegions = 4096;
std::atomic<std::int64_t> live_mser_regions{0};

/// Couples one detected point region to its validated bounding rectangle.
struct RegionCandidate final {
  std::vector<cv::Point> points;
  cv::Rect bounds;
};

/// Orders points in row-major image order.
bool point_less(const cv::Point& left, const cv::Point& right) noexcept {
  return left.y < right.y || (left.y == right.y && left.x < right.x);
}

/// Orders stronger region candidates before bounded truncation.
bool stronger_region(const RegionCandidate& left,
                     const RegionCandidate& right) noexcept {
  if (left.points.size() != right.points.size()) {
    return left.points.size() > right.points.size();
  }
  if (left.bounds.y != right.bounds.y) {
    return left.bounds.y < right.bounds.y;
  }
  if (left.bounds.x != right.bounds.x) {
    return left.bounds.x < right.bounds.x;
  }
  if (left.bounds.height != right.bounds.height) {
    return left.bounds.height < right.bounds.height;
  }
  return left.bounds.width < right.bounds.width;
}

/// Orders retained regions canonically by bounds and then point count.
bool earlier_region(const RegionCandidate& left,
                    const RegionCandidate& right) noexcept {
  if (left.bounds.y != right.bounds.y) {
    return left.bounds.y < right.bounds.y;
  }
  if (left.bounds.x != right.bounds.x) {
    return left.bounds.x < right.bounds.x;
  }
  if (left.bounds.height != right.bounds.height) {
    return left.bounds.height < right.bounds.height;
  }
  if (left.bounds.width != right.bounds.width) {
    return left.bounds.width < right.bounds.width;
  }
  if (left.points.size() != right.points.size()) {
    return left.points.size() < right.points.size();
  }
  return std::lexicographical_compare(
      left.points.begin(), left.points.end(), right.points.begin(),
      right.points.end(), point_less);
}

/// Validates and canonicalizes one native MSER result.
bool canonicalize_regions(std::vector<std::vector<cv::Point>> detected,
                          std::vector<cv::Rect> detected_bounds,
                          int rows, int cols, std::int64_t minimum_area,
                          std::int64_t maximum_area,
                          std::int64_t max_regions,
                          std::vector<std::vector<cv::Point>>& regions,
                          std::vector<cv::Rect>& bounds) {
  if (detected.size() != detected_bounds.size()) {
    return false;
  }
  std::vector<RegionCandidate> candidates;
  candidates.reserve(detected.size());
  for (std::size_t index = 0; index < detected.size(); ++index) {
    auto& points = detected[index];
    const auto& bound = detected_bounds[index];
    if (points.size() < static_cast<std::size_t>(minimum_area) ||
        points.size() > static_cast<std::size_t>(maximum_area) ||
        bound.x < 0 || bound.y < 0 || bound.width <= 0 || bound.height <= 0 ||
        bound.x > cols - bound.width || bound.y > rows - bound.height) {
      return false;
    }
    std::sort(points.begin(), points.end(), point_less);
    for (std::size_t point_index = 0; point_index < points.size();
         ++point_index) {
      const auto& point = points[point_index];
      if (point.x < bound.x || point.y < bound.y ||
          point.x >= bound.x + bound.width ||
          point.y >= bound.y + bound.height ||
          (point_index > 0 && points[point_index - 1] == point)) {
        return false;
      }
    }
    if (cv::boundingRect(points) != bound) {
      return false;
    }
    candidates.push_back({std::move(points), bound});
  }
  std::stable_sort(candidates.begin(), candidates.end(), stronger_region);
  if (candidates.size() > static_cast<std::size_t>(max_regions)) {
    candidates.resize(static_cast<std::size_t>(max_regions));
  }
  std::sort(candidates.begin(), candidates.end(), earlier_region);
  std::size_t point_count = 0;
  regions.reserve(candidates.size());
  bounds.reserve(candidates.size());
  for (auto& candidate : candidates) {
    if (candidate.points.size() >
        static_cast<std::size_t>(validation::kMaxCopiedAnalysisPixels) -
            point_count) {
      return false;
    }
    point_count += candidate.points.size();
    regions.push_back(std::move(candidate.points));
    bounds.push_back(candidate.bounds);
  }
  return true;
}

}  // namespace

struct OpenCvMserRegions::Impl final {
  /// Stores canonical point regions and their aligned bounds.
  Impl(std::vector<std::vector<cv::Point>> regions,
       std::vector<cv::Rect> bounds)
      : regions(std::move(regions)), bounds(std::move(bounds)) {}

  std::vector<std::vector<cv::Point>> regions;
  std::vector<cv::Rect> bounds;
};

std::unique_ptr<OpenCvMserRegions> OpenCvMat::mser_regions_owned(
    std::int64_t delta, std::int64_t minimum_area,
    std::int64_t maximum_area, std::int64_t max_regions) const noexcept {
  failure::clear();
  const std::int64_t image_area = impl_ == nullptr
                                      ? 0
                                      : static_cast<std::int64_t>(
                                            impl_->value.rows) *
                                            impl_->value.cols;
  if (impl_ == nullptr || !valid_u8_grayscale_image(impl_->value) ||
      !validation::valid_analysis_image(
          impl_->value.rows, impl_->value.cols, impl_->value.channels()) ||
      impl_->value.rows < 3 || impl_->value.cols < 3 || delta < 1 ||
      delta > 255 || minimum_area < 1 || maximum_area < minimum_area ||
      maximum_area > image_area || max_regions < 1 ||
      max_regions > kMaximumMserRegions ||
      maximum_area > validation::kMaxCopiedAnalysisPixels / max_regions) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<OpenCvMserRegions>(
      [this, delta, minimum_area, maximum_area, max_regions] {
        auto detector = cv::MSER::create(
            static_cast<int>(delta), static_cast<int>(minimum_area),
            static_cast<int>(maximum_area), kMserMaximumVariation,
            kMserMinimumDiversity, kMserMaximumEvolution, kMserAreaThreshold,
            kMserMinimumMargin, kMserEdgeBlurSize);
        if (detector.empty() || detector->getDelta() != delta ||
            detector->getMinArea() != minimum_area ||
            detector->getMaxArea() != maximum_area ||
            detector->getMaxVariation() != kMserMaximumVariation ||
            detector->getMinDiversity() != kMserMinimumDiversity ||
            detector->getMaxEvolution() != kMserMaximumEvolution ||
            detector->getAreaThreshold() != kMserAreaThreshold ||
            detector->getMinMargin() != kMserMinimumMargin ||
            detector->getEdgeBlurSize() != kMserEdgeBlurSize ||
            detector->getPass2Only()) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMserRegions>{};
        }
        std::vector<std::vector<cv::Point>> detected;
        std::vector<cv::Rect> detected_bounds;
        detector->detectRegions(impl_->value, detected, detected_bounds);
        std::vector<std::vector<cv::Point>> regions;
        std::vector<cv::Rect> bounds;
        if (!canonicalize_regions(
                std::move(detected), std::move(detected_bounds),
                impl_->value.rows, impl_->value.cols, minimum_area,
                maximum_area, max_regions, regions, bounds)) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::unique_ptr<OpenCvMserRegions>{};
        }
        return std::make_unique<OpenCvMserRegions>(
            std::make_unique<OpenCvMserRegions::Impl>(
                std::move(regions), std::move(bounds)));
      });
}

OpenCvMserRegions::OpenCvMserRegions(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
  live_mser_regions.fetch_add(1, std::memory_order_relaxed);
}

OpenCvMserRegions::~OpenCvMserRegions() noexcept {
  live_mser_regions.fetch_sub(1, std::memory_order_relaxed);
}

std::int64_t OpenCvMserRegions::count() const noexcept {
  return impl_ == nullptr ? 0
                          : static_cast<std::int64_t>(impl_->regions.size());
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMserRegions::points_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return region::copy_points(impl_->regions); });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMserRegions::offsets_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return region::copy_offsets(impl_->regions); });
}

std::unique_ptr<std::vector<std::int64_t>>
OpenCvMserRegions::bounds_owned() const noexcept {
  failure::clear();
  if (impl_ == nullptr) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::int64_t>>(
      [this] { return region::copy_bounds(impl_->bounds); });
}

std::int64_t live_mser_region_count() noexcept {
  return live_mser_regions.load(std::memory_order_relaxed);
}

}  // namespace terlan_opencv
