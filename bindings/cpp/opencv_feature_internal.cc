#include "opencv_feature_internal.hpp"

#include "opencv_failure.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <utility>

namespace terlan_opencv {
namespace feature {

Data::Data(std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors,
           int descriptor_width, int descriptor_type)
    : keypoints(std::move(keypoints)),
      descriptors(std::move(descriptors)),
      descriptor_width(descriptor_width),
      descriptor_type(descriptor_type) {}

bool valid_keypoint(const cv::KeyPoint& keypoint, int rows, int cols,
                    int maximum_octaves, bool allow_unoriented) noexcept {
  const float minimum_angle = allow_unoriented ? -1.0F : 0.0F;
  return std::isfinite(keypoint.pt.x) && std::isfinite(keypoint.pt.y) &&
         std::isfinite(keypoint.size) && std::isfinite(keypoint.angle) &&
         std::isfinite(keypoint.response) && keypoint.pt.x >= 0.0F &&
         keypoint.pt.x < static_cast<float>(cols) && keypoint.pt.y >= 0.0F &&
         keypoint.pt.y < static_cast<float>(rows) && keypoint.size > 0.0F &&
         keypoint.angle >= minimum_angle && keypoint.angle < 360.0F &&
         (maximum_octaves < 0 ||
          (keypoint.octave >= 0 && keypoint.octave < maximum_octaves));
}

bool keypoint_less(const cv::KeyPoint& left,
                   const cv::KeyPoint& right) noexcept {
  if (left.pt.y != right.pt.y) {
    return left.pt.y < right.pt.y;
  }
  if (left.pt.x != right.pt.x) {
    return left.pt.x < right.pt.x;
  }
  if (left.octave != right.octave) {
    return left.octave < right.octave;
  }
  if (left.response != right.response) {
    return left.response > right.response;
  }
  if (left.size != right.size) {
    return left.size < right.size;
  }
  return left.angle < right.angle;
}

bool truncate_by_response(std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors,
                          std::int64_t maximum) {
  if (maximum < 1 || descriptors.rows != static_cast<int>(keypoints.size())) {
    return false;
  }
  if (keypoints.size() <= static_cast<std::size_t>(maximum)) {
    return true;
  }
  for (const auto& keypoint : keypoints) {
    if (!std::isfinite(keypoint.response)) {
      return false;
    }
  }
  std::vector<std::size_t> order(keypoints.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&keypoints](std::size_t left,
                                                            std::size_t right) {
    if (keypoints[left].response != keypoints[right].response) {
      return keypoints[left].response > keypoints[right].response;
    }
    return keypoint_less(keypoints[left], keypoints[right]);
  });
  const auto retained = static_cast<std::size_t>(maximum);
  std::vector<cv::KeyPoint> truncated_keypoints;
  truncated_keypoints.reserve(retained);
  cv::Mat truncated_descriptors(static_cast<int>(retained), descriptors.cols,
                                descriptors.type());
  for (std::size_t row = 0; row < retained; ++row) {
    truncated_keypoints.push_back(keypoints[order[row]]);
    descriptors.row(static_cast<int>(order[row]))
        .copyTo(truncated_descriptors.row(static_cast<int>(row)));
  }
  if (!truncated_descriptors.isContinuous()) {
    return false;
  }
  keypoints = std::move(truncated_keypoints);
  descriptors = std::move(truncated_descriptors);
  return true;
}

bool canonicalize(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                  std::int64_t maximum, int rows, int cols,
                  int descriptor_width, int descriptor_type,
                  int maximum_octaves, bool allow_unoriented) {
  if (keypoints.size() > static_cast<std::size_t>(maximum)) {
    return false;
  }
  if (keypoints.empty()) {
    if (!descriptors.empty()) {
      return false;
    }
    descriptors.release();
    return true;
  }
  if (descriptors.type() != descriptor_type ||
      descriptors.rows != static_cast<int>(keypoints.size()) ||
      descriptors.cols != descriptor_width) {
    return false;
  }
  for (const auto& keypoint : keypoints) {
    if (!valid_keypoint(keypoint, rows, cols, maximum_octaves,
                        allow_unoriented)) {
      return false;
    }
  }
  if (descriptor_type == CV_32FC1) {
    for (int row = 0; row < descriptors.rows; ++row) {
      const float* values = descriptors.ptr<float>(row);
      for (int col = 0; col < descriptors.cols; ++col) {
        if (!std::isfinite(values[col])) {
          return false;
        }
      }
    }
  }

  std::vector<std::size_t> order(keypoints.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&keypoints](std::size_t left,
                                                            std::size_t right) {
    return keypoint_less(keypoints[left], keypoints[right]);
  });
  std::vector<cv::KeyPoint> sorted_keypoints;
  sorted_keypoints.reserve(keypoints.size());
  cv::Mat sorted_descriptors(descriptors.rows, descriptors.cols,
                             descriptors.type());
  for (std::size_t row = 0; row < order.size(); ++row) {
    sorted_keypoints.push_back(keypoints[order[row]]);
    descriptors.row(static_cast<int>(order[row]))
        .copyTo(sorted_descriptors.row(static_cast<int>(row)));
  }
  if (!sorted_descriptors.isContinuous()) {
    return false;
  }
  keypoints = std::move(sorted_keypoints);
  descriptors = std::move(sorted_descriptors);
  return true;
}

std::unique_ptr<std::vector<double>> copy_keypoints(const Data& data) {
  auto result = std::make_unique<std::vector<double>>();
  result->reserve(data.keypoints.size() * 2);
  for (const auto& keypoint : data.keypoints) {
    result->push_back(keypoint.pt.x);
    result->push_back(keypoint.pt.y);
  }
  return result;
}

std::unique_ptr<std::vector<std::uint8_t>> copy_u8_descriptors(
    const Data& data) {
  auto result = std::make_unique<std::vector<std::uint8_t>>();
  if (data.descriptors.empty()) {
    return result;
  }
  if (data.descriptor_type != CV_8UC1 || !data.descriptors.isContinuous()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  const auto count = data.descriptors.total();
  result->resize(count);
  std::memcpy(result->data(), data.descriptors.ptr<std::uint8_t>(), count);
  return result;
}

std::unique_ptr<std::vector<double>> copy_f32_descriptors(const Data& data) {
  auto result = std::make_unique<std::vector<double>>();
  if (data.descriptors.empty()) {
    return result;
  }
  if (data.descriptor_type != CV_32FC1 || !data.descriptors.isContinuous()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  const auto count = data.descriptors.total();
  const float* values = data.descriptors.ptr<float>();
  result->reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    result->push_back(values[index]);
  }
  return result;
}

bool match(const Data& query, const Data& train, std::int64_t maximum,
           int norm_type, double maximum_distance, bool integral_distance,
           std::vector<Match>& result) {
  if (query.keypoints.empty() || train.keypoints.empty()) {
    return true;
  }
  if (query.descriptor_width != train.descriptor_width ||
      query.descriptor_width < 1 ||
      query.descriptor_type != train.descriptor_type ||
      query.descriptors.type() != query.descriptor_type ||
      train.descriptors.type() != train.descriptor_type ||
      query.descriptors.rows != static_cast<int>(query.keypoints.size()) ||
      train.descriptors.rows != static_cast<int>(train.keypoints.size()) ||
      query.descriptors.cols != query.descriptor_width ||
      train.descriptors.cols != train.descriptor_width) {
    failure::set(failure::NativeFailure::NativeFailure);
    return false;
  }
  auto matcher = cv::BFMatcher::create(norm_type, true);
  if (matcher.empty()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return false;
  }
  std::vector<cv::DMatch> matches;
  matcher->match(query.descriptors, train.descriptors, matches);
  if (matches.size() > query.keypoints.size() ||
      matches.size() > train.keypoints.size()) {
    failure::set(failure::NativeFailure::NativeFailure);
    return false;
  }
  std::vector<bool> seen_query(query.keypoints.size(), false);
  std::vector<bool> seen_train(train.keypoints.size(), false);
  result.reserve(matches.size());
  for (const auto& item : matches) {
    if (item.queryIdx < 0 || item.trainIdx < 0 || item.imgIdx != 0 ||
        static_cast<std::size_t>(item.queryIdx) >= query.keypoints.size() ||
        static_cast<std::size_t>(item.trainIdx) >= train.keypoints.size() ||
        !std::isfinite(item.distance) || item.distance < 0.0F ||
        item.distance > maximum_distance ||
        (integral_distance && std::trunc(item.distance) != item.distance) ||
        seen_query[item.queryIdx] || seen_train[item.trainIdx]) {
      failure::set(failure::NativeFailure::NativeFailure);
      return false;
    }
    seen_query[item.queryIdx] = true;
    seen_train[item.trainIdx] = true;
    const auto& query_point = query.keypoints[item.queryIdx].pt;
    const auto& train_point = train.keypoints[item.trainIdx].pt;
    result.push_back({item.queryIdx, item.trainIdx, item.distance,
                      query_point.x, query_point.y, train_point.x,
                      train_point.y});
  }
  std::sort(result.begin(), result.end(), [](const Match& left,
                                             const Match& right) {
    if (left.distance != right.distance) {
      return left.distance < right.distance;
    }
    if (left.query_index != right.query_index) {
      return left.query_index < right.query_index;
    }
    return left.train_index < right.train_index;
  });
  if (result.size() > static_cast<std::size_t>(maximum)) {
    result.resize(static_cast<std::size_t>(maximum));
  }
  return true;
}

std::unique_ptr<std::vector<std::int64_t>> copy_match_indices(
    const std::vector<Match>& matches) {
  auto result = std::make_unique<std::vector<std::int64_t>>();
  result->reserve(matches.size() * 2);
  for (const auto& item : matches) {
    result->push_back(item.query_index);
    result->push_back(item.train_index);
  }
  return result;
}

std::unique_ptr<std::vector<std::int64_t>> copy_integral_match_distances(
    const std::vector<Match>& matches) {
  auto result = std::make_unique<std::vector<std::int64_t>>();
  result->reserve(matches.size());
  for (const auto& item : matches) {
    if (std::trunc(item.distance) != item.distance) {
      failure::set(failure::NativeFailure::NativeFailure);
      return nullptr;
    }
    result->push_back(static_cast<std::int64_t>(item.distance));
  }
  return result;
}

std::unique_ptr<std::vector<double>> copy_match_distances(
    const std::vector<Match>& matches) {
  auto result = std::make_unique<std::vector<double>>();
  result->reserve(matches.size());
  for (const auto& item : matches) {
    result->push_back(item.distance);
  }
  return result;
}

std::unique_ptr<std::vector<double>> copy_match_coordinates(
    const std::vector<Match>& matches) {
  auto result = std::make_unique<std::vector<double>>();
  result->reserve(matches.size() * 4);
  for (const auto& item : matches) {
    result->push_back(item.query_x);
    result->push_back(item.query_y);
    result->push_back(item.train_x);
    result->push_back(item.train_y);
  }
  return result;
}

}  // namespace feature
}  // namespace terlan_opencv
