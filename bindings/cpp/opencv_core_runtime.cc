#include "opencv_adapter.hpp"
#include "opencv_failure.hpp"
#include "opencv_internal.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace terlan_opencv {
namespace {

/// Maximum copied OpenCV build-information size.
constexpr std::size_t kMaxBuildInformationBytes = 1024U * 1024U;

/// Maximum copied tensor payload admitted by the public native boundary.
constexpr std::size_t kMaxTensorPacketBytes = 16U * 1024U * 1024U;

/// Fixed byte count preceding tensor dimensions and strides.
constexpr std::size_t kTensorPacketHeaderBytes = 16U;

/// Reads one little-endian unsigned 16-bit packet field.
std::uint16_t read_u16_le(const std::uint8_t* input) noexcept {
  return static_cast<std::uint16_t>(input[0]) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(input[1]) << 8U);
}

/// Reads one little-endian unsigned 32-bit packet field.
std::uint32_t read_u32_le(const std::uint8_t* input) noexcept {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < 4U; ++index) {
    value |= static_cast<std::uint32_t>(input[index]) << (index * 8U);
  }
  return value;
}

/// Reads one little-endian unsigned 64-bit packet field.
std::uint64_t read_u64_le(const std::uint8_t* input) noexcept {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8U; ++index) {
    value |= static_cast<std::uint64_t>(input[index]) << (index * 8U);
  }
  return value;
}

/// Appends one unsigned 32-bit value in the packet's fixed byte order.
void append_u32_le(std::vector<std::uint8_t>* output,
                   std::uint32_t value) {
  for (std::size_t index = 0; index < 4U; ++index) {
    output->push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

/// Appends one unsigned 64-bit value in the packet's fixed byte order.
void append_u64_le(std::vector<std::uint8_t>* output,
                   std::uint64_t value) {
  for (std::size_t index = 0; index < 8U; ++index) {
    output->push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

/// Returns the packet marker for the host's scalar byte order.
std::uint8_t native_endian_code() noexcept {
  const std::uint16_t value = 1U;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1U ? 1U : 2U;
}

/// Maps one admitted DLPack scalar descriptor to an OpenCV depth.
int packet_matrix_depth(std::uint8_t code, std::uint8_t bits) noexcept {
  if ((code == 1U || code == 6U) && bits == 8U) {
    return CV_8U;
  }
  if (code == 0U && bits == 32U) {
    return CV_32S;
  }
  if (code == 2U && bits == 32U) {
    return CV_32F;
  }
  if (code == 2U && bits == 64U) {
    return CV_64F;
  }
  return -1;
}

/// Maps one OpenCV depth to a DLPack scalar descriptor.
bool matrix_packet_dtype(int depth, std::uint8_t* code,
                         std::uint8_t* bits) noexcept {
  switch (depth) {
    case CV_8U:
      *code = 1U;
      *bits = 8U;
      return true;
    case CV_32S:
      *code = 0U;
      *bits = 32U;
      return true;
    case CV_32F:
      *code = 2U;
      *bits = 32U;
      return true;
    case CV_64F:
      *code = 2U;
      *bits = 64U;
      return true;
    default:
      return false;
  }
}

/// Copies one bounded nonempty runtime string.
std::unique_ptr<std::string> copy_runtime_string(const std::string& value) {
  if (value.empty() || value.size() > kMaxBuildInformationBytes) {
    failure::set(failure::NativeFailure::NativeFailure);
    return nullptr;
  }
  return std::make_unique<std::string>(value);
}

}  // namespace

std::unique_ptr<std::string> opencv_version_owned() noexcept {
  failure::clear();
  return failure::contain<std::string>(
      [] { return copy_runtime_string(cv::getVersionString()); });
}

std::unique_ptr<std::string> opencv_build_information_owned() noexcept {
  failure::clear();
  return failure::contain<std::string>(
      [] { return copy_runtime_string(cv::getBuildInformation()); });
}

std::int64_t opencv_processor_count() noexcept {
  failure::clear();
  return failure::contain_value<std::int64_t>(
      [] {
        const int count = cv::getNumberOfCPUs();
        if (count <= 0) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::int64_t{0};
        }
        return static_cast<std::int64_t>(count);
      },
      0);
}

std::int64_t opencv_thread_count() noexcept {
  failure::clear();
  return failure::contain_value<std::int64_t>(
      [] {
        const int count = cv::getNumThreads();
        if (count <= 0) {
          failure::set(failure::NativeFailure::NativeFailure);
          return std::int64_t{0};
        }
        return static_cast<std::int64_t>(count);
      },
      0);
}

double opencv_tick_frequency() noexcept {
  failure::clear();
  return failure::contain_value<double>(
      [] {
        const double frequency = cv::getTickFrequency();
        if (!std::isfinite(frequency) || frequency <= 0.0) {
          failure::set(failure::NativeFailure::NativeFailure);
          return 0.0;
        }
        return frequency;
      },
      0.0);
}

bool opencv_optimized_enabled() noexcept {
  failure::clear();
  return failure::contain_value<bool>([] { return cv::useOptimized(); }, false);
}

std::unique_ptr<OpenCvMat> mat_from_tensor_packet(
    rust::Slice<const std::uint8_t> packet) noexcept {
  failure::clear();
  return failure::contain<OpenCvMat>([packet] {
    if (packet.size() < kTensorPacketHeaderBytes || packet.size() > kMaxTensorPacketBytes ||
        std::memcmp(packet.data(), "TNXP", 4U) != 0 || packet[4] != 1U ||
        packet[5] != native_endian_code() || read_u16_le(packet.data() + 8U) != 1U ||
        packet[10] != 1U || packet[11] != 0U) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    const std::uint32_t rank = read_u32_le(packet.data() + 12U);
    if (rank != 2U && rank != 3U) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    const std::size_t metadata_bytes =
        kTensorPacketHeaderBytes + static_cast<std::size_t>(rank) * 16U;
    if (packet.size() < metadata_bytes) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    std::vector<std::uint64_t> dimensions(rank);
    std::uint64_t expected_stride = 1U;
    for (std::size_t index = 0; index < rank; ++index) {
      dimensions[index] = read_u64_le(packet.data() + 16U + index * 8U);
      if (dimensions[index] == 0U ||
          dimensions[index] > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        failure::set(failure::NativeFailure::InvalidArgument);
        return std::unique_ptr<OpenCvMat>{};
      }
    }
    for (std::size_t index = rank; index > 0U; --index) {
      const std::uint64_t stride = read_u64_le(
          packet.data() + 16U + static_cast<std::size_t>(rank) * 8U + (index - 1U) * 8U);
      if (stride != expected_stride ||
          expected_stride > std::numeric_limits<std::uint64_t>::max() / dimensions[index - 1U]) {
        failure::set(failure::NativeFailure::InvalidArgument);
        return std::unique_ptr<OpenCvMat>{};
      }
      expected_stride *= dimensions[index - 1U];
    }
    const int channels = rank == 3U ? static_cast<int>(dimensions[2]) : 1;
    if (channels <= 0 || channels > validation::kMaxMatrixChannels) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    const int depth = packet_matrix_depth(packet[6], packet[7]);
    if (depth < 0) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    const std::size_t data_bytes = packet.size() - metadata_bytes;
    const std::size_t scalar_bytes = CV_ELEM_SIZE1(depth);
    if (expected_stride > kMaxTensorPacketBytes / scalar_bytes ||
        data_bytes != static_cast<std::size_t>(expected_stride) * scalar_bytes) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<OpenCvMat>{};
    }
    cv::Mat matrix(static_cast<int>(dimensions[0]), static_cast<int>(dimensions[1]),
                   CV_MAKETYPE(depth, channels));
    std::memcpy(matrix.data, packet.data() + metadata_bytes, data_bytes);
    return std::make_unique<OpenCvMat>(
        std::make_unique<OpenCvMat::Impl>(std::move(matrix)));
  });
}

std::unique_ptr<std::vector<std::uint8_t>> OpenCvMat::to_tensor_packet(
    rust::Str consumer) const noexcept {
  failure::clear();
  (void)consumer;
  if (impl_ == nullptr || !impl_->value.isContinuous() || impl_->value.dims != 2) {
    failure::set(failure::NativeFailure::InvalidArgument);
    return nullptr;
  }
  return failure::contain<std::vector<std::uint8_t>>([this] {
    std::uint8_t dtype_code = 0U;
    std::uint8_t dtype_bits = 0U;
    if (!matrix_packet_dtype(impl_->value.depth(), &dtype_code, &dtype_bits)) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<std::vector<std::uint8_t>>{};
    }
    const std::uint32_t rank = impl_->value.channels() == 1 ? 2U : 3U;
    const std::uint64_t rows = static_cast<std::uint64_t>(impl_->value.rows);
    const std::uint64_t cols = static_cast<std::uint64_t>(impl_->value.cols);
    const std::uint64_t channels = static_cast<std::uint64_t>(impl_->value.channels());
    const std::size_t data_bytes = impl_->value.total() * impl_->value.elemSize();
    if (data_bytes > kMaxTensorPacketBytes) {
      failure::set(failure::NativeFailure::InvalidArgument);
      return std::unique_ptr<std::vector<std::uint8_t>>{};
    }
    auto packet = std::make_unique<std::vector<std::uint8_t>>();
    packet->reserve(kTensorPacketHeaderBytes + static_cast<std::size_t>(rank) * 16U + data_bytes);
    packet->insert(packet->end(), {'T', 'N', 'X', 'P', 1U, native_endian_code(), dtype_code,
                                   dtype_bits, 1U, 0U, 1U, 0U});
    append_u32_le(packet.get(), rank);
    append_u64_le(packet.get(), rows);
    append_u64_le(packet.get(), cols);
    if (rank == 3U) {
      append_u64_le(packet.get(), channels);
    }
    append_u64_le(packet.get(), cols * channels);
    append_u64_le(packet.get(), channels);
    if (rank == 3U) {
      append_u64_le(packet.get(), 1U);
    }
    packet->insert(packet->end(), impl_->value.data, impl_->value.data + data_bytes);
    return packet;
  });
}

}  // namespace terlan_opencv
