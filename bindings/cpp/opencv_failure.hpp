#ifndef TERLAN_OPENCV_FAILURE_HPP
#define TERLAN_OPENCV_FAILURE_HPP

#include <cstdint>
#include <exception>
#include <memory>
#include <new>

namespace terlan_opencv {
namespace failure {

/// Finite native statuses consumed by the generated Rust helper.
enum class NativeFailure : std::int64_t {
  /// No classified failure is pending.
  None = 0,
  /// Public arguments violate package policy.
  InvalidArgument = 1,
  /// A native allocation could not be completed.
  AllocationFailed = 2,
  /// OpenCV or another standard native operation threw.
  NativeException = 3,
  /// An unknown native exception was contained.
  NativeFailure = 4,
  /// Matrix bytes are not contiguous and cannot be copied safely.
  NonContiguous = 5,
  /// Copied input is not a supported encoded image.
  InvalidEncodedImage = 6,
  /// The pinned native build does not provide the requested codec.
  CodecUnavailable = 7,
  /// Encoded image input or output exceeds package policy.
  EncodedImageTooLarge = 8,
  /// Decoded matrix storage exceeds package policy.
  DecodedImageTooLarge = 9,
};

/// Returns the current thread's private failure slot.
inline NativeFailure& current() noexcept {
  static thread_local NativeFailure value = NativeFailure::None;
  return value;
}

/// Clears stale status before one fallible package operation.
inline void clear() noexcept { current() = NativeFailure::None; }

/// Records a finite failure without retaining native payloads.
inline void set(NativeFailure value) noexcept { current() = value; }

/// Returns and clears the current thread's failure status.
inline std::int64_t take() noexcept {
  const auto value = current();
  clear();
  return static_cast<std::int64_t>(value);
}

/// Executes one owned-result operation while containing every C++ exception.
template <typename T, typename Operation>
std::unique_ptr<T> contain(Operation operation) noexcept {
  try {
    return operation();
  } catch (const std::bad_alloc&) {
    set(NativeFailure::AllocationFailed);
  } catch (const std::exception&) {
    set(NativeFailure::NativeException);
  } catch (...) {
    set(NativeFailure::NativeFailure);
  }
  return nullptr;
}

}  // namespace failure
}  // namespace terlan_opencv

#endif
