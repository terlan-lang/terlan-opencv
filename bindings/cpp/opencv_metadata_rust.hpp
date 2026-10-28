#pragma once

#include <cstddef>

namespace rust {

/// Declares the read-only `rust::Slice` surface needed for metadata extraction.
template <typename T>
class Slice {
 public:
  /// Returns the first element address.
  const T* data() const noexcept;
  /// Returns the element count.
  std::size_t size() const noexcept;
  /// Reports whether the slice has no elements.
  bool empty() const noexcept;
  /// Returns one indexed element.
  const T& operator[](std::size_t index) const noexcept;
  /// Returns the first iterator.
  const T* begin() const noexcept;
  /// Returns the past-the-end iterator.
  const T* end() const noexcept;
};

/// Declares the borrowed UTF-8 string surface needed for metadata extraction.
class Str {
 public:
  /// Returns the first UTF-8 byte address.
  const char* data() const noexcept;
  /// Returns the UTF-8 byte count.
  std::size_t size() const noexcept;
  /// Reports whether the string has no bytes.
  bool empty() const noexcept;
};

}  // namespace rust
