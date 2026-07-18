//! Direct grayscale gradient-magnitude tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact magnitude, saturation, validation, immutability, and cleanup.
#[test]
fn gradient_magnitude_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);
    let horizontal_bytes = [0, 3, 6, 255, 5, 8];
    let vertical_bytes = [0, 4, 8, 255, 12, 15];
    let horizontal = ffi::mat_from_u8(&horizontal_bytes, 2, 3, 1);
    let vertical = ffi::mat_from_u8(&vertical_bytes, 2, 3, 1);
    let horizontal_ref = horizontal.as_ref().expect("horizontal component");
    let vertical_ref = vertical.as_ref().expect("vertical component");
    let magnitude = horizontal_ref.gradient_magnitude_owned(vertical_ref);
    assert_eq!(
        mat_bytes(magnitude.as_ref().expect("gradient magnitude")),
        [0, 5, 10, 255, 13, 17]
    );
    assert_eq!(mat_bytes(horizontal_ref), horizontal_bytes);
    assert_eq!(mat_bytes(vertical_ref), vertical_bytes);

    let zero = ffi::mat_zeros(2, 3, 1);
    let zero_ref = zero.as_ref().expect("zero component");
    let aliased_zero = zero_ref.gradient_magnitude_owned(zero_ref);
    assert_eq!(
        mat_bytes(aliased_zero.as_ref().expect("aliased zero magnitude")),
        [0; 6]
    );
    let wrong_shape = ffi::mat_zeros(2, 2, 1);
    let bgr = ffi::mat_zeros(2, 3, 3);
    let bgra = ffi::mat_zeros(2, 3, 4);
    for other in [&wrong_shape, &bgr, &bgra] {
        assert!(horizontal_ref
            .gradient_magnitude_owned(other.as_ref().expect("invalid component"))
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(bgra);
    drop(bgr);
    drop(wrong_shape);
    drop(aliased_zero);
    drop(zero);
    drop(magnitude);
    drop(vertical);
    drop(horizontal);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native matrix into a Rust byte vector for exact assertions.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
