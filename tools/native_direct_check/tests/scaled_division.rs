//! Direct scaled image division tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves scaling, saturation, zero-divisor behavior, validation, and cleanup.
#[test]
fn scaled_division_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first_bytes = [0, 2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242];
    let second_bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let first = ffi::mat_from_u8(&first_bytes, 2, 2, 3);
    let second = ffi::mat_from_u8(&second_bytes, 2, 2, 3);
    let first_ref = first.as_ref().expect("first BGR input");
    let second_ref = second.as_ref().expect("second BGR input");
    let fractional = first_ref.divide_scaled_owned(second_ref, 0.5);
    assert_eq!(
        mat_bytes(fractional.as_ref().expect("fractional quotient")),
        second_bytes
    );
    let zero = first_ref.divide_scaled_owned(second_ref, 0.0);
    assert_eq!(mat_bytes(zero.as_ref().expect("zero quotient")), [0; 12]);
    let unscaled = first_ref.divide_scaled_owned(second_ref, 1.0);
    assert_eq!(
        mat_bytes(unscaled.as_ref().expect("unscaled quotient")),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    );
    let saturated = first_ref.divide_scaled_owned(second_ref, 255.0);
    assert_eq!(
        mat_bytes(saturated.as_ref().expect("saturated quotient")),
        [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    );
    let aliased = first_ref.divide_scaled_owned(first_ref, 255.0);
    assert_eq!(
        mat_bytes(aliased.as_ref().expect("aliased quotient")),
        [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    );
    assert_eq!(mat_bytes(first_ref), first_bytes);
    assert_eq!(mat_bytes(second_ref), second_bytes);

    let gray_first = ffi::mat_from_u8(&[0, 18], 1, 2, 1);
    let gray_second = ffi::mat_from_u8(&[0, 3], 1, 2, 1);
    let gray_quotient = gray_first
        .as_ref()
        .expect("first gray input")
        .divide_scaled_owned(gray_second.as_ref().expect("second gray input"), 0.5);
    assert_eq!(
        mat_bytes(gray_quotient.as_ref().expect("gray quotient")),
        [0, 3]
    );
    let bgra_first = ffi::mat_from_u8(&[2, 8, 18, 32], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[1, 2, 3, 4], 1, 1, 4);
    let bgra_quotient = bgra_first
        .as_ref()
        .expect("first BGRA input")
        .divide_scaled_owned(bgra_second.as_ref().expect("second BGRA input"), 0.5);
    assert_eq!(
        mat_bytes(bgra_quotient.as_ref().expect("BGRA quotient")),
        [1, 2, 3, 4]
    );

    for scale in [-0.0001, 255.0001, f64::NAN, f64::INFINITY] {
        assert!(first_ref.divide_scaled_owned(second_ref, scale).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let wrong_shape = ffi::mat_zeros(1, 2, 3);
    let wrong_layout = ffi::mat_zeros(2, 2, 1);
    for other in [&wrong_shape, &wrong_layout] {
        assert!(first_ref
            .divide_scaled_owned(other.as_ref().expect("incompatible input"), 1.0)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(wrong_layout);
    drop(wrong_shape);
    drop(bgra_quotient);
    drop(bgra_second);
    drop(bgra_first);
    drop(gray_quotient);
    drop(gray_second);
    drop(gray_first);
    drop(aliased);
    drop(saturated);
    drop(unscaled);
    drop(zero);
    drop(fractional);
    drop(second);
    drop(first);
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
