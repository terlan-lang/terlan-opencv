//! Direct scaled image multiplication tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves scaling, saturation, aliasing, validation, and cleanup.
#[test]
fn scaled_multiplication_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first_bytes = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44];
    let second_bytes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48];
    let first = ffi::mat_from_u8(&first_bytes, 2, 2, 3);
    let second = ffi::mat_from_u8(&second_bytes, 2, 2, 3);
    let first_ref = first.as_ref().expect("first BGR input");
    let second_ref = second.as_ref().expect("second BGR input");
    let fractional = first_ref.multiply_scaled_owned(second_ref, 0.25);
    assert_eq!(
        mat_bytes(fractional.as_ref().expect("fractional product")),
        [0, 8, 24, 48, 80, 120, 168, 224, 255, 255, 255, 255]
    );
    let zero = first_ref.multiply_scaled_owned(second_ref, 0.0);
    assert_eq!(mat_bytes(zero.as_ref().expect("zero product")), [0; 12]);
    let saturated = first_ref.multiply_scaled_owned(second_ref, 1.0);
    assert_eq!(
        mat_bytes(saturated.as_ref().expect("saturated product")),
        [0, 32, 96, 192, 255, 255, 255, 255, 255, 255, 255, 255]
    );
    let aliased = first_ref.multiply_scaled_owned(first_ref, 0.25);
    assert_eq!(
        mat_bytes(aliased.as_ref().expect("aliased product")),
        [0, 4, 16, 36, 64, 100, 144, 196, 255, 255, 255, 255]
    );
    assert_eq!(mat_bytes(first_ref), first_bytes);
    assert_eq!(mat_bytes(second_ref), second_bytes);

    let gray_first = ffi::mat_from_u8(&[10, 20], 1, 2, 1);
    let gray_second = ffi::mat_from_u8(&[2, 3], 1, 2, 1);
    let gray_product = gray_first
        .as_ref()
        .expect("first gray input")
        .multiply_scaled_owned(gray_second.as_ref().expect("second gray input"), 0.5);
    assert_eq!(
        mat_bytes(gray_product.as_ref().expect("gray product")),
        [10, 30]
    );
    let bgra_first = ffi::mat_from_u8(&[2, 4, 8, 16], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[4, 8, 16, 32], 1, 1, 4);
    let bgra_product = bgra_first
        .as_ref()
        .expect("first BGRA input")
        .multiply_scaled_owned(bgra_second.as_ref().expect("second BGRA input"), 0.5);
    assert_eq!(
        mat_bytes(bgra_product.as_ref().expect("BGRA product")),
        [4, 16, 64, 255]
    );

    for scale in [-0.0001, 1.0001, f64::NAN, f64::INFINITY] {
        assert!(first_ref.multiply_scaled_owned(second_ref, scale).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let wrong_shape = ffi::mat_zeros(1, 2, 3);
    let wrong_layout = ffi::mat_zeros(2, 2, 1);
    for other in [&wrong_shape, &wrong_layout] {
        assert!(first_ref
            .multiply_scaled_owned(other.as_ref().expect("incompatible input"), 1.0)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(wrong_layout);
    drop(wrong_shape);
    drop(bgra_product);
    drop(bgra_second);
    drop(bgra_first);
    drop(gray_product);
    drop(gray_second);
    drop(gray_first);
    drop(aliased);
    drop(saturated);
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
