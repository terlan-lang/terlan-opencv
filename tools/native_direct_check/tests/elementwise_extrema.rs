//! Direct element-wise image extrema tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves gray, BGR, and BGRA selection, aliasing, rejection, and cleanup.
#[test]
fn image_extrema_are_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);

    let first_bytes = [0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220];
    let second_bytes = [120, 80, 40, 0, 40, 80, 220, 180, 140, 100, 60, 20];
    let first = ffi::mat_from_u8(&first_bytes, 2, 2, 3);
    let second = ffi::mat_from_u8(&second_bytes, 2, 2, 3);
    let first_ref = first.as_ref().expect("first BGR input");
    let second_ref = second.as_ref().expect("second BGR input");
    let minimum = first_ref.minimum_owned(second_ref);
    assert_eq!(
        mat_bytes(minimum.as_ref().expect("element-wise minimum")),
        [0, 40, 40, 0, 40, 80, 20, 60, 100, 100, 60, 20]
    );
    let maximum = first_ref.maximum_owned(second_ref);
    assert_eq!(
        mat_bytes(maximum.as_ref().expect("element-wise maximum")),
        [120, 80, 80, 120, 160, 200, 220, 180, 140, 140, 180, 220]
    );
    assert_eq!(mat_bytes(first_ref), first_bytes);
    assert_eq!(mat_bytes(second_ref), second_bytes);

    let alias_minimum = first_ref.minimum_owned(first_ref);
    let alias_maximum = first_ref.maximum_owned(first_ref);
    assert_eq!(
        mat_bytes(alias_minimum.as_ref().expect("aliased minimum")),
        first_bytes
    );
    assert_eq!(
        mat_bytes(alias_maximum.as_ref().expect("aliased maximum")),
        first_bytes
    );

    let gray_first = ffi::mat_from_u8(&[250, 5], 1, 2, 1);
    let gray_second = ffi::mat_from_u8(&[10, 10], 1, 2, 1);
    assert_pair(
        gray_first.as_ref().expect("first gray input"),
        gray_second.as_ref().expect("second gray input"),
        &[10, 5],
        &[250, 10],
    );
    let bgra_first = ffi::mat_from_u8(&[250, 1, 100, 0], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[10, 2, 200, 1], 1, 1, 4);
    assert_pair(
        bgra_first.as_ref().expect("first BGRA input"),
        bgra_second.as_ref().expect("second BGRA input"),
        &[10, 1, 100, 0],
        &[250, 2, 200, 1],
    );

    let wrong_shape = ffi::mat_zeros(1, 2, 3);
    let wrong_channels = ffi::mat_zeros(2, 2, 1);
    for other in [&wrong_shape, &wrong_channels] {
        let other_ref = other.as_ref().expect("incompatible image");
        assert!(first_ref.minimum_owned(other_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(first_ref.maximum_owned(other_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(wrong_channels);
    drop(wrong_shape);
    drop(bgra_second);
    drop(bgra_first);
    drop(gray_second);
    drop(gray_first);
    drop(alias_maximum);
    drop(alias_minimum);
    drop(maximum);
    drop(minimum);
    drop(second);
    drop(first);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires exact minimum and maximum bytes for one compatible image pair.
fn assert_pair(first: &ffi::OpenCvMat, second: &ffi::OpenCvMat, min: &[u8], max: &[u8]) {
    assert_eq!(
        mat_bytes(first.minimum_owned(second).as_ref().expect("minimum")),
        min
    );
    assert_eq!(
        mat_bytes(first.maximum_owned(second).as_ref().expect("maximum")),
        max
    );
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
