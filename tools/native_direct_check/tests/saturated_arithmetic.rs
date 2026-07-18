//! Direct saturated arithmetic tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves gray, BGR, and BGRA saturation, aliasing, rejection, and cleanup.
#[test]
fn saturated_arithmetic_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);

    let first_bytes = [0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220];
    let second_bytes = [120, 80, 40, 0, 40, 80, 220, 180, 140, 100, 60, 20];
    let first = ffi::mat_from_u8(&first_bytes, 2, 2, 3);
    let second = ffi::mat_from_u8(&second_bytes, 2, 2, 3);
    let first_ref = first.as_ref().expect("first BGR input");
    let second_ref = second.as_ref().expect("second BGR input");
    let added = first_ref.add_saturated_owned(second_ref);
    assert_eq!(
        mat_bytes(added.as_ref().expect("saturated addition")),
        [120, 120, 120, 120, 200, 255, 240, 240, 240, 240, 240, 240]
    );
    let subtracted = first_ref.subtract_saturated_owned(second_ref);
    assert_eq!(
        mat_bytes(subtracted.as_ref().expect("saturated subtraction")),
        [0, 0, 40, 120, 120, 120, 0, 0, 0, 40, 120, 200]
    );
    assert_eq!(mat_bytes(first_ref), first_bytes);
    assert_eq!(mat_bytes(second_ref), second_bytes);

    let doubled = first_ref.add_saturated_owned(first_ref);
    assert_eq!(
        mat_bytes(doubled.as_ref().expect("aliased saturated addition")),
        [0, 80, 160, 240, 255, 255, 40, 120, 200, 255, 255, 255]
    );
    let zeroed = first_ref.subtract_saturated_owned(first_ref);
    assert_eq!(
        mat_bytes(zeroed.as_ref().expect("aliased saturated subtraction")),
        [0; 12]
    );

    let gray_first = ffi::mat_from_u8(&[250, 5], 1, 2, 1);
    let gray_second = ffi::mat_from_u8(&[10, 10], 1, 2, 1);
    let gray_first_ref = gray_first.as_ref().expect("first gray input");
    let gray_second_ref = gray_second.as_ref().expect("second gray input");
    assert_eq!(
        mat_bytes(
            gray_first_ref
                .add_saturated_owned(gray_second_ref)
                .as_ref()
                .expect("gray addition"),
        ),
        [255, 15]
    );
    assert_eq!(
        mat_bytes(
            gray_first_ref
                .subtract_saturated_owned(gray_second_ref)
                .as_ref()
                .expect("gray subtraction"),
        ),
        [240, 0]
    );

    let bgra_first = ffi::mat_from_u8(&[250, 1, 100, 0], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[10, 2, 200, 1], 1, 1, 4);
    let bgra_first_ref = bgra_first.as_ref().expect("first BGRA input");
    let bgra_second_ref = bgra_second.as_ref().expect("second BGRA input");
    assert_eq!(
        mat_bytes(
            bgra_first_ref
                .add_saturated_owned(bgra_second_ref)
                .as_ref()
                .expect("BGRA addition"),
        ),
        [255, 3, 255, 1]
    );
    assert_eq!(
        mat_bytes(
            bgra_first_ref
                .subtract_saturated_owned(bgra_second_ref)
                .as_ref()
                .expect("BGRA subtraction"),
        ),
        [240, 0, 0, 0]
    );

    let wrong_shape = ffi::mat_zeros(1, 2, 3);
    let wrong_channels = ffi::mat_zeros(2, 2, 1);
    for other in [&wrong_shape, &wrong_channels] {
        let other_ref = other.as_ref().expect("incompatible image");
        assert!(first_ref.add_saturated_owned(other_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(first_ref.subtract_saturated_owned(other_ref).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(wrong_channels);
    drop(wrong_shape);
    drop(bgra_second);
    drop(bgra_first);
    drop(gray_second);
    drop(gray_first);
    drop(zeroed);
    drop(doubled);
    drop(subtracted);
    drop(added);
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
