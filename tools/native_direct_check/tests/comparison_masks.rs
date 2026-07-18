//! Direct typed grayscale comparison tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves every comparison mode, aliasing, validation, and cleanup.
#[test]
fn grayscale_comparisons_produce_checked_binary_masks() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first_bytes = [0, 10, 20, 30];
    let second_bytes = [0, 20, 10, 30];
    let first = ffi::mat_from_u8(&first_bytes, 2, 2, 1);
    let second = ffi::mat_from_u8(&second_bytes, 2, 2, 1);
    let first_ref = first.as_ref().expect("first grayscale input");
    let second_ref = second.as_ref().expect("second grayscale input");

    for (mode, expected) in [
        (0, [255, 0, 0, 255]),
        (1, [0, 255, 255, 0]),
        (2, [0, 255, 0, 0]),
        (3, [255, 255, 0, 255]),
        (4, [0, 0, 255, 0]),
        (5, [255, 0, 255, 255]),
    ] {
        let mask = first_ref.compare_mask_owned(second_ref, mode);
        let mask_ref = mask.as_ref().expect("comparison mask");
        assert_eq!(
            (mask_ref.rows(), mask_ref.cols(), mask_ref.channels()),
            (2, 2, 1)
        );
        assert_eq!(mat_bytes(mask_ref), expected);
    }
    assert_eq!(mat_bytes(first_ref), first_bytes);
    assert_eq!(mat_bytes(second_ref), second_bytes);

    let alias_equal = first_ref.compare_mask_owned(first_ref, 0);
    let alias_not_equal = first_ref.compare_mask_owned(first_ref, 1);
    assert_eq!(
        mat_bytes(alias_equal.as_ref().expect("alias equality")),
        [255; 4]
    );
    assert_eq!(
        mat_bytes(alias_not_equal.as_ref().expect("alias inequality")),
        [0; 4]
    );

    for mode in [-1, 6, i64::MAX] {
        assert!(first_ref.compare_mask_owned(second_ref, mode).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let wrong_shape = ffi::mat_zeros(1, 2, 1);
    let color = ffi::mat_zeros(2, 2, 3);
    for other in [&wrong_shape, &color] {
        assert!(first_ref
            .compare_mask_owned(other.as_ref().expect("invalid comparison input"), 0)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(color);
    drop(wrong_shape);
    drop(alias_not_equal);
    drop(alias_equal);
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
