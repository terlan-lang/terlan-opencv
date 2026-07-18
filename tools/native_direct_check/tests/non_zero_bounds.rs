//! Direct copied nonzero grayscale-mask bounds tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact, empty, bounded, immutable, and rejected bounds analysis.
#[test]
fn non_zero_bounds_are_copied_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0];
    let mask = ffi::mat_from_u8(&mask_bytes, 4, 5, 1);
    let mask_ref = mask.as_ref().expect("bounds mask");
    assert_eq!(
        mask_ref
            .non_zero_bounds_owned()
            .as_ref()
            .expect("mask bounds")
            .as_slice(),
        [1, 1, 3, 2]
    );
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, 1);
    assert_eq!(
        single
            .as_ref()
            .expect("single pixel mask")
            .non_zero_bounds_owned()
            .as_ref()
            .expect("single pixel bounds")
            .as_slice(),
        [3, 2, 1, 1]
    );
    let empty = ffi::mat_zeros(3, 4, 1);
    assert!(empty
        .as_ref()
        .expect("empty mask")
        .non_zero_bounds_owned()
        .as_ref()
        .expect("empty bounds")
        .is_empty());

    let bgr = ffi::mat_zeros(4, 5, 3);
    let bgra = ffi::mat_zeros(4, 5, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid bounds mask")
            .non_zero_bounds_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(single);
    drop(mask);
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
