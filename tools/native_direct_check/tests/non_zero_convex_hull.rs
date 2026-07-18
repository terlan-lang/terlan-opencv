//! Direct copied nonzero-mask convex-hull tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves hull geometry, canonical ordering, validation, and cleanup.
#[test]
fn non_zero_convex_hull_is_copied_and_canonical() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [1, 0, 0, 2, 0, 9, 0, 0, 3, 0, 0, 4];
    let mask = ffi::mat_from_u8(&mask_bytes, 3, 4, 1);
    let mask_ref = mask.as_ref().expect("convex-hull mask");
    assert_eq!(hull(mask_ref), [0, 0, 3, 0, 3, 2, 0, 2]);
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let line = ffi::mat_from_u8(&[0, 0, 0, 0, 2, 7, 0, 9], 2, 4, 1);
    assert_eq!(
        hull(line.as_ref().expect("collinear hull mask")),
        [0, 1, 3, 1]
    );
    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, 1);
    assert_eq!(hull(single.as_ref().expect("single hull mask")), [3, 2]);
    let empty = ffi::mat_zeros(3, 4, 1);
    assert!(hull(empty.as_ref().expect("empty hull mask")).is_empty());

    let bgr = ffi::mat_zeros(3, 4, 3);
    let bgra = ffi::mat_zeros(3, 4, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid hull mask")
            .non_zero_convex_hull_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(single);
    drop(line);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native convex hull into a Rust integer vector.
fn hull(image: &ffi::OpenCvMat) -> Vec<i64> {
    image
        .non_zero_convex_hull_owned()
        .as_ref()
        .expect("nonzero convex hull")
        .as_slice()
        .to_vec()
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
