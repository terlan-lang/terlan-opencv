//! Direct copied nonzero-mask minimum-enclosing-triangle tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves triangle geometry, degenerate normalization, validation, and cleanup.
#[test]
fn non_zero_enclosing_triangle_is_copied_and_normalized() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [0, 1, 0, 0, 9, 0, 2, 0, 3];
    let mask = ffi::mat_from_u8(&mask_bytes, 3, 3, 1);
    let mask_ref = mask.as_ref().expect("enclosing-triangle mask");
    assert_triangle(triangle(mask_ref), &[1.0, 0.0, 2.0, 2.0, 0.0, 2.0]);
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let line = ffi::mat_from_u8(&[0, 0, 0, 0, 2, 7, 0, 9], 2, 4, 1);
    assert_triangle(
        triangle(line.as_ref().expect("line triangle mask")),
        &[0.0, 1.0, 3.0, 1.0],
    );
    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, 1);
    assert_triangle(
        triangle(single.as_ref().expect("single triangle mask")),
        &[3.0, 2.0],
    );
    let empty = ffi::mat_zeros(3, 4, 1);
    assert!(triangle(empty.as_ref().expect("empty triangle mask")).is_empty());

    let bgr = ffi::mat_zeros(3, 3, 3);
    let bgra = ffi::mat_zeros(3, 3, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid enclosing-triangle mask")
            .non_zero_enclosing_triangle_owned()
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

/// Copies one native enclosing triangle into a Rust floating-point vector.
fn triangle(image: &ffi::OpenCvMat) -> Vec<f64> {
    image
        .non_zero_enclosing_triangle_owned()
        .as_ref()
        .expect("nonzero enclosing triangle")
        .as_slice()
        .to_vec()
}

/// Requires copied triangle coordinates to match within native precision.
fn assert_triangle(actual: Vec<f64>, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() <= 1.0e-4));
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
