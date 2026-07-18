//! Direct copied nonzero-mask enclosing-circle tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves circle geometry, empty handling, validation, and cleanup.
#[test]
fn non_zero_enclosing_circle_is_copied_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [0, 0, 0, 2, 0, 9, 0, 0, 0];
    let mask = ffi::mat_from_u8(&mask_bytes, 3, 3, 1);
    let mask_ref = mask.as_ref().expect("enclosing-circle mask");
    assert_circle(circle(mask_ref), 1.0, 1.0, 1.0, 1.001);
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let corners = ffi::mat_from_u8(&[1, 0, 1, 0, 0, 0, 1, 0, 1], 3, 3, 1);
    assert_circle(
        circle(corners.as_ref().expect("corner mask")),
        1.0,
        1.0,
        2.0_f64.sqrt(),
        2.0_f64.sqrt() + 0.001,
    );
    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, 1);
    assert_eq!(
        circle(single.as_ref().expect("single-pixel mask")),
        [3.0, 2.0, 0.0]
    );
    let empty = ffi::mat_zeros(3, 4, 1);
    assert!(empty
        .as_ref()
        .expect("empty circle mask")
        .non_zero_enclosing_circle_owned()
        .as_ref()
        .expect("empty enclosing circle")
        .is_empty());

    let bgr = ffi::mat_zeros(3, 4, 3);
    let bgra = ffi::mat_zeros(3, 4, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid circle mask")
            .non_zero_enclosing_circle_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(single);
    drop(corners);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native enclosing circle into a fixed triple.
fn circle(image: &ffi::OpenCvMat) -> [f64; 3] {
    let values = image
        .non_zero_enclosing_circle_owned()
        .as_ref()
        .expect("nonzero enclosing circle")
        .as_slice()
        .to_vec();
    values.try_into().expect("three circle values")
}

/// Requires copied circle coordinates and radius to remain within bounds.
fn assert_circle(
    actual: [f64; 3],
    expected_x: f64,
    expected_y: f64,
    minimum_radius: f64,
    maximum_radius: f64,
) {
    assert!((actual[0] - expected_x).abs() <= 1.0e-5);
    assert!((actual[1] - expected_y).abs() <= 1.0e-5);
    assert!(actual[2] >= minimum_radius && actual[2] <= maximum_radius);
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
