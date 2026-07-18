//! Direct copied nonzero-mask minimum-area-box tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves box geometry, degenerate normalization, validation, and cleanup.
#[test]
fn non_zero_min_area_box_is_copied_and_normalized() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [1, 0, 0, 2, 0, 9, 0, 0, 3, 0, 0, 4];
    let mask = ffi::mat_from_u8(&mask_bytes, 3, 4, 1);
    let mask_ref = mask.as_ref().expect("minimum-area-box mask");
    assert_box(
        box_points(mask_ref),
        &[0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 2.0],
    );
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let diamond = ffi::mat_from_u8(
        &[
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 9, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
        ],
        5,
        5,
        1,
    );
    assert_box(
        box_points(diamond.as_ref().expect("diamond box mask")),
        &[2.0, 0.0, 4.0, 2.0, 2.0, 4.0, 0.0, 2.0],
    );
    let line = ffi::mat_from_u8(&[0, 0, 0, 0, 2, 7, 0, 9], 2, 4, 1);
    assert_box(
        box_points(line.as_ref().expect("line box mask")),
        &[0.0, 1.0, 3.0, 1.0],
    );
    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, 1);
    assert_box(
        box_points(single.as_ref().expect("single box mask")),
        &[3.0, 2.0],
    );
    let empty = ffi::mat_zeros(3, 4, 1);
    assert!(box_points(empty.as_ref().expect("empty box mask")).is_empty());

    let bgr = ffi::mat_zeros(3, 4, 3);
    let bgra = ffi::mat_zeros(3, 4, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid box mask")
            .non_zero_min_area_box_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(single);
    drop(line);
    drop(diamond);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native minimum-area box into a Rust floating-point vector.
fn box_points(image: &ffi::OpenCvMat) -> Vec<f64> {
    image
        .non_zero_min_area_box_owned()
        .as_ref()
        .expect("nonzero minimum-area box")
        .as_slice()
        .to_vec()
}

/// Requires copied box coordinates to match within native precision.
fn assert_box(actual: Vec<f64>, expected: &[f64]) {
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
