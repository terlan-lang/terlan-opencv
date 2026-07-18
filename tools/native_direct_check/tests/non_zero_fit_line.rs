//! Direct copied nonzero-mask fitted-line tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves fitted-line geometry, normalization, validation, and cleanup.
#[test]
fn non_zero_fit_line_is_copied_and_canonical() {
    assert_eq!(ffi::live_mat_count(), 0);
    let horizontal_bytes = [2, 7, 9];
    let horizontal = ffi::mat_from_u8(&horizontal_bytes, 1, 3, 1);
    let horizontal_ref = horizontal.as_ref().expect("horizontal line mask");
    assert_line(line(horizontal_ref), [1.0, 0.0, 1.0, 0.0]);
    assert_eq!(mat_bytes(horizontal_ref), horizontal_bytes);

    let vertical = ffi::mat_from_u8(&[2, 0, 7, 0, 9, 0], 3, 2, 1);
    assert_line(
        line(vertical.as_ref().expect("vertical line mask")),
        [0.0, 1.0, 0.0, 1.0],
    );
    let diagonal = ffi::mat_from_u8(&[1, 0, 0, 0, 2, 0, 0, 0, 3], 3, 3, 1);
    assert_line(
        line(diagonal.as_ref().expect("diagonal line mask")),
        [2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0, 1.0, 1.0],
    );
    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 1, 0], 2, 3, 1);
    let empty = ffi::mat_zeros(2, 3, 1);
    for image in [&single, &empty] {
        assert!(line(image.as_ref().expect("insufficient line mask")).is_empty());
    }

    let bgr = ffi::mat_zeros(2, 3, 3);
    let bgra = ffi::mat_zeros(2, 3, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid line mask")
            .non_zero_fit_line_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(single);
    drop(diagonal);
    drop(vertical);
    drop(horizontal);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native fitted line into a Rust floating-point vector.
fn line(image: &ffi::OpenCvMat) -> Vec<f64> {
    let line = image.non_zero_fit_line_owned();
    line.as_ref()
        .unwrap_or_else(|| panic!("fitted-line failure code {}", ffi::take_last_failure()))
        .as_slice()
        .to_vec()
}

/// Requires one copied fitted line to match within native precision.
fn assert_line(actual: Vec<f64>, expected: [f64; 4]) {
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
