//! Direct signed point-to-contour distance tests for the CXX bridge.

use terlan_opencv_native::ffi;

const RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
];

/// Proves signed geometry, validation, source stability, and cleanup.
#[test]
fn contour_signed_distances_classify_bounded_points() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    let image = ffi::mat_from_u8(RECTANGLE, 5, 5, 1);
    let contours = image
        .as_ref()
        .expect("rectangle image")
        .find_external_contours_owned();
    let contours_ref = contours.as_ref().expect("rectangle contours");
    assert_close(&distances(contours_ref, 2.0, 2.0), &[1.0]);
    assert_close(&distances(contours_ref, 1.0, 2.0), &[0.0]);
    assert_close(&distances(contours_ref, 0.0, 0.0), &[-2.0_f64.sqrt()]);
    assert_eq!(bytes(image.as_ref().expect("rectangle image")), RECTANGLE);

    for (x, y) in [
        (-0.1, 0.0),
        (5.0, 0.0),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        assert!(contours_ref.signed_distances_owned(x, y).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(contours);
    drop(image);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies signed distances from one point to every retained contour.
fn distances(contours: &ffi::OpenCvContours, x: f64, y: f64) -> Vec<f64> {
    contours
        .signed_distances_owned(x, y)
        .as_ref()
        .expect("signed distances")
        .as_slice()
        .to_vec()
}

/// Requires copied floating-point values within native precision.
fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() <= 1.0e-9));
}

/// Copies one live native matrix into Rust-owned bytes.
fn bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
