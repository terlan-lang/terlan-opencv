//! Direct immutable contour-approximation tests for the CXX bridge.

use terlan_opencv_native::ffi;

#[rustfmt::skip]
const OCTAGON: &[u8] = &[
    0, 0, 0, 1, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 0,
];

/// Proves independent ownership, simplification, validation, and cleanup.
#[test]
fn contour_approximation_returns_a_fresh_validated_owner() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    let image = ffi::mat_from_u8(OCTAGON, 9, 9, 1);
    let contours = image
        .as_ref()
        .expect("octagon image")
        .find_external_contours_owned();
    let contours_ref = contours.as_ref().expect("octagon contours");
    let source_points = points(contours_ref);

    let exact = contours_ref.approximate_owned(0.0);
    assert_eq!(ffi::live_contour_analysis_count(), 2);
    assert_eq!(
        points(exact.as_ref().expect("exact approximation")),
        source_points
    );
    drop(exact);
    assert_eq!(ffi::live_contour_analysis_count(), 1);

    let reduced = contours_ref.approximate_owned(2.0);
    let reduced_points = points(reduced.as_ref().expect("reduced approximation"));
    assert!(reduced_points.len() >= 6);
    assert!(reduced_points.len() < source_points.len());
    assert_eq!(points(contours_ref), source_points);

    for epsilon in [-0.1, 12.0, f64::NAN, f64::INFINITY] {
        assert!(contours_ref.approximate_owned(epsilon).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(reduced);
    drop(contours);
    drop(image);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves closed degenerate contours retain their geometrically distinct ends.
#[test]
fn contour_approximation_preserves_line_endpoints() {
    let image = ffi::mat_from_u8(&[1, 1, 1], 1, 3, 1);
    let contours = image
        .as_ref()
        .expect("line image")
        .find_external_contours_owned();
    let approximation = contours
        .as_ref()
        .expect("line contours")
        .approximate_owned(2.0);

    assert_eq!(
        points(approximation.as_ref().expect("line approximation")),
        [0, 0, 2, 0]
    );
}

/// Copies flattened points from one live contour owner.
fn points(contours: &ffi::OpenCvContours) -> Vec<i64> {
    contours
        .points_owned()
        .as_ref()
        .expect("contour points")
        .as_slice()
        .to_vec()
}
