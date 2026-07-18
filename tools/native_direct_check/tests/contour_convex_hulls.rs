//! Direct immutable contour convex-hull tests for the CXX bridge.

use terlan_opencv_native::ffi;

#[rustfmt::skip]
const CONCAVE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0,
];

/// Proves hull geometry, independent ownership, source stability, and cleanup.
#[test]
fn contour_convex_hulls_return_a_fresh_analyzed_owner() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    let image = ffi::mat_from_u8(CONCAVE, 7, 7, 1);
    let contours = image
        .as_ref()
        .expect("concave image")
        .find_external_contours_owned();
    let contours_ref = contours.as_ref().expect("concave contours");
    let source_points = points(contours_ref);
    let source_area = area(contours_ref, false);
    let source_convex_area = area(contours_ref, true);

    let hulls = contours_ref.convex_hulls_owned();
    let hulls_ref = hulls.as_ref().expect("contour hulls");
    assert_eq!(ffi::live_contour_analysis_count(), 2);
    assert!(points(hulls_ref).len() < source_points.len());
    assert!(source_convex_area > source_area);
    assert_eq!(area(hulls_ref, false), source_convex_area);
    assert_eq!(area(hulls_ref, true), source_convex_area);
    assert_eq!(points(contours_ref), source_points);

    drop(hulls);
    assert_eq!(ffi::live_contour_analysis_count(), 1);
    drop(contours);
    drop(image);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
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

/// Copies one ordinary or convex area from a live contour owner.
fn area(contours: &ffi::OpenCvContours, convex: bool) -> f64 {
    let values = if convex {
        contours.convex_areas_owned()
    } else {
        contours.areas_owned()
    };
    let values = values.as_ref().expect("contour area");
    assert_eq!(values.len(), 1, "unexpected contour area length");
    *values.get(0).expect("first contour area")
}
