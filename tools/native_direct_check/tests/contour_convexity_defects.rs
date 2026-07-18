//! Direct indexed contour convexity-defect tests for the CXX bridge.

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

/// Proves concave records, convex elimination, index rejection, and cleanup.
#[test]
fn contour_convexity_defects_are_bounded_geometric_records() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    let image = ffi::mat_from_u8(CONCAVE, 7, 7, 1);
    let contours = image
        .as_ref()
        .expect("concave image")
        .find_external_contours_owned();
    let contours_ref = contours.as_ref().expect("concave contours");
    validate(&defects(contours_ref, 0));

    let hulls = contours_ref.convex_hulls_owned();
    assert!(defects(hulls.as_ref().expect("convex hulls"), 0).is_empty());
    for index in [-1, 1] {
        assert!(contours_ref.convexity_defects_owned(index).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(hulls);
    drop(contours);
    drop(image);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies convexity-defect values for one contour index.
fn defects(contours: &ffi::OpenCvContours, index: i64) -> Vec<f64> {
    contours
        .convexity_defects_owned(index)
        .as_ref()
        .expect("convexity defects")
        .as_slice()
        .to_vec()
}

/// Requires finite seven-scalar records with positive bounded depth.
fn validate(values: &[f64]) {
    assert!(!values.is_empty());
    assert!(values.len().is_multiple_of(7));
    assert!(values.chunks_exact(7).all(|record| {
        record.iter().all(|value| value.is_finite())
            && record[..6]
                .iter()
                .all(|coordinate| (0.0..=6.0).contains(coordinate))
            && record[6] > 0.0
            && record[6] <= 6.0_f64.hypot(6.0)
    }));
}
