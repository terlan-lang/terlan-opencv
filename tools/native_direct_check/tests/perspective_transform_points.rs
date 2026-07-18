//! Direct copied perspective point-transform tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// Proves identity, translation, repeatability, and empty copied results.
#[test]
fn perspective_points_transform_without_native_resource_state() {
    let points = [0.0, 0.0, 4.0, 4.0];
    assert_points(&transform(&points, IDENTITY), &points);
    let translation = [1.0, 0.0, 2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
    let first = transform(&points, translation);
    let repeated = transform(&points, translation);
    assert_points(&first, &[2.0, -1.0, 6.0, 3.0]);
    assert_eq!(first, repeated);
    assert!(transform(&[], IDENTITY).is_empty());
}

/// Proves malformed, excessive, and non-finite work is rejected consistently.
#[test]
fn perspective_points_reject_invalid_inputs_and_outputs() {
    let points = [0.0, 0.0, 4.0, 4.0];
    for invalid in [
        vec![1.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
        vec![0.0; 4097 * 2],
    ] {
        assert_rejected(&invalid, IDENTITY);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut transform = IDENTITY;
        transform[0] = invalid;
        assert_rejected(&points, transform);
    }
    assert_rejected(
        &[1.0, 0.0],
        [1.0, 0.0, 1_000_000.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
}

/// Invokes the direct bridge and copies the returned vector into Rust.
fn transform(points: &[f64], transform: [f64; 9]) -> Vec<f64> {
    ffi::perspective_transform_points_owned(
        points,
        transform[0],
        transform[1],
        transform[2],
        transform[3],
        transform[4],
        transform[5],
        transform[6],
        transform[7],
        transform[8],
    )
    .as_ref()
    .expect("perspective point transform")
    .as_slice()
    .to_vec()
}

/// Requires one direct call to report a finite invalid-argument failure.
fn assert_rejected(points: &[f64], transform: [f64; 9]) {
    assert!(ffi::perspective_transform_points_owned(
        points,
        transform[0],
        transform[1],
        transform[2],
        transform[3],
        transform[4],
        transform[5],
        transform[6],
        transform[7],
        transform[8],
    )
    .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Requires copied coordinates to match within native floating-point tolerance.
fn assert_points(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }
}
