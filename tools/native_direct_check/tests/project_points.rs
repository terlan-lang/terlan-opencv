//! Direct 3D camera-projection tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

const OBJECT_POINTS: [f64; 9] = [0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0];

/// Proves identity, translation, repeatability, and empty projection behavior.
#[test]
fn projects_bounded_object_points() {
    assert_points(
        &project(&OBJECT_POINTS, [0.0; 3], [0.0; 3]),
        &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
    );
    assert_points(
        &project(
            &OBJECT_POINTS,
            [0.0, 0.0, std::f64::consts::FRAC_PI_2],
            [0.0; 3],
        ),
        &[1.0, 1.0, 1.0, 2.0, 0.0, 1.0],
    );
    let first = project(&OBJECT_POINTS, [0.0; 3], [0.0, 0.0, 2.0]);
    let repeated = project(&OBJECT_POINTS, [0.0; 3], [0.0, 0.0, 2.0]);
    assert_points(&first, &[1.0, 1.0, 1.5, 1.0, 1.0, 1.5]);
    assert_points(&repeated, &copied(&first));
    assert!(copied(&project(&[], [0.0; 3], [0.0; 3])).is_empty());
}

/// Proves malformed, excessive, non-finite, and excessive-output rejection.
#[test]
fn rejects_invalid_object_point_projection() {
    for points in [vec![0.0], vec![0.0, 0.0], vec![0.0; 4], vec![0.0; 4097 * 3]] {
        assert_rejected(&points, [0.0; 3], [0.0; 3]);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut points = OBJECT_POINTS;
        points[0] = invalid;
        assert_rejected(&points, [0.0; 3], [0.0; 3]);
        assert_rejected(&OBJECT_POINTS, [invalid, 0.0, 0.0], [0.0; 3]);
        assert_rejected(&OBJECT_POINTS, [0.0; 3], [0.0, 0.0, invalid]);
    }
    assert_rejected(&[1_000_000.0, 0.0, 0.000001], [0.0; 3], [0.0; 3]);
}

/// Invokes projection with fixed camera and distortion policy.
fn project(
    points: &[f64],
    rotation: [f64; 3],
    translation: [f64; 3],
) -> cxx::UniquePtr<cxx::CxxVector<f64>> {
    ffi::project_points_owned(
        points,
        rotation[0],
        rotation[1],
        rotation[2],
        translation[0],
        translation[1],
        translation[2],
        2.0,
        2.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
}

/// Requires one direct projection to fail with invalid argument.
fn assert_rejected(points: &[f64], rotation: [f64; 3], translation: [f64; 3]) {
    let result = ffi::project_points_owned(
        points,
        rotation[0],
        rotation[1],
        rotation[2],
        translation[0],
        translation[1],
        translation[2],
        2.0,
        2.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    assert!(result.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Copies one non-null CXX vector.
fn copied(points: &cxx::UniquePtr<cxx::CxxVector<f64>>) -> Vec<f64> {
    points
        .as_ref()
        .expect("projected points")
        .iter()
        .copied()
        .collect()
}

/// Requires copied coordinates to match within projection tolerance.
fn assert_points(actual: &cxx::UniquePtr<cxx::CxxVector<f64>>, expected: &[f64]) {
    let actual = copied(actual);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }
}
