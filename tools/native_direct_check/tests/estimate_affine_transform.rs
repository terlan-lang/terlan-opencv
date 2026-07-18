//! Direct affine-transform estimation tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

const SOURCE: [f64; 6] = [0.0, 0.0, 4.0, 0.0, 0.0, 4.0];

/// Proves identity, translation, repeatability, and copied coefficient access.
#[test]
fn estimates_affine_transform_without_resource_state() {
    let identity = estimate(&SOURCE, &SOURCE);
    assert_transform(&identity, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let destination = [2.0, -1.0, 6.0, -1.0, 2.0, 3.0];
    let first = estimate(&SOURCE, &destination);
    let repeated = estimate(&SOURCE, &destination);
    assert_transform(&first, &[1.0, 0.0, 2.0, 0.0, 1.0, -1.0]);
    assert_transform(&repeated, &coefficients(&first));
}

/// Proves malformed, degenerate, excessive, and non-finite points are rejected.
#[test]
fn rejects_invalid_affine_correspondences() {
    for (source, destination) in [
        (vec![], SOURCE.to_vec()),
        (vec![0.0, 0.0], SOURCE.to_vec()),
        (SOURCE[..4].to_vec(), SOURCE.to_vec()),
        ([SOURCE.as_slice(), &[5.0, 5.0]].concat(), SOURCE.to_vec()),
        (SOURCE.to_vec(), vec![]),
        (vec![0.0, 0.0, 0.0, 0.0, 0.0, 4.0], SOURCE.to_vec()),
        (vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0], SOURCE.to_vec()),
        (SOURCE.to_vec(), vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0]),
    ] {
        assert_rejected(&source, &destination);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut source = SOURCE;
        source[0] = invalid;
        assert_rejected(&source, &SOURCE);
        let mut destination = SOURCE;
        destination[5] = invalid;
        assert_rejected(&SOURCE, &destination);
    }
}

/// Invokes the direct CXX bridge.
fn estimate(source: &[f64], destination: &[f64]) -> cxx::UniquePtr<ffi::AffineTransform> {
    ffi::estimate_affine_transform_owned(source, destination)
}

/// Copies projected coefficients from one non-null native result.
fn coefficients(transform: &cxx::UniquePtr<ffi::AffineTransform>) -> [f64; 6] {
    let value = transform.as_ref().expect("affine transform");
    [
        value.m00_value(),
        value.m01_value(),
        value.m02_value(),
        value.m10_value(),
        value.m11_value(),
        value.m12_value(),
    ]
}

/// Requires one direct invocation to fail with invalid argument.
fn assert_rejected(source: &[f64], destination: &[f64]) {
    assert!(estimate(source, destination).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Requires projected coefficients to match within native tolerance.
fn assert_transform(actual: &cxx::UniquePtr<ffi::AffineTransform>, expected: &[f64; 6]) {
    for (actual, expected) in coefficients(actual).iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }
}
