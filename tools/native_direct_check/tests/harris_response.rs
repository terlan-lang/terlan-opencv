//! Direct copied Harris corner-response tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves copied values, source integrity, bounds, and no result owner.
#[test]
fn copied_harris_response_is_finite_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [0; 25];
    let source = ffi::mat_from_u8(&source_bytes, 5, 5, 1);
    let source_ref = source.as_ref().expect("Harris source");
    let response = source_ref.harris_response_owned(3, 3, 0.04);
    let values = response.as_ref().expect("Harris response").as_slice();
    assert_eq!(values, [0.0; 25]);
    assert!(values.iter().all(|value| value.is_finite()));
    assert_eq!(mat_bytes(source_ref), source_bytes);
    assert_eq!(ffi::live_mat_count(), 1);

    for parameters in [
        (2, 3, 0.04),
        (7, 3, 0.04),
        (3, 1, 0.04),
        (3, 9, 0.04),
        (3, 3, 0.0),
        (3, 3, 0.26),
        (3, 3, f64::NAN),
        (3, 3, f64::INFINITY),
    ] {
        assert!(source_ref
            .harris_response_owned(parameters.0, parameters.1, parameters.2)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    let color = ffi::mat_zeros(5, 5, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid Harris source")
            .harris_response_owned(3, 3, 0.04)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(response);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native matrix into a Rust byte vector.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
