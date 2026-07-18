//! Direct copied pre-corner derivative-response tests.

use terlan_opencv_native::ffi;

mod support;

use support::CORNER_IMAGE;

/// Proves finite copied responses, source integrity, bounds, and cleanup.
#[test]
fn pre_corner_response_is_finite_nonzero_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source = ffi::mat_from_u8(CORNER_IMAGE, 7, 7, 1);
    let source_ref = source.as_ref().expect("pre-corner source");
    let response = source_ref.pre_corner_response_owned(3);
    let values = response.as_ref().expect("pre-corner response").as_slice();
    assert_eq!(values.len(), 49);
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| value.abs() > 0.0));
    assert_eq!(mat_bytes(source_ref), CORNER_IMAGE);
    assert_eq!(ffi::live_mat_count(), 1);

    let zero = ffi::mat_zeros(5, 5, 1);
    assert_eq!(
        zero.as_ref()
            .expect("zero source")
            .pre_corner_response_owned(3)
            .as_ref()
            .expect("zero pre-corner response")
            .as_slice(),
        [0.0; 25]
    );
    for aperture in [1, 9] {
        assert!(source_ref.pre_corner_response_owned(aperture).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    let color = ffi::mat_zeros(5, 5, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid pre-corner source")
            .pre_corner_response_owned(3)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(zero);
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
