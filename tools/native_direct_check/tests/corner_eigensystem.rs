//! Direct copied corner eigensystem tests.

use terlan_opencv_native::ffi;

mod support;

use support::CORNER_IMAGE;

/// Proves tuple shape, finite values, source integrity, bounds, and cleanup.
#[test]
fn corner_eigensystem_is_finite_ordered_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source = ffi::mat_from_u8(CORNER_IMAGE, 7, 7, 1);
    let source_ref = source.as_ref().expect("corner eigensystem source");
    let response = source_ref.corner_eigensystem_owned(3, 3);
    let values = response.as_ref().expect("corner eigensystem").as_slice();
    assert_eigensystems(values, 49, true);
    assert_eq!(mat_bytes(source_ref), CORNER_IMAGE);
    assert_eq!(ffi::live_mat_count(), 1);

    let zero = ffi::mat_zeros(5, 5, 1);
    let zero_response = zero
        .as_ref()
        .expect("zero source")
        .corner_eigensystem_owned(3, 3);
    assert_eigensystems(
        zero_response
            .as_ref()
            .expect("zero corner eigensystem")
            .as_slice(),
        25,
        false,
    );

    for parameters in [(2, 3), (9, 3), (3, 1), (3, 9)] {
        assert!(source_ref
            .corner_eigensystem_owned(parameters.0, parameters.1)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(5, 5, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid eigensystem source")
            .corner_eigensystem_owned(3, 3)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(zero_response);
    drop(zero);
    drop(response);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires one flattened eigensystem map to satisfy its tuple invariants.
fn assert_eigensystems(values: &[f64], pixels: usize, require_response: bool) {
    assert_eq!(values.len(), pixels * 6);
    assert!(values.iter().all(|value| value.is_finite()));
    for tuple in values.chunks_exact(6) {
        assert!(tuple[0] >= tuple[1]);
        assert!(tuple[1] >= 0.0);
    }
    assert_eq!(
        values.chunks_exact(6).any(|tuple| tuple[0] > 0.0),
        require_response
    );
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
