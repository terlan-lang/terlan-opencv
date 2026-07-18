//! Direct precise Euclidean mask distance-transform tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves precise distances, validation, copied output, and cleanup.
#[test]
fn euclidean_distance_transform_is_bounded_and_precise() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [
        0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 255, 5, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0,
    ];
    let mask = ffi::mat_from_u8(&mask_bytes, 5, 5, 1);
    let mask_ref = mask.as_ref().expect("Euclidean distance mask");
    assert_close(
        mask_ref
            .euclidean_distance_transform_values_owned()
            .as_ref()
            .expect("Euclidean distances")
            .as_slice(),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let diagonal = ffi::mat_from_u8(&[0, 255, 255, 255, 255, 255, 255, 255, 255], 3, 3, 1);
    assert_close(
        diagonal
            .as_ref()
            .expect("diagonal mask")
            .euclidean_distance_transform_values_owned()
            .as_ref()
            .expect("diagonal distances")
            .as_slice(),
        &[
            0.0,
            1.0,
            2.0,
            1.0,
            2.0_f64.sqrt(),
            5.0_f64.sqrt(),
            2.0,
            5.0_f64.sqrt(),
            8.0_f64.sqrt(),
        ],
    );
    let zero = ffi::mat_zeros(3, 3, 1);
    assert_close(
        zero.as_ref()
            .expect("zero mask")
            .euclidean_distance_transform_values_owned()
            .as_ref()
            .expect("zero distances")
            .as_slice(),
        &[0.0; 9],
    );

    let all_nonzero = ffi::mat_from_u8(&[255; 9], 3, 3, 1);
    let bgr = ffi::mat_zeros(3, 3, 3);
    let bgra = ffi::mat_zeros(3, 3, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&all_nonzero, &bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid Euclidean mask")
            .euclidean_distance_transform_values_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(all_nonzero);
    drop(zero);
    drop(diagonal);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native matrix into a Rust byte vector for exact assertions.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}

/// Requires native distances to match within bounded single-precision roundoff.
fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }
}
