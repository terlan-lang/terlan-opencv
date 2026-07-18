//! Direct camera-undistortion tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves deterministic correction, source immutability, and result ownership.
#[test]
fn undistort_preserves_layout_and_owns_its_result() {
    assert_eq!(ffi::live_mat_count(), 0);
    let fixture = grayscale_fixture();
    let source = ffi::mat_from_u8(&fixture, 9, 9, 1);
    let source_ref = source.as_ref().expect("undistortion source");
    let identity = source_ref.undistort_owned(8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let corrected = source_ref.undistort_owned(8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0);
    let repeated = source_ref.undistort_owned(8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0);
    let optimal = source_ref
        .undistort_optimal_owned(8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0, 0.0, false);
    let optimal_repeated = source_ref
        .undistort_optimal_owned(8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0, 0.0, false);
    let optimal_centered = source_ref
        .undistort_optimal_owned(8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0, 1.0, true);
    let identity_ref = identity.as_ref().expect("identity correction");
    let corrected_ref = corrected.as_ref().expect("lens correction");
    assert_eq!(identity_ref.rows(), 9);
    assert_eq!(identity_ref.cols(), 9);
    assert_eq!(identity_ref.channels(), 1);
    assert_eq!(matrix_bytes(identity_ref), fixture);
    assert_eq!(matrix_bytes(source_ref), fixture);
    assert_eq!(
        matrix_bytes(corrected_ref),
        matrix_bytes(repeated.as_ref().expect("repeat"))
    );
    let optimal_ref = optimal.as_ref().expect("optimal correction");
    assert_eq!(optimal_ref.rows(), 9);
    assert_eq!(optimal_ref.cols(), 9);
    assert_eq!(optimal_ref.channels(), 1);
    assert_eq!(
        matrix_bytes(optimal_ref),
        matrix_bytes(optimal_repeated.as_ref().expect("optimal repeat"))
    );
    assert_eq!(optimal_centered.as_ref().expect("centered").rows(), 9);

    drop(source);
    assert_eq!(matrix_bytes(identity_ref), fixture);
    drop(optimal_centered);
    drop(optimal_repeated);
    drop(optimal);
    drop(repeated);
    drop(corrected);
    drop(identity);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves successful color handling and exhaustive numeric-bound rejection.
#[test]
fn undistort_rejects_invalid_camera_models_and_work() {
    let source = ffi::mat_zeros(9, 9, 1);
    let source_ref = source.as_ref().expect("undistortion source");
    for intrinsics in [
        [0.0, 8.0, 4.0, 4.0],
        [-1.0, 8.0, 4.0, 4.0],
        [8.0, 0.0, 4.0, 4.0],
        [8.0, -1.0, 4.0, 4.0],
        [f64::NAN, 8.0, 4.0, 4.0],
        [8.0, f64::INFINITY, 4.0, 4.0],
        [1_000_001.0, 8.0, 4.0, 4.0],
        [8.0, 1_000_001.0, 4.0, 4.0],
        [8.0, 8.0, -1.0, 4.0],
        [8.0, 8.0, 9.0, 4.0],
        [8.0, 8.0, 4.0, -1.0],
        [8.0, 8.0, 4.0, 9.0],
    ] {
        assert_rejected(source_ref, intrinsics, [0.0; 5]);
    }
    for index in 0..5 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 10.1, -10.1] {
            let mut distortion = [0.0; 5];
            distortion[index] = invalid;
            assert_rejected(source_ref, [8.0, 8.0, 4.0, 4.0], distortion);
        }
    }
    for alpha in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_optimal_rejected(source_ref, [8.0, 8.0, 4.0, 4.0], [0.0; 5], alpha);
    }
    let color = ffi::mat_zeros(2, 2, 3);
    let color_result = color
        .as_ref()
        .expect("color source")
        .undistort_owned(2.0, 2.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0);
    let color_optimal = color
        .as_ref()
        .expect("color source")
        .undistort_optimal_owned(2.0, 2.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, true);
    assert_eq!(
        color_result.as_ref().expect("color correction").channels(),
        3
    );
    assert_eq!(
        color_optimal
            .as_ref()
            .expect("optimal color correction")
            .channels(),
        3
    );
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized source")
        .undistort_owned(1000.0, 1000.0, 512.0, 512.0, 0.0, 0.0, 0.0, 0.0, 0.0,)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(oversized
        .as_ref()
        .expect("oversized source")
        .undistort_optimal_owned(1000.0, 1000.0, 512.0, 512.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, false,)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(oversized);
    drop(color_optimal);
    drop(color_result);
    drop(color);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves copied point correction is deterministic, bounded, and resource-free.
#[test]
fn undistort_points_copies_pixel_coordinates_and_rejects_invalid_inputs() {
    let points = [0.0, 0.0, 4.0, 4.0, 8.0, 8.0];
    let identity =
        ffi::undistort_points_owned(&points, 8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let identity = identity.as_ref().expect("identity points").as_slice();
    assert_eq!(identity.len(), points.len());
    for (actual, expected) in identity.iter().zip(points) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }
    let first =
        ffi::undistort_points_owned(&points, 8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0);
    let second =
        ffi::undistort_points_owned(&points, 8.0, 8.0, 4.0, 4.0, -0.2, 0.03, 0.01, -0.01, 0.0);
    assert_eq!(
        first.as_ref().expect("corrected points").as_slice(),
        second.as_ref().expect("repeated points").as_slice()
    );
    assert!(
        ffi::undistort_points_owned(&[], 8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,)
            .as_ref()
            .expect("empty points")
            .is_empty()
    );

    for invalid in [
        vec![1.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
        vec![0.0; 4097 * 2],
    ] {
        assert!(
            ffi::undistort_points_owned(&invalid, 8.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,)
                .is_null()
        );
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert!(
        ffi::undistort_points_owned(&points, 0.0, 8.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,)
            .is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(
        ffi::undistort_points_owned(&points, 8.0, 8.0, 4.0, 4.0, 10.1, 0.0, 0.0, 0.0, 0.0,)
            .is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Requires one camera model to be rejected as an invalid argument.
fn assert_rejected(image: &ffi::OpenCvMat, intrinsics: [f64; 4], distortion: [f64; 5]) {
    assert!(image
        .undistort_owned(
            intrinsics[0],
            intrinsics[1],
            intrinsics[2],
            intrinsics[3],
            distortion[0],
            distortion[1],
            distortion[2],
            distortion[3],
            distortion[4],
        )
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert_optimal_rejected(image, intrinsics, distortion, 0.5);
}

/// Requires one optimal-camera invocation to be rejected as invalid.
fn assert_optimal_rejected(
    image: &ffi::OpenCvMat,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
    alpha: f64,
) {
    assert!(image
        .undistort_optimal_owned(
            intrinsics[0],
            intrinsics[1],
            intrinsics[2],
            intrinsics[3],
            distortion[0],
            distortion[1],
            distortion[2],
            distortion[3],
            distortion[4],
            alpha,
            false,
        )
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Copies one native matrix into Rust ownership.
fn matrix_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}

/// Builds an asymmetric grayscale fixture that exposes remapping errors.
fn grayscale_fixture() -> Vec<u8> {
    (0..81)
        .map(|index| ((index * 37 + 11) % 256) as u8)
        .collect()
}
