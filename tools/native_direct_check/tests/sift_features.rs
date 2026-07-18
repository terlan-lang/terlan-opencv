//! Direct owned SIFT feature tests for the generated CXX bridge.

mod support;

use support::{checkerboard_fixture, validate_feature_points};
use terlan_opencv_native::ffi;

/// Proves descriptor-row alignment, deterministic copies, and source independence.
#[test]
fn sift_features_preserve_keypoint_descriptor_ownership() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_sift_feature_count(), 0);
    let fixture = checkerboard_fixture();
    let image = ffi::mat_from_u8(&fixture, 160, 160, 1);
    let image_ref = image.as_ref().expect("SIFT fixture");
    let features = image_ref.sift_features_owned(0.04, 64);
    let features_ref = features.as_ref().expect("SIFT analysis");
    let count = features_ref.count();
    assert!((1..=64).contains(&count));
    assert_eq!(features_ref.descriptor_width(), 128);
    let points = floats(features_ref.keypoints_owned());
    validate_feature_points("SIFT", &points, count as usize);
    let descriptors = floats(features_ref.descriptors_owned());
    assert_eq!(descriptors.len(), count as usize * 128);
    assert!(descriptors.iter().all(|value| value.is_finite()));
    assert_eq!(floats(features_ref.descriptors_owned()), descriptors);
    assert_eq!(matrix_bytes(image_ref), fixture);

    drop(image);
    assert_eq!(features_ref.count(), count);
    assert_eq!(floats(features_ref.descriptors_owned()), descriptors);
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_sift_feature_count(), 1);
    drop(features);
    assert_eq!(ffi::live_sift_feature_count(), 0);
}

/// Proves empty output normalization and bounded invalid-input rejection.
#[test]
fn sift_features_reject_invalid_sources_and_limits() {
    let empty_image = ffi::mat_zeros(160, 160, 1);
    let empty_ref = empty_image.as_ref().expect("empty SIFT image");
    let empty = empty_ref.sift_features_owned(0.04, 8);
    let empty_features = empty.as_ref().expect("empty SIFT analysis");
    assert_eq!(empty_features.count(), 0);
    assert_eq!(empty_features.descriptor_width(), 128);
    assert!(floats(empty_features.keypoints_owned()).is_empty());
    assert!(floats(empty_features.descriptors_owned()).is_empty());
    for (threshold, maximum) in [
        (0.0, 8),
        (-0.1, 8),
        (1.1, 8),
        (f64::NAN, 8),
        (f64::INFINITY, 8),
        (0.04, 0),
        (0.04, 4097),
    ] {
        assert!(empty_ref.sift_features_owned(threshold, maximum).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(1, 1, 3);
    assert!(color
        .as_ref()
        .expect("color image")
        .sift_features_owned(0.04, 8)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized image")
        .sift_features_owned(0.04, 8)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(oversized);
    drop(color);
    drop(empty);
    drop(empty_image);
    assert_eq!(ffi::live_sift_feature_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native floating-point vector into Rust ownership.
fn floats(values: cxx::UniquePtr<cxx::CxxVector<f64>>) -> Vec<f64> {
    values.as_ref().expect("SIFT values").as_slice().to_vec()
}

/// Copies a live matrix into Rust-owned bytes.
fn matrix_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
