//! Direct affine-invariant SIFT tests for the generated CXX bridge.

mod support;

use std::sync::Mutex;
use support::{checkerboard_fixture, validate_feature_points};
use terlan_opencv_native::ffi;

/// Serializes tests that assert process-global native ownership counters.
static LIVE_COUNT_TEST: Mutex<()> = Mutex::new(());

/// Proves bounded descriptors, deterministic truncation, matching, and ownership.
#[test]
fn affine_sift_reuses_canonical_sift_owners() {
    let _guard = LIVE_COUNT_TEST.lock().expect("live-count test lock");
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_sift_feature_count(), 0);
    let fixture = checkerboard_fixture();
    let image = ffi::mat_from_u8(&fixture, 160, 160, 1);
    let image_ref = image.as_ref().expect("affine SIFT fixture");
    let first = image_ref.affine_sift_features_owned(0.04, 64);
    let second = image_ref.affine_sift_features_owned(0.04, 64);
    let limited = image_ref.affine_sift_features_owned(0.04, 1);
    let first_ref = first.as_ref().expect("first affine SIFT analysis");
    let second_ref = second.as_ref().expect("second affine SIFT analysis");
    let count = first_ref.count();
    assert!((1..=64).contains(&count));
    assert_eq!(second_ref.count(), count);
    assert_eq!(first_ref.descriptor_width(), 128);
    let points = floats(first_ref.keypoints_owned());
    validate_feature_points("affine SIFT", &points, count as usize);
    let descriptors = floats(first_ref.descriptors_owned());
    assert_eq!(descriptors.len(), count as usize * 128);
    assert!(descriptors.iter().all(|value| value.is_finite()));
    assert_eq!(floats(second_ref.descriptors_owned()), descriptors);
    assert_eq!(limited.as_ref().expect("limited affine SIFT").count(), 1);
    let matches = first_ref.match_owned(first_ref, 64);
    assert!((1..=count).contains(&matches.as_ref().expect("self matches").count()));
    assert_eq!(matrix_bytes(image_ref), fixture);

    drop(image);
    assert_eq!(first_ref.count(), count);
    drop(matches);
    drop(limited);
    drop(second);
    drop(first);
    assert_eq!(ffi::live_sift_match_count(), 0);
    assert_eq!(ffi::live_sift_feature_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves empty normalization and bounded invalid-input rejection.
#[test]
fn affine_sift_rejects_invalid_sources_and_limits() {
    let _guard = LIVE_COUNT_TEST.lock().expect("live-count test lock");
    let empty_image = ffi::mat_zeros(160, 160, 1);
    let empty_ref = empty_image.as_ref().expect("empty affine SIFT image");
    let empty = empty_ref.affine_sift_features_owned(0.04, 8);
    let features = empty.as_ref().expect("empty affine SIFT analysis");
    assert_eq!(features.count(), 0);
    assert_eq!(features.descriptor_width(), 128);
    assert!(floats(features.keypoints_owned()).is_empty());
    assert!(floats(features.descriptors_owned()).is_empty());
    for (threshold, maximum) in [
        (0.0, 8),
        (-0.1, 8),
        (1.1, 8),
        (f64::NAN, 8),
        (f64::INFINITY, 8),
        (0.04, 0),
        (0.04, 4097),
    ] {
        assert!(empty_ref
            .affine_sift_features_owned(threshold, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(1, 1, 3);
    assert!(color
        .as_ref()
        .expect("color image")
        .affine_sift_features_owned(0.04, 8)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized image")
        .affine_sift_features_owned(0.04, 8)
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
    values
        .as_ref()
        .expect("affine SIFT values")
        .as_slice()
        .to_vec()
}

/// Copies one matrix into Rust-owned bytes.
fn matrix_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
