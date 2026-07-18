//! Direct cross-checked AKAZE matching tests for the generated CXX bridge.

mod support;

use support::{checkerboard_fixture, validate_binary_matches};
use terlan_opencv_native::ffi;

/// Proves bounded canonical matching and ownership independence.
#[test]
fn akaze_matching_returns_aligned_owned_projections() {
    assert_eq!(ffi::live_akaze_match_count(), 0);
    let image = ffi::mat_from_u8(&checkerboard_fixture(), 160, 160, 1);
    let image_ref = image.as_ref().expect("AKAZE match image");
    let first = image_ref.akaze_features_owned(0.001, 64);
    let second = image_ref.akaze_features_owned(0.001, 64);
    let first_ref = first.as_ref().expect("query features");
    let second_ref = second.as_ref().expect("train features");
    let matches = first_ref.match_owned(second_ref, 64);
    let matches_ref = matches.as_ref().expect("AKAZE matches");
    let count = matches_ref.count() as usize;
    assert!((1..=64).contains(&count));
    validate_matches(matches_ref, count, first_ref.count(), second_ref.count());

    let truncated = first_ref.match_owned(second_ref, 1);
    let truncated_ref = truncated.as_ref().expect("truncated AKAZE matches");
    assert_eq!(truncated_ref.count(), 1);
    validate_matches(truncated_ref, 1, first_ref.count(), second_ref.count());

    drop(first);
    drop(second);
    drop(image);
    validate_matches(matches_ref, count, 64, 64);
    drop(truncated);
    drop(matches);
    assert_eq!(ffi::live_akaze_match_count(), 0);
    assert_eq!(ffi::live_akaze_feature_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves empty matching and finite request limits.
#[test]
fn akaze_matching_normalizes_empty_sets_and_rejects_bad_limits() {
    let image = ffi::mat_zeros(160, 160, 1);
    let features = image
        .as_ref()
        .expect("empty AKAZE image")
        .akaze_features_owned(0.001, 8);
    let features_ref = features.as_ref().expect("empty AKAZE features");
    let matches = features_ref.match_owned(features_ref, 8);
    let matches_ref = matches.as_ref().expect("empty AKAZE matches");
    assert_eq!(matches_ref.count(), 0);
    validate_matches(matches_ref, 0, 8, 8);
    for maximum in [0, 4097] {
        assert!(features_ref.match_owned(features_ref, maximum).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    drop(matches);
    drop(features);
    drop(image);
    assert_eq!(ffi::live_akaze_match_count(), 0);
    assert_eq!(ffi::live_akaze_feature_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Validates copied AKAZE match projections.
fn validate_matches(
    matches: &ffi::OpenCvAkazeMatches,
    count: usize,
    query_count: i64,
    train_count: i64,
) {
    validate_binary_matches(
        "AKAZE",
        &ints(matches.indices_owned()),
        &ints(matches.distances_owned()),
        &floats(matches.coordinates_owned()),
        count,
        query_count,
        train_count,
        488,
    );
}

/// Copies one native integer vector.
fn ints(values: cxx::UniquePtr<cxx::CxxVector<i64>>) -> Vec<i64> {
    values
        .as_ref()
        .expect("AKAZE match integers")
        .as_slice()
        .to_vec()
}

/// Copies one native floating-point vector.
fn floats(values: cxx::UniquePtr<cxx::CxxVector<f64>>) -> Vec<f64> {
    values
        .as_ref()
        .expect("AKAZE match coordinates")
        .as_slice()
        .to_vec()
}
