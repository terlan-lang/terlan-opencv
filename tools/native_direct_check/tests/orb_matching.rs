//! Direct cross-checked ORB matching tests for the generated CXX bridge.

mod support;

use support::{checkerboard_fixture, validate_binary_matches};
use terlan_opencv_native::ffi;

/// Proves bounded canonical matching and ownership independence.
#[test]
fn orb_matching_returns_aligned_owned_projections() {
    assert_eq!(ffi::live_orb_match_count(), 0);
    let image = ffi::mat_from_u8(&checkerboard_fixture(), 160, 160, 1);
    let image_ref = image.as_ref().expect("ORB match image");
    let first = image_ref.orb_features_owned(64);
    let second = image_ref.orb_features_owned(64);
    let first_ref = first.as_ref().expect("query features");
    let second_ref = second.as_ref().expect("train features");
    let matches = first_ref.match_owned(second_ref, 64);
    let matches_ref = matches.as_ref().expect("ORB matches");
    let count = matches_ref.count() as usize;
    assert!((1..=64).contains(&count));
    validate_matches(matches_ref, count, first_ref.count(), second_ref.count());

    let truncated = first_ref.match_owned(second_ref, 1);
    let truncated_ref = truncated.as_ref().expect("truncated ORB matches");
    assert_eq!(truncated_ref.count(), 1);
    validate_matches(truncated_ref, 1, first_ref.count(), second_ref.count());

    drop(first);
    drop(second);
    drop(image);
    validate_matches(matches_ref, count, 64, 64);
    drop(truncated);
    drop(matches);
    assert_eq!(ffi::live_orb_match_count(), 0);
    assert_eq!(ffi::live_orb_feature_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves empty matching and finite request limits.
#[test]
fn orb_matching_normalizes_empty_sets_and_rejects_bad_limits() {
    let image = ffi::mat_zeros(160, 160, 1);
    let features = image
        .as_ref()
        .expect("empty ORB image")
        .orb_features_owned(8);
    let features_ref = features.as_ref().expect("empty ORB features");
    let matches = features_ref.match_owned(features_ref, 8);
    let matches_ref = matches.as_ref().expect("empty ORB matches");
    assert_eq!(matches_ref.count(), 0);
    validate_matches(matches_ref, 0, 8, 8);
    for maximum in [0, 4097] {
        assert!(features_ref.match_owned(features_ref, maximum).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    drop(matches);
    drop(features);
    drop(image);
    assert_eq!(ffi::live_orb_match_count(), 0);
    assert_eq!(ffi::live_orb_feature_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Validates row alignment, cross-check uniqueness, and canonical ordering.
fn validate_matches(
    matches: &ffi::OpenCvOrbMatches,
    count: usize,
    query_count: i64,
    train_count: i64,
) {
    validate_binary_matches(
        "ORB",
        &ints(matches.indices_owned()),
        &ints(matches.distances_owned()),
        &floats(matches.coordinates_owned()),
        count,
        query_count,
        train_count,
        256,
    );
}

/// Copies one native integer vector.
fn ints(values: cxx::UniquePtr<cxx::CxxVector<i64>>) -> Vec<i64> {
    values
        .as_ref()
        .expect("ORB match integers")
        .as_slice()
        .to_vec()
}

/// Copies one native floating-point vector.
fn floats(values: cxx::UniquePtr<cxx::CxxVector<f64>>) -> Vec<f64> {
    values
        .as_ref()
        .expect("ORB match coordinates")
        .as_slice()
        .to_vec()
}
