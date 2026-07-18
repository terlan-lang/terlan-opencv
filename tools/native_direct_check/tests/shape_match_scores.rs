//! Direct binary-mask Hu-invariant shape-match tests for the CXX bridge.

use terlan_opencv_native::ffi;

const FIRST_SHAPE: &[u8] = &[1, 0, 0, 0, 2, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0];
const EQUIVALENT_SHAPE: &[u8] = &[255, 0, 0, 0, 127, 0, 0, 0, 63, 31, 15, 0, 0, 0, 0, 0];
const DIFFERENT_SHAPE: &[u8] = &[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0];
const TRANSLATED_SHAPE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];

/// Proves fixed score order, binary normalization, validation, and cleanup.
#[test]
fn shape_match_scores_are_copied_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first = ffi::mat_from_u8(FIRST_SHAPE, 4, 4, 1);
    let first_ref = first.as_ref().expect("first shape");
    assert_eq!(scores(first_ref, first_ref), [0.0, 0.0, 0.0]);

    let equivalent = ffi::mat_from_u8(EQUIVALENT_SHAPE, 4, 4, 1);
    let equivalent_ref = equivalent.as_ref().expect("equivalent shape");
    assert_eq!(scores(first_ref, equivalent_ref), [0.0, 0.0, 0.0]);
    assert_eq!(mat_bytes(first_ref), FIRST_SHAPE);
    assert_eq!(mat_bytes(equivalent_ref), EQUIVALENT_SHAPE);

    let different = ffi::mat_from_u8(DIFFERENT_SHAPE, 4, 4, 1);
    let different_scores = scores(first_ref, different.as_ref().expect("different shape"));
    assert!(different_scores.iter().any(|score| *score > 1.0e-6));
    let translated = ffi::mat_from_u8(TRANSLATED_SHAPE, 6, 6, 1);
    validate_scores(&scores(
        first_ref,
        translated.as_ref().expect("translated shape"),
    ));

    let empty = ffi::mat_zeros(4, 4, 1);
    let empty_ref = empty.as_ref().expect("empty shape");
    for (left, right) in [(first_ref, empty_ref), (empty_ref, first_ref)] {
        assert!(left.shape_match_scores_owned(right).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let bgr = ffi::mat_zeros(4, 4, 3);
    let bgra = ffi::mat_zeros(4, 4, 4);
    let oversized = ffi::mat_from_u8(&vec![1; 1025 * 1024], 1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        let image_ref = image.as_ref().expect("invalid shape");
        for (left, right) in [(first_ref, image_ref), (image_ref, first_ref)] {
            assert!(left.shape_match_scores_owned(right).is_null());
            assert_eq!(ffi::take_last_failure(), 1);
        }
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(translated);
    drop(different);
    drop(equivalent);
    drop(first);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native shape comparison into a fixed score array.
fn scores(first: &ffi::OpenCvMat, second: &ffi::OpenCvMat) -> [f64; 3] {
    let values = first
        .shape_match_scores_owned(second)
        .as_ref()
        .expect("shape-match scores")
        .as_slice()
        .to_vec();
    let scores = values.try_into().expect("three shape-match scores");
    validate_scores(&scores);
    scores
}

/// Requires three finite nonnegative shape-match scores.
fn validate_scores(values: &[f64; 3]) {
    assert!(values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0));
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
