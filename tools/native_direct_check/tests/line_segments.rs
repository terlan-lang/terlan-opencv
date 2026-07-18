//! Direct bounded probabilistic Hough line-segment tests for the CXX bridge.

use terlan_opencv_native::ffi;

#[rustfmt::skip]
const LINE_MASK: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
];

/// Proves deterministic copied output, validation, immutability, and cleanup.
#[test]
fn line_segments_are_canonical_and_bounded() {
    assert_eq!(LINE_MASK.len(), 49);
    assert_eq!(ffi::live_mat_count(), 0);
    let mask = ffi::mat_from_u8(LINE_MASK, 7, 7, 1);
    let mask_ref = mask.as_ref().expect("line mask");
    assert_eq!(segments(mask_ref, 3, 4, 0), [1, 3, 5, 3]);
    assert_eq!(mat_bytes(mask_ref), LINE_MASK);

    let binary_bytes = LINE_MASK
        .iter()
        .map(|value| if *value == 0 { 0 } else { 255 })
        .collect::<Vec<_>>();
    let binary = ffi::mat_from_u8(&binary_bytes, 7, 7, 1);
    assert_eq!(
        segments(binary.as_ref().expect("binary line mask"), 3, 4, 0),
        segments(mask_ref, 3, 4, 0)
    );
    let empty = ffi::mat_zeros(7, 7, 1);
    assert!(segments(empty.as_ref().expect("empty mask"), 1, 0, 0).is_empty());

    for parameters in [
        (0, 0, 0),
        (8, 0, 0),
        (1, -1, 0),
        (1, 9, 0),
        (1, 0, -1),
        (1, 0, 9),
    ] {
        assert!(mask_ref
            .line_segments_owned(parameters.0, parameters.1, parameters.2)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let bgr = ffi::mat_zeros(7, 7, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&bgr, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid line mask")
            .line_segments_owned(1, 0, 0)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgr);
    drop(empty);
    drop(binary);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies flattened native line segments into Rust-owned integers.
fn segments(image: &ffi::OpenCvMat, votes: i64, length: i64, gap: i64) -> Vec<i64> {
    image
        .line_segments_owned(votes, length, gap)
        .as_ref()
        .expect("line segments")
        .as_slice()
        .to_vec()
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
