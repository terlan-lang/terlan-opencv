//! Direct copied nonzero grayscale-mask centroid tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact, fractional, empty, bounded, immutable centroid analysis.
#[test]
fn non_zero_centroid_is_copied_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [0, 2, 0, 0, 0, 0, 9, 0, 0, 0, 0, 5];
    let mask = ffi::mat_from_u8(&mask_bytes, 3, 4, 1);
    let mask_ref = mask.as_ref().expect("centroid mask");
    assert_eq!(centroid(mask_ref), [2.0, 1.0]);
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let fractional = ffi::mat_from_u8(&[1, 0, 0, 0, 0, 2], 2, 3, 1);
    assert_eq!(
        centroid(fractional.as_ref().expect("fractional centroid mask")),
        [1.0, 0.5]
    );
    let single = ffi::mat_from_u8(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, 1);
    assert_eq!(
        centroid(single.as_ref().expect("single centroid mask")),
        [3.0, 2.0]
    );
    let empty = ffi::mat_zeros(3, 4, 1);
    assert!(empty
        .as_ref()
        .expect("empty centroid mask")
        .non_zero_centroid_owned()
        .as_ref()
        .expect("empty centroid")
        .is_empty());

    let bgr = ffi::mat_zeros(3, 4, 3);
    let bgra = ffi::mat_zeros(3, 4, 4);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for image in [&bgr, &bgra, &oversized] {
        assert!(image
            .as_ref()
            .expect("invalid centroid mask")
            .non_zero_centroid_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(bgra);
    drop(bgr);
    drop(empty);
    drop(single);
    drop(fractional);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native centroid into a fixed pair for exact assertions.
fn centroid(image: &ffi::OpenCvMat) -> [f64; 2] {
    let values = image
        .non_zero_centroid_owned()
        .as_ref()
        .expect("nonzero centroid")
        .as_slice()
        .to_vec();
    values.try_into().expect("two centroid coordinates")
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
