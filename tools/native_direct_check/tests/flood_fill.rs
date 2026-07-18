//! Direct immutable exact-value flood-fill tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves connectivity, layouts, validation, immutability, and cleanup.
#[test]
fn flood_fill_clones_before_mutation() {
    assert_eq!(ffi::live_mat_count(), 0);
    let gray_bytes = [1, 0, 1, 0, 1, 0, 1, 0, 1];
    let gray = ffi::mat_from_u8(&gray_bytes, 3, 3, 1);
    let gray_ref = gray.as_ref().expect("gray flood-fill source");
    let four = gray_ref.flood_fill_owned(1, 1, 9, 4);
    assert_eq!(
        mat_bytes(four.as_ref().expect("four-connected fill")),
        [1, 0, 1, 0, 9, 0, 1, 0, 1]
    );
    let eight = gray_ref.flood_fill_owned(1, 1, 9, 8);
    assert_eq!(
        mat_bytes(eight.as_ref().expect("eight-connected fill")),
        [9, 0, 9, 0, 9, 0, 9, 0, 9]
    );
    assert_eq!(mat_bytes(gray_ref), gray_bytes);

    let bgr = ffi::mat_from_u8(&[1, 2, 3, 1, 2, 3], 1, 2, 3);
    let filled_bgr = bgr
        .as_ref()
        .expect("BGR source")
        .flood_fill_owned(0, 0, 7, 4);
    assert_eq!(
        mat_bytes(filled_bgr.as_ref().expect("BGR fill")),
        [7, 7, 7, 7, 7, 7]
    );
    let bgra = ffi::mat_from_u8(&[1, 2, 3, 4], 1, 1, 4);
    let filled_bgra = bgra
        .as_ref()
        .expect("BGRA source")
        .flood_fill_owned(0, 0, 8, 8);
    assert!(filled_bgra.is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    for (seed_x, seed_y, fill_value, connectivity) in [
        (-1, 0, 0, 4),
        (3, 0, 0, 4),
        (0, -1, 0, 4),
        (0, 3, 0, 4),
        (0, 0, -1, 4),
        (0, 0, 256, 4),
        (0, 0, 0, 5),
    ] {
        assert!(gray_ref
            .flood_fill_owned(seed_x, seed_y, fill_value, connectivity)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(filled_bgra);
    drop(bgra);
    drop(filled_bgr);
    drop(bgr);
    drop(eight);
    drop(four);
    drop(gray);
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
