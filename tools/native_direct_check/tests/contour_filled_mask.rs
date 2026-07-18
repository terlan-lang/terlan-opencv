//! Direct contour-to-mask rasterization tests for the CXX bridge.

use terlan_opencv_native::ffi;

const RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
];
const FILLED_RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0,
];
const OUTLINED_RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 6, 0, 6, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 0,
];
const MULTI_POINT: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Proves exact rasterization, independent ownership, validation, and cleanup.
#[test]
fn filled_contour_mask_is_a_fresh_source_sized_matrix() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    let image = ffi::mat_from_u8(RECTANGLE, 5, 5, 1);
    let contours = image
        .as_ref()
        .expect("rectangle image")
        .find_external_contours_owned();
    let contours_ref = contours.as_ref().expect("rectangle contours");
    let filled = contours_ref.filled_mask_owned(9);
    assert_eq!(ffi::live_mat_count(), 2);
    assert_eq!(
        bytes(filled.as_ref().expect("filled mask")),
        FILLED_RECTANGLE
    );
    assert_eq!(bytes(image.as_ref().expect("rectangle image")), RECTANGLE);
    let outlined = contours_ref.outline_mask_owned(6, 1);
    assert_eq!(
        bytes(outlined.as_ref().expect("outlined mask")),
        OUTLINED_RECTANGLE
    );

    let multi_image = ffi::mat_from_u8(MULTI_POINT, 3, 7, 1);
    let multi = multi_image
        .as_ref()
        .expect("multi-point image")
        .find_external_contours_owned();
    let multi_ref = multi.as_ref().expect("multi-point contours");
    assert_eq!(multi_ref.count(), 2);
    let all = multi_ref.filled_mask_owned(11);
    let selected_first = multi_ref.selected_filled_mask_owned(0, 11);
    let selected_second = multi_ref.selected_filled_mask_owned(1, 11);
    validate_selection(
        &bytes(all.as_ref().expect("complete multi-point mask")),
        &bytes(selected_first.as_ref().expect("first selected contour")),
        &bytes(selected_second.as_ref().expect("second selected contour")),
    );

    for value in [-1, 256] {
        assert!(contours_ref.filled_mask_owned(value).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (value, thickness) in [(-1, 1), (256, 1), (9, 0), (9, 256)] {
        assert!(contours_ref.outline_mask_owned(value, thickness).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (index, value) in [(-1, 9), (1, 9), (0, -1), (0, 256)] {
        assert!(contours_ref
            .selected_filled_mask_owned(index, value)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(contours);
    assert_eq!(
        bytes(filled.as_ref().expect("independent filled mask")),
        FILLED_RECTANGLE
    );
    drop(selected_second);
    drop(selected_first);
    drop(all);
    drop(multi);
    drop(multi_image);
    drop(outlined);
    drop(filled);
    drop(image);
    assert_eq!(ffi::live_contour_analysis_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires disjoint selected masks to reconstruct the complete mask.
fn validate_selection(all: &[u8], first: &[u8], second: &[u8]) {
    assert_eq!(all.len(), first.len());
    assert_eq!(all.len(), second.len());
    assert_eq!(first.iter().filter(|value| **value != 0).count(), 1);
    assert_eq!(second.iter().filter(|value| **value != 0).count(), 1);
    for (combined, (left, right)) in all.iter().zip(first.iter().zip(second)) {
        assert!(*left == 0 || *right == 0);
        assert_eq!(*combined, *left | *right);
    }
}

/// Copies one live native matrix into Rust-owned bytes.
fn bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
