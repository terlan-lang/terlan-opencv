//! Direct immutable mask-selection tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves gray, BGR, and BGRA selection, aliasing, rejection, and cleanup.
#[test]
fn mask_selection_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);
    let true_bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let false_bytes = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112];
    let when_true = ffi::mat_from_u8(&true_bytes, 2, 2, 3);
    let when_false = ffi::mat_from_u8(&false_bytes, 2, 2, 3);
    let mask = ffi::mat_from_u8(&[255, 0, 1, 0], 2, 2, 1);
    let true_ref = when_true.as_ref().expect("true BGR branch");
    let false_ref = when_false.as_ref().expect("false BGR branch");
    let mask_ref = mask.as_ref().expect("selection mask");
    let selected = true_ref.select_mask_owned(false_ref, mask_ref);
    assert_eq!(
        mat_bytes(selected.as_ref().expect("selected BGR image")),
        [1, 2, 3, 104, 105, 106, 7, 8, 9, 110, 111, 112]
    );
    assert_eq!(mat_bytes(true_ref), true_bytes);
    assert_eq!(mat_bytes(false_ref), false_bytes);
    assert_eq!(mat_bytes(mask_ref), [255, 0, 1, 0]);

    let aliased = true_ref.select_mask_owned(true_ref, mask_ref);
    assert_eq!(
        mat_bytes(aliased.as_ref().expect("aliased branches")),
        true_bytes
    );
    let gray_true = ffi::mat_from_u8(&[0, 2], 1, 2, 1);
    let gray_false = ffi::mat_from_u8(&[101, 102], 1, 2, 1);
    let gray_true_ref = gray_true.as_ref().expect("true gray branch");
    let gray_selected = gray_true_ref.select_mask_owned(
        gray_false.as_ref().expect("false gray branch"),
        gray_true_ref,
    );
    assert_eq!(
        mat_bytes(gray_selected.as_ref().expect("gray selection")),
        [101, 2]
    );
    let bgra_true = ffi::mat_from_u8(&[1, 2, 3, 4], 1, 1, 4);
    let bgra_false = ffi::mat_from_u8(&[101, 102, 103, 104], 1, 1, 4);
    let bgra_mask = ffi::mat_from_u8(&[255], 1, 1, 1);
    let bgra_selected = bgra_true
        .as_ref()
        .expect("true BGRA branch")
        .select_mask_owned(
            bgra_false.as_ref().expect("false BGRA branch"),
            bgra_mask.as_ref().expect("BGRA mask"),
        );
    assert_eq!(
        mat_bytes(bgra_selected.as_ref().expect("BGRA selection")),
        [1, 2, 3, 4]
    );

    let wrong_shape_image = ffi::mat_zeros(1, 2, 3);
    let wrong_layout_image = ffi::mat_zeros(2, 2, 1);
    let wrong_shape_mask = ffi::mat_zeros(1, 2, 1);
    let color_mask = ffi::mat_zeros(2, 2, 3);
    for other in [&wrong_shape_image, &wrong_layout_image] {
        assert!(true_ref
            .select_mask_owned(other.as_ref().expect("invalid branch"), mask_ref)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid_mask in [&wrong_shape_mask, &color_mask] {
        assert!(true_ref
            .select_mask_owned(false_ref, invalid_mask.as_ref().expect("invalid mask"))
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(color_mask);
    drop(wrong_shape_mask);
    drop(wrong_layout_image);
    drop(wrong_shape_image);
    drop(bgra_selected);
    drop(bgra_mask);
    drop(bgra_false);
    drop(bgra_true);
    drop(gray_selected);
    drop(gray_false);
    drop(gray_true);
    drop(aliased);
    drop(selected);
    drop(mask);
    drop(when_false);
    drop(when_true);
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
