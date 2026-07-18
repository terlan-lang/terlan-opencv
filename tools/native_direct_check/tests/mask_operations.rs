//! Direct semantic mask and copied-point tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves mask operations, copied nonzero coordinates, rejection, and cleanup.
#[test]
fn mask_operations_and_copied_points_work() {
    assert_eq!(ffi::live_mat_count(), 0);

    let image_bytes = [0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0];
    let image = ffi::mat_from_u8(&image_bytes, 2, 2, 3);
    let image_ref = image.as_ref().expect("source image");
    let mask_bytes = [255, 0, 0, 255];
    let mask = ffi::mat_from_u8(&mask_bytes, 2, 2, 1);
    let mask_ref = mask.as_ref().expect("image mask");
    let masked = image_ref.apply_mask_owned(mask_ref);
    assert_eq!(
        masked
            .as_ref()
            .expect("masked image")
            .to_u8()
            .as_ref()
            .expect("masked image bytes")
            .as_slice(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0]
    );

    let grayscale = ffi::mat_from_u8(&[0, 10, 20, 30], 2, 2, 1);
    let grayscale_ref = grayscale.as_ref().expect("grayscale mask");
    let aliased = grayscale_ref.apply_mask_owned(grayscale_ref);
    assert_eq!(
        aliased
            .as_ref()
            .expect("aliased masked image")
            .to_u8()
            .as_ref()
            .expect("aliased masked image bytes")
            .as_slice(),
        [0, 10, 20, 30]
    );
    let range_mask = grayscale_ref.in_range_owned(10, 20);
    let range_mask_ref = range_mask.as_ref().expect("inclusive range mask");
    assert_eq!(
        range_mask_ref
            .to_u8()
            .as_ref()
            .expect("inclusive range-mask bytes")
            .as_slice(),
        [0, 255, 255, 0]
    );
    assert_eq!(
        range_mask_ref
            .non_zero_points_owned()
            .as_ref()
            .expect("range-mask points")
            .as_slice(),
        [1, 0, 0, 1]
    );
    assert_eq!(
        mask_ref
            .non_zero_points_owned()
            .as_ref()
            .expect("image-mask points")
            .as_slice(),
        [0, 0, 1, 1]
    );
    let exact_mask = grayscale_ref.in_range_owned(20, 20);
    assert_eq!(
        exact_mask
            .as_ref()
            .expect("exact-value range mask")
            .to_u8()
            .as_ref()
            .expect("exact-value range-mask bytes")
            .as_slice(),
        [0, 0, 255, 0]
    );
    let mask_xor =
        range_mask_ref.mask_xor_owned(exact_mask.as_ref().expect("exact-value range mask"));
    assert_eq!(
        mask_xor
            .as_ref()
            .expect("exclusive mask")
            .to_u8()
            .as_ref()
            .expect("exclusive mask bytes")
            .as_slice(),
        [0, 255, 0, 0]
    );

    let mask_and = range_mask_ref.mask_and_owned(mask_ref);
    let mask_and_ref = mask_and.as_ref().expect("intersected mask");
    assert_eq!(
        mask_and_ref
            .to_u8()
            .as_ref()
            .expect("intersected mask bytes")
            .as_slice(),
        [0, 0, 0, 0]
    );
    assert!(mask_and_ref
        .non_zero_points_owned()
        .as_ref()
        .expect("empty mask points")
        .is_empty());
    let mask_or = range_mask_ref.mask_or_owned(mask_ref);
    assert_eq!(
        mask_or
            .as_ref()
            .expect("united mask")
            .to_u8()
            .as_ref()
            .expect("united mask bytes")
            .as_slice(),
        [255, 255, 255, 255]
    );
    let mask_not = range_mask_ref.mask_not_owned();
    assert_eq!(
        mask_not
            .as_ref()
            .expect("inverted mask")
            .to_u8()
            .as_ref()
            .expect("inverted mask bytes")
            .as_slice(),
        [255, 0, 0, 255]
    );
    let same_mask_and = range_mask_ref.mask_and_owned(range_mask_ref);
    let same_mask_or = range_mask_ref.mask_or_owned(range_mask_ref);
    for same in [&same_mask_and, &same_mask_or] {
        assert_eq!(
            same.as_ref()
                .expect("same-handle mask result")
                .to_u8()
                .as_ref()
                .expect("same-handle mask bytes")
                .as_slice(),
            [0, 255, 255, 0]
        );
    }
    let same_mask_xor = range_mask_ref.mask_xor_owned(range_mask_ref);
    assert_eq!(
        same_mask_xor
            .as_ref()
            .expect("same-handle exclusive mask")
            .to_u8()
            .as_ref()
            .expect("same-handle exclusive mask bytes")
            .as_slice(),
        [0, 0, 0, 0]
    );

    let range_selected = image_ref.apply_mask_owned(range_mask_ref);
    assert_eq!(
        range_selected
            .as_ref()
            .expect("range-selected image")
            .to_u8()
            .as_ref()
            .expect("range-selected image bytes")
            .as_slice(),
        [0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 0, 0]
    );
    assert_eq!(
        image_ref
            .to_u8()
            .as_ref()
            .expect("unchanged image bytes")
            .as_slice(),
        image_bytes
    );
    assert_eq!(
        mask_ref
            .to_u8()
            .as_ref()
            .expect("unchanged mask bytes")
            .as_slice(),
        mask_bytes
    );

    assert!(image_ref.in_range_owned(0, 255).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (lower, upper) in [(-1, 20), (10, 256), (21, 20)] {
        assert!(grayscale_ref.in_range_owned(lower, upper).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let wrong_shape = ffi::mat_zeros(1, 2, 1);
    let wrong_shape_ref = wrong_shape.as_ref().expect("wrong-shape mask");
    assert!(image_ref.apply_mask_owned(wrong_shape_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let color_mask = ffi::mat_zeros(2, 2, 3);
    let color_mask_ref = color_mask.as_ref().expect("color mask");
    assert!(image_ref.apply_mask_owned(color_mask_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(range_mask_ref.mask_and_owned(wrong_shape_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(range_mask_ref.mask_or_owned(color_mask_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(range_mask_ref.mask_xor_owned(wrong_shape_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(range_mask_ref.mask_xor_owned(color_mask_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(color_mask_ref.mask_not_owned().is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(color_mask_ref.non_zero_points_owned().is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized_mask = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized_mask
        .as_ref()
        .expect("oversized mask")
        .non_zero_points_owned()
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(oversized_mask);
    drop(color_mask);
    drop(wrong_shape);
    drop(range_selected);
    drop(same_mask_xor);
    drop(same_mask_or);
    drop(same_mask_and);
    drop(mask_not);
    drop(mask_or);
    drop(mask_and);
    drop(mask_xor);
    drop(exact_mask);
    drop(range_mask);
    drop(aliased);
    drop(grayscale);
    drop(masked);
    drop(mask);
    drop(image);
    assert_eq!(ffi::live_mat_count(), 0);
}
