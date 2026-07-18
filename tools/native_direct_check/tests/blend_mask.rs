//! Direct bounded per-pixel byte-mask blending tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact interpolation, layouts, aliases, validation, and cleanup.
#[test]
fn blend_mask_is_bounded_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first = ffi::mat_from_u8(&[200, 200, 200], 1, 3, 1);
    let second = ffi::mat_from_u8(&[100, 100, 100], 1, 3, 1);
    let weights = ffi::mat_from_u8(&[0, 128, 255], 1, 3, 1);
    let first_ref = first.as_ref().expect("first grayscale image");
    let second_ref = second.as_ref().expect("second grayscale image");
    let weights_ref = weights.as_ref().expect("grayscale weights");
    let blended = first_ref.blend_mask_owned(second_ref, weights_ref);
    assert_eq!(
        image_bytes(blended.as_ref().expect("blended image")),
        [100, 150, 200]
    );
    assert_eq!(image_bytes(first_ref), [200, 200, 200]);
    assert_eq!(image_bytes(second_ref), [100, 100, 100]);
    assert_eq!(image_bytes(weights_ref), [0, 128, 255]);

    let bgr_first = ffi::mat_from_u8(&[200, 100, 0, 255, 0, 100], 1, 2, 3);
    let bgr_second = ffi::mat_from_u8(&[0, 100, 200, 0, 200, 100], 1, 2, 3);
    let bgr_weights = ffi::mat_from_u8(&[128, 0], 1, 2, 1);
    let bgr_blended = bgr_first
        .as_ref()
        .expect("first BGR image")
        .blend_mask_owned(
            bgr_second.as_ref().expect("second BGR image"),
            bgr_weights.as_ref().expect("BGR weights"),
        );
    assert_eq!(
        image_bytes(bgr_blended.as_ref().expect("blended BGR image")),
        [100, 100, 100, 0, 200, 100]
    );
    let bgra_first = ffi::mat_from_u8(&[10, 20, 30, 40], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[90, 80, 70, 60], 1, 1, 4);
    let bgra_weights = ffi::mat_from_u8(&[255], 1, 1, 1);
    let bgra_blended = bgra_first
        .as_ref()
        .expect("first BGRA image")
        .blend_mask_owned(
            bgra_second.as_ref().expect("second BGRA image"),
            bgra_weights.as_ref().expect("BGRA weights"),
        );
    assert_eq!(
        image_bytes(bgra_blended.as_ref().expect("blended BGRA image")),
        [10, 20, 30, 40]
    );
    let aliased = first_ref.blend_mask_owned(first_ref, weights_ref);
    assert_eq!(
        image_bytes(aliased.as_ref().expect("aliased blend")),
        [200; 3]
    );

    let wrong_shape = ffi::mat_zeros(2, 2, 1);
    let wrong_layout = ffi::mat_zeros(1, 3, 3);
    let color_weights = ffi::mat_zeros(1, 3, 3);
    for (invalid_second, invalid_weights) in [
        (&wrong_shape, &weights),
        (&wrong_layout, &weights),
        (&second, &wrong_shape),
        (&second, &color_weights),
    ] {
        assert!(first_ref
            .blend_mask_owned(
                invalid_second.as_ref().expect("invalid second image"),
                invalid_weights.as_ref().expect("invalid weights"),
            )
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let oversized_first = ffi::mat_zeros(1025, 1024, 1);
    let oversized_second = ffi::mat_zeros(1025, 1024, 1);
    let oversized_weights = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized_first
        .as_ref()
        .expect("oversized first")
        .blend_mask_owned(
            oversized_second.as_ref().expect("oversized second"),
            oversized_weights.as_ref().expect("oversized weights"),
        )
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(oversized_weights);
    drop(oversized_second);
    drop(oversized_first);
    drop(color_weights);
    drop(wrong_layout);
    drop(wrong_shape);
    drop(aliased);
    drop(bgra_blended);
    drop(bgra_weights);
    drop(bgra_second);
    drop(bgra_first);
    drop(bgr_blended);
    drop(bgr_weights);
    drop(bgr_second);
    drop(bgr_first);
    drop(blended);
    drop(weights);
    drop(second);
    drop(first);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native image into Rust bytes for deterministic assertions.
fn image_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("image bytes")
        .as_slice()
        .to_vec()
}
