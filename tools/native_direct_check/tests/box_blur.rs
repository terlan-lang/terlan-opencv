//! Direct normalized box-filter and stack-blur tests for the generated bridge.

use terlan_opencv_native::ffi;

/// Proves normalized output, supported layouts, rejection, and fresh ownership.
#[test]
fn box_blur_preserves_shape_layout_and_source() {
    assert_eq!(ffi::live_mat_count(), 0);
    let impulse_bytes = [0, 0, 0, 0, 90, 0, 0, 0, 0];
    let impulse = ffi::mat_from_u8(&impulse_bytes, 3, 3, 1);
    let impulse_ref = impulse.as_ref().expect("grayscale impulse");
    let blurred = impulse_ref.box_blur_owned(3, 3);
    let blurred_ref = blurred.as_ref().expect("box-filtered image");
    assert_eq!((blurred_ref.rows(), blurred_ref.cols()), (3, 3));
    assert_eq!(blurred_ref.channels(), 1);
    assert_eq!(
        blurred_ref
            .to_u8()
            .as_ref()
            .expect("box-filtered bytes")
            .as_slice(),
        [10; 9]
    );
    assert_eq!(
        impulse_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        impulse_bytes
    );
    let stack_identity = impulse_ref.stack_blur_owned(1, 1);
    assert_eq!(
        stack_identity
            .as_ref()
            .expect("stack-blur identity")
            .to_u8()
            .as_ref()
            .expect("stack-blur identity bytes")
            .as_slice(),
        impulse_bytes
    );

    let bgr = ffi::mat_zeros(3, 3, 3);
    let blurred_bgr = bgr.as_ref().expect("BGR image").box_blur_owned(1, 3);
    let stacked_bgr = bgr.as_ref().expect("BGR image").stack_blur_owned(3, 3);
    assert_eq!(blurred_bgr.as_ref().expect("blurred BGR").channels(), 3);
    let bgra = ffi::mat_zeros(3, 3, 4);
    let blurred_bgra = bgra.as_ref().expect("BGRA image").box_blur_owned(3, 1);
    let stacked_bgra = bgra.as_ref().expect("BGRA image").stack_blur_owned(3, 3);
    assert_eq!(blurred_bgra.as_ref().expect("blurred BGRA").channels(), 4);

    for (width, height) in [(0, 3), (2, 3), (3, 0), (3, 4), (257, 3), (3, 257)] {
        assert!(impulse_ref.box_blur_owned(width, height).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(impulse_ref.stack_blur_owned(width, height).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 9);

    drop(stacked_bgra);
    drop(blurred_bgra);
    drop(bgra);
    drop(stacked_bgr);
    drop(blurred_bgr);
    drop(bgr);
    drop(stack_identity);
    drop(blurred);
    drop(impulse);
    assert_eq!(ffi::live_mat_count(), 0);
}
