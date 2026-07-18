//! Direct fresh-owner image-filter tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves bounded image filtering, supported layouts, rejection, and cleanup.
#[test]
fn bounded_median_blur_preserves_layout_and_ownership() {
    assert_eq!(ffi::live_mat_count(), 0);
    let impulse_bytes = [0, 0, 0, 0, 255, 0, 0, 0, 0];
    let impulse = ffi::mat_from_u8(&impulse_bytes, 3, 3, 1);
    let impulse_ref = impulse.as_ref().expect("grayscale impulse");
    let filtered = impulse_ref.median_blur_owned(3);
    let filtered_ref = filtered.as_ref().expect("filtered grayscale image");
    assert_eq!((filtered_ref.rows(), filtered_ref.cols()), (3, 3));
    assert_eq!(filtered_ref.channels(), 1);
    assert_eq!(
        filtered_ref
            .to_u8()
            .as_ref()
            .expect("filtered grayscale bytes")
            .as_slice(),
        [0; 9]
    );
    assert_eq!(
        impulse_ref
            .to_u8()
            .as_ref()
            .expect("unchanged impulse bytes")
            .as_slice(),
        impulse_bytes
    );

    let bgr = ffi::mat_zeros(3, 3, 3);
    let filtered_bgr = bgr.as_ref().expect("zero BGR image").median_blur_owned(3);
    assert_eq!(filtered_bgr.as_ref().expect("filtered BGR").channels(), 3);
    let bilateral_gray = filtered_ref.bilateral_filter_owned(3, 10.0, 10.0);
    assert_eq!(
        bilateral_gray
            .as_ref()
            .expect("bilateral grayscale image")
            .to_u8()
            .as_ref()
            .expect("bilateral grayscale bytes")
            .as_slice(),
        [0; 9]
    );
    let bilateral_bgr = bgr
        .as_ref()
        .expect("zero BGR image")
        .bilateral_filter_owned(3, 10.0, 10.0);
    assert_eq!(
        bilateral_bgr
            .as_ref()
            .expect("bilateral BGR image")
            .to_u8()
            .as_ref()
            .expect("bilateral BGR bytes")
            .as_slice(),
        [0; 27]
    );
    let bgra = ffi::mat_zeros(3, 3, 4);
    let filtered_bgra = bgra.as_ref().expect("zero BGRA image").median_blur_owned(3);
    assert_eq!(filtered_bgra.as_ref().expect("filtered BGRA").channels(), 4);

    for kernel_size in [-3, 0, 1, 2, 4, 257] {
        assert!(impulse_ref.median_blur_owned(kernel_size).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (diameter, sigma_color, sigma_space) in [
        (-1, 1.0, 1.0),
        (0, 1.0, 1.0),
        (2, 1.0, 1.0),
        (257, 1.0, 1.0),
        (3, 0.0, 1.0),
        (3, -1.0, 1.0),
        (3, 65_536.0, 1.0),
        (3, f64::NAN, 1.0),
        (3, f64::INFINITY, 1.0),
        (3, 1.0, 0.0),
        (3, 1.0, -1.0),
        (3, 1.0, 65_536.0),
        (3, 1.0, f64::NAN),
        (3, 1.0, f64::INFINITY),
    ] {
        assert!(impulse_ref
            .bilateral_filter_owned(diameter, sigma_color, sigma_space)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert!(bgra
        .as_ref()
        .expect("zero BGRA image")
        .bilateral_filter_owned(3, 10.0, 10.0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert_eq!(ffi::live_mat_count(), 8);

    drop(filtered_bgra);
    drop(bgra);
    drop(bilateral_bgr);
    drop(bilateral_gray);
    drop(filtered_bgr);
    drop(bgr);
    drop(filtered);
    drop(impulse);
    assert_eq!(ffi::live_mat_count(), 0);
}
