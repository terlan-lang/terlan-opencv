//! Direct adaptive-threshold tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves methods, modes, parameter bounds, source integrity, and cleanup.
#[test]
fn adaptive_threshold_is_bounded_and_grayscale_only() {
    assert_eq!(ffi::live_mat_count(), 0);
    let gray = ffi::mat_zeros(3, 3, 1);
    let gray_ref = gray.as_ref().expect("zero grayscale image");
    let mean = gray_ref.adaptive_threshold_owned(200.0, 0, 0, 3, 0.0);
    let mean_ref = mean.as_ref().expect("mean adaptive threshold");
    assert_eq!(
        (mean_ref.rows(), mean_ref.cols(), mean_ref.channels()),
        (3, 3, 1)
    );
    assert_eq!(
        mean_ref
            .to_u8()
            .as_ref()
            .expect("mean threshold bytes")
            .as_slice(),
        [0; 9]
    );
    let gaussian_inverse = gray_ref.adaptive_threshold_owned(200.0, 1, 1, 3, 0.0);
    assert_eq!(
        gaussian_inverse
            .as_ref()
            .expect("Gaussian inverse adaptive threshold")
            .to_u8()
            .as_ref()
            .expect("Gaussian inverse bytes")
            .as_slice(),
        [200; 9]
    );
    let shifted = gray_ref.adaptive_threshold_owned(200.0, 0, 0, 3, 1.0);
    assert_eq!(
        shifted
            .as_ref()
            .expect("shifted adaptive threshold")
            .to_u8()
            .as_ref()
            .expect("shifted threshold bytes")
            .as_slice(),
        [200; 9]
    );
    assert_eq!(
        gray_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        [0; 9]
    );

    let color = ffi::mat_zeros(3, 3, 3);
    assert!(color
        .as_ref()
        .expect("zero BGR image")
        .adaptive_threshold_owned(200.0, 0, 0, 3, 0.0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (maximum, method, mode, block_size, constant) in [
        (200.0, -1, 0, 3, 0.0),
        (200.0, 2, 0, 3, 0.0),
        (200.0, 0, -1, 3, 0.0),
        (200.0, 0, 2, 3, 0.0),
        (200.0, 0, 0, 1, 0.0),
        (200.0, 0, 0, 2, 0.0),
        (200.0, 0, 0, 257, 0.0),
        (-0.1, 0, 0, 3, 0.0),
        (255.1, 0, 0, 3, 0.0),
        (f64::NAN, 0, 0, 3, 0.0),
        (f64::INFINITY, 0, 0, 3, 0.0),
        (200.0, 0, 0, 3, -255.1),
        (200.0, 0, 0, 3, 255.1),
        (200.0, 0, 0, 3, f64::NAN),
        (200.0, 0, 0, 3, f64::INFINITY),
    ] {
        assert!(gray_ref
            .adaptive_threshold_owned(maximum, method, mode, block_size, constant)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(color);
    drop(shifted);
    drop(gaussian_inverse);
    drop(mean);
    drop(gray);
    assert_eq!(ffi::live_mat_count(), 0);
}
