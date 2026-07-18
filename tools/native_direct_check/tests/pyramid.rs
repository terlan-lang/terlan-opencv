//! Direct Gaussian image-pyramid tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves deterministic sizing, constant values, layouts, ownership, and cleanup.
#[test]
fn gaussian_pyramid_levels_are_checked_and_fresh() {
    assert_eq!(ffi::live_mat_count(), 0);
    let gray = ffi::mat_from_u8(&[64; 16], 4, 4, 1);
    let gray_ref = gray.as_ref().expect("constant grayscale image");
    let down = gray_ref.pyr_down_owned();
    let down_ref = down.as_ref().expect("downsampled grayscale image");
    assert_eq!(
        (down_ref.rows(), down_ref.cols(), down_ref.channels()),
        (2, 2, 1)
    );
    assert_eq!(
        down_ref
            .to_u8()
            .as_ref()
            .expect("downsampled grayscale bytes")
            .as_slice(),
        [64; 4]
    );
    let restored = down_ref.pyr_up_owned();
    let restored_ref = restored.as_ref().expect("upsampled grayscale image");
    assert_eq!(
        (
            restored_ref.rows(),
            restored_ref.cols(),
            restored_ref.channels()
        ),
        (4, 4, 1)
    );
    assert_eq!(
        restored_ref
            .to_u8()
            .as_ref()
            .expect("upsampled grayscale bytes")
            .as_slice(),
        [64; 16]
    );
    assert_eq!(
        gray_ref
            .to_u8()
            .as_ref()
            .expect("unchanged grayscale source")
            .as_slice(),
        [64; 16]
    );

    let odd_bgr = ffi::mat_zeros(3, 5, 3);
    let odd_down = odd_bgr.as_ref().expect("odd BGR image").pyr_down_owned();
    let odd_down_ref = odd_down.as_ref().expect("odd downsampled BGR image");
    assert_eq!(
        (
            odd_down_ref.rows(),
            odd_down_ref.cols(),
            odd_down_ref.channels()
        ),
        (2, 3, 3)
    );
    let bgra = ffi::mat_zeros(2, 2, 4);
    let bgra_up = bgra.as_ref().expect("BGRA image").pyr_up_owned();
    let bgra_up_ref = bgra_up.as_ref().expect("upsampled BGRA image");
    assert_eq!(
        (
            bgra_up_ref.rows(),
            bgra_up_ref.cols(),
            bgra_up_ref.channels()
        ),
        (4, 4, 4)
    );

    drop(bgra_up);
    drop(bgra);
    drop(odd_down);
    drop(odd_bgr);
    drop(restored);
    drop(down);
    drop(gray);
    assert_eq!(ffi::live_mat_count(), 0);
}
