//! Direct checked image-transpose tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact gray, BGR, and BGRA transposition with fresh ownership.
#[test]
fn transpose_swaps_axes_and_preserves_pixel_layout() {
    assert_eq!(ffi::live_mat_count(), 0);

    let gray = ffi::mat_from_u8(&[1, 2, 3, 4, 5, 6], 2, 3, 1);
    let gray_ref = gray.as_ref().expect("grayscale source");
    let gray_transposed = gray_ref.transpose_owned();
    let gray_transposed_ref = gray_transposed.as_ref().expect("transposed grayscale");
    assert_eq!(
        (
            gray_transposed_ref.rows(),
            gray_transposed_ref.cols(),
            gray_transposed_ref.channels()
        ),
        (3, 2, 1)
    );
    assert_eq!(
        gray_transposed_ref
            .to_u8()
            .as_ref()
            .expect("transposed grayscale bytes")
            .as_slice(),
        [1, 4, 2, 5, 3, 6]
    );
    assert_eq!(
        gray_ref
            .to_u8()
            .as_ref()
            .expect("unchanged grayscale bytes")
            .as_slice(),
        [1, 2, 3, 4, 5, 6]
    );

    let bgr = ffi::mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2, 2, 3);
    let bgr_transposed = bgr.as_ref().expect("BGR source").transpose_owned();
    assert_eq!(
        bgr_transposed
            .as_ref()
            .expect("transposed BGR")
            .to_u8()
            .as_ref()
            .expect("transposed BGR bytes")
            .as_slice(),
        [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]
    );

    let bgra = ffi::mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, 4);
    let bgra_transposed = bgra.as_ref().expect("BGRA source").transpose_owned();
    let bgra_transposed_ref = bgra_transposed.as_ref().expect("transposed BGRA");
    assert_eq!(
        (
            bgra_transposed_ref.rows(),
            bgra_transposed_ref.cols(),
            bgra_transposed_ref.channels()
        ),
        (2, 1, 4)
    );
    assert_eq!(
        bgra_transposed_ref
            .to_u8()
            .as_ref()
            .expect("transposed BGRA bytes")
            .as_slice(),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );

    drop(bgra_transposed);
    drop(bgra);
    drop(bgr_transposed);
    drop(bgr);
    drop(gray_transposed);
    drop(gray);
    assert_eq!(ffi::live_mat_count(), 0);
}
