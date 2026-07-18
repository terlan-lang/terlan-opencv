//! Direct checked image-tiling tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact gray, BGR, and BGRA tiling plus count rejection and cleanup.
#[test]
fn tiling_is_bounded_and_preserves_pixel_layout() {
    assert_eq!(ffi::live_mat_count(), 0);

    let gray = ffi::mat_from_u8(&[1, 2, 3, 4], 2, 2, 1);
    let gray_ref = gray.as_ref().expect("grayscale source");
    let gray_tiled = gray_ref.tile_owned(2, 3);
    let gray_tiled_ref = gray_tiled.as_ref().expect("tiled grayscale");
    assert_eq!(
        (
            gray_tiled_ref.rows(),
            gray_tiled_ref.cols(),
            gray_tiled_ref.channels()
        ),
        (4, 6, 1)
    );
    assert_eq!(
        gray_tiled_ref
            .to_u8()
            .as_ref()
            .expect("tiled grayscale bytes")
            .as_slice(),
        [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4,]
    );
    assert_eq!(
        gray_ref
            .to_u8()
            .as_ref()
            .expect("unchanged grayscale bytes")
            .as_slice(),
        [1, 2, 3, 4]
    );

    let bgr = ffi::mat_from_u8(&[5, 6, 7], 1, 1, 3);
    let bgr_tiled = bgr.as_ref().expect("BGR source").tile_owned(2, 2);
    assert_eq!(
        bgr_tiled
            .as_ref()
            .expect("tiled BGR")
            .to_u8()
            .as_ref()
            .expect("tiled BGR bytes")
            .as_slice(),
        [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7]
    );

    let bgra = ffi::mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, 4);
    let bgra_tiled = bgra.as_ref().expect("BGRA source").tile_owned(2, 1);
    assert_eq!(
        bgra_tiled
            .as_ref()
            .expect("tiled BGRA")
            .to_u8()
            .as_ref()
            .expect("tiled BGRA bytes")
            .as_slice(),
        [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    );

    for (vertical_count, horizontal_count) in [(0, 1), (1, 0), (-1, 1), (1, -1)] {
        assert!(gray_ref
            .tile_owned(vertical_count, horizontal_count)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert!(gray_ref.tile_owned(i64::MAX, 1).is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(bgra_tiled);
    drop(bgra);
    drop(bgr_tiled);
    drop(bgr);
    drop(gray_tiled);
    drop(gray);
    assert_eq!(ffi::live_mat_count(), 0);
}
