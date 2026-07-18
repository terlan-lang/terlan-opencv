//! Direct fixed-arity BGR and BGRA merge tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact reconstruction, aliasing, rejection, ownership, and cleanup.
#[test]
fn grayscale_channels_merge_into_checked_color_owners() {
    assert_eq!(ffi::live_mat_count(), 0);

    let blue = ffi::mat_from_u8(&[1, 2, 3, 4], 2, 2, 1);
    let green = ffi::mat_from_u8(&[10, 20, 30, 40], 2, 2, 1);
    let red = ffi::mat_from_u8(&[100, 110, 120, 130], 2, 2, 1);
    let alpha = ffi::mat_from_u8(&[200, 201, 202, 203], 2, 2, 1);
    let blue_ref = blue.as_ref().expect("blue channel");
    let green_ref = green.as_ref().expect("green channel");
    let red_ref = red.as_ref().expect("red channel");
    let alpha_ref = alpha.as_ref().expect("alpha channel");

    let bgr = blue_ref.merge_bgr_owned(green_ref, red_ref);
    let bgr_ref = bgr.as_ref().expect("merged BGR image");
    assert_eq!(
        (bgr_ref.rows(), bgr_ref.cols(), bgr_ref.channels()),
        (2, 2, 3)
    );
    assert_eq!(
        bgr_ref
            .to_u8()
            .as_ref()
            .expect("merged BGR bytes")
            .as_slice(),
        [1, 10, 100, 2, 20, 110, 3, 30, 120, 4, 40, 130]
    );

    let bgra = blue_ref.merge_bgra_owned(green_ref, red_ref, alpha_ref);
    let bgra_ref = bgra.as_ref().expect("merged BGRA image");
    assert_eq!(
        (bgra_ref.rows(), bgra_ref.cols(), bgra_ref.channels()),
        (2, 2, 4)
    );
    assert_eq!(
        bgra_ref
            .to_u8()
            .as_ref()
            .expect("merged BGRA bytes")
            .as_slice(),
        [1, 10, 100, 200, 2, 20, 110, 201, 3, 30, 120, 202, 4, 40, 130, 203,]
    );
    assert_eq!(
        green_ref
            .to_u8()
            .as_ref()
            .expect("unchanged green channel")
            .as_slice(),
        [10, 20, 30, 40]
    );

    let aliased = blue_ref.merge_bgr_owned(blue_ref, blue_ref);
    assert_eq!(
        aliased
            .as_ref()
            .expect("aliased BGR merge")
            .to_u8()
            .as_ref()
            .expect("aliased BGR bytes")
            .as_slice(),
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    );

    let wrong_shape = ffi::mat_zeros(1, 4, 1);
    assert!(blue_ref
        .merge_bgr_owned(green_ref, wrong_shape.as_ref().expect("wrong shape"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(blue_ref.merge_bgr_owned(bgr_ref, red_ref).is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(wrong_shape);
    drop(aliased);
    drop(bgra);
    drop(bgr);
    drop(alpha);
    drop(red);
    drop(green);
    drop(blue);
    assert_eq!(ffi::live_mat_count(), 0);
}
