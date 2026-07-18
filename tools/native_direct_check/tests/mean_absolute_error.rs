//! Direct per-channel mean absolute error tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact channel ordering, aliases, rejection, preservation, and cleanup.
#[test]
fn mean_absolute_error_channels_are_copied_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);
    let first_bytes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
    let second_bytes = [10, 8, 25, 20, 60, 40, 65, 90, 70, 100, 80, 120];
    let first = ffi::mat_from_u8(&first_bytes, 2, 2, 3);
    let second = ffi::mat_from_u8(&second_bytes, 2, 2, 3);
    let first_ref = first.as_ref().expect("first BGR image");
    let second_ref = second.as_ref().expect("second BGR image");
    assert_eq!(
        first_ref
            .mean_absolute_error_channels_owned(second_ref)
            .as_ref()
            .expect("BGR channel errors")
            .as_slice(),
        [8.75, 15.5, 8.75]
    );
    assert_eq!(
        first_ref
            .mean_absolute_error_channels_owned(first_ref)
            .as_ref()
            .expect("aliased channel errors")
            .as_slice(),
        [0.0, 0.0, 0.0]
    );
    assert_eq!(
        first_ref
            .to_u8()
            .as_ref()
            .expect("unchanged first")
            .as_slice(),
        first_bytes
    );
    assert_eq!(
        second_ref
            .to_u8()
            .as_ref()
            .expect("unchanged second")
            .as_slice(),
        second_bytes
    );

    let gray_first = ffi::mat_from_u8(&[0, 10, 20, 30], 2, 2, 1);
    let gray_second = ffi::mat_from_u8(&[0, 20, 10, 50], 2, 2, 1);
    assert_eq!(
        gray_first
            .as_ref()
            .expect("first grayscale image")
            .mean_absolute_error_channels_owned(
                gray_second.as_ref().expect("second grayscale image")
            )
            .as_ref()
            .expect("grayscale error")
            .as_slice(),
        [10.0]
    );
    let bgra_first = ffi::mat_from_u8(&[0, 10, 20, 30, 40, 50, 60, 70], 1, 2, 4);
    let bgra_second = ffi::mat_from_u8(&[10, 0, 30, 20, 50, 40, 70, 60], 1, 2, 4);
    assert_eq!(
        bgra_first
            .as_ref()
            .expect("first BGRA image")
            .mean_absolute_error_channels_owned(bgra_second.as_ref().expect("second BGRA image"))
            .as_ref()
            .expect("BGRA channel errors")
            .as_slice(),
        [10.0, 10.0, 10.0, 10.0]
    );

    let wrong_shape = ffi::mat_zeros(1, 2, 3);
    assert!(first_ref
        .mean_absolute_error_channels_owned(wrong_shape.as_ref().expect("wrong shape"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let wrong_layout = ffi::mat_zeros(2, 2, 1);
    assert!(first_ref
        .mean_absolute_error_channels_owned(wrong_layout.as_ref().expect("wrong layout"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(wrong_layout);
    drop(wrong_shape);
    drop(bgra_second);
    drop(bgra_first);
    drop(gray_second);
    drop(gray_first);
    drop(second);
    drop(first);
    assert_eq!(ffi::live_mat_count(), 0);
}
