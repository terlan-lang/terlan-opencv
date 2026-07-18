//! Direct immutable channel-replacement tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact BGR, BGRA, and aliased grayscale replacement and rejection.
#[test]
fn channel_replacement_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);

    let bgr_bytes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
    let bgr = ffi::mat_from_u8(&bgr_bytes, 2, 2, 3);
    let replacement = ffi::mat_from_u8(&[9, 8, 7, 6], 2, 2, 1);
    let bgr_ref = bgr.as_ref().expect("BGR source");
    let replacement_ref = replacement.as_ref().expect("replacement channel");
    let replaced_red = bgr_ref.replace_channel_owned(replacement_ref, 2);
    assert_image(
        replaced_red.as_ref().expect("replaced red channel"),
        2,
        2,
        3,
        &[0, 10, 9, 30, 40, 8, 60, 70, 7, 90, 100, 6],
    );
    assert_eq!(mat_bytes(bgr_ref), bgr_bytes);
    assert_eq!(mat_bytes(replacement_ref), [9, 8, 7, 6]);

    let bgra = ffi::mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, 4);
    let alpha = ffi::mat_from_u8(&[200, 201], 1, 2, 1);
    let replaced_alpha = bgra
        .as_ref()
        .expect("BGRA source")
        .replace_channel_owned(alpha.as_ref().expect("alpha replacement"), 3);
    assert_image(
        replaced_alpha.as_ref().expect("replaced alpha channel"),
        1,
        2,
        4,
        &[1, 2, 3, 200, 5, 6, 7, 201],
    );

    let gray = ffi::mat_from_u8(&[1, 2, 3, 4], 2, 2, 1);
    let gray_ref = gray.as_ref().expect("grayscale source");
    let aliased = gray_ref.replace_channel_owned(gray_ref, 0);
    assert_image(
        aliased.as_ref().expect("aliased grayscale replacement"),
        2,
        2,
        1,
        &[1, 2, 3, 4],
    );

    for index in [-1, 3, i64::MAX] {
        assert!(bgr_ref
            .replace_channel_owned(replacement_ref, index)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let wrong_shape = ffi::mat_zeros(1, 4, 1);
    assert!(bgr_ref
        .replace_channel_owned(wrong_shape.as_ref().expect("wrong shape"), 0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgr_ref.replace_channel_owned(bgr_ref, 0).is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(wrong_shape);
    drop(aliased);
    drop(gray);
    drop(replaced_alpha);
    drop(alpha);
    drop(bgra);
    drop(replaced_red);
    drop(replacement);
    drop(bgr);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires exact dimensions, channel count, and copied bytes.
fn assert_image(image: &ffi::OpenCvMat, rows: i64, cols: i64, channels: i64, bytes: &[u8]) {
    assert_eq!(
        (image.rows(), image.cols(), image.channels()),
        (rows, cols, channels)
    );
    assert_eq!(mat_bytes(image), bytes);
}

/// Copies one native matrix into a Rust byte vector for exact assertions.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
