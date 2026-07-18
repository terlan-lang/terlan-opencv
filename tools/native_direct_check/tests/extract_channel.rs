//! Direct checked image-channel extraction tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves zero-based channel order, fresh ownership, rejection, and cleanup.
#[test]
fn channel_extraction_is_checked_and_grayscale() {
    assert_eq!(ffi::live_mat_count(), 0);

    let bgr_bytes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
    let bgr = ffi::mat_from_u8(&bgr_bytes, 2, 2, 3);
    let bgr_ref = bgr.as_ref().expect("BGR source");
    let blue = bgr_ref.extract_channel_owned(0);
    let green = bgr_ref.extract_channel_owned(1);
    let red = bgr_ref.extract_channel_owned(2);
    assert_channel(blue.as_ref().expect("blue channel"), &[0, 30, 60, 90], 2, 2);
    assert_channel(
        green.as_ref().expect("green channel"),
        &[10, 40, 70, 100],
        2,
        2,
    );
    assert_channel(red.as_ref().expect("red channel"), &[20, 50, 80, 110], 2, 2);
    assert_eq!(
        bgr_ref
            .to_u8()
            .as_ref()
            .expect("unchanged BGR bytes")
            .as_slice(),
        bgr_bytes
    );

    let gray = ffi::mat_from_u8(&[1, 2, 3, 4], 2, 2, 1);
    let gray_channel = gray
        .as_ref()
        .expect("grayscale source")
        .extract_channel_owned(0);
    assert_channel(
        gray_channel.as_ref().expect("grayscale channel"),
        &[1, 2, 3, 4],
        2,
        2,
    );

    let bgra = ffi::mat_from_u8(&[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, 4);
    let alpha = bgra.as_ref().expect("BGRA source").extract_channel_owned(3);
    assert_channel(alpha.as_ref().expect("alpha channel"), &[4, 8], 1, 2);

    for index in [-1, 3, i64::MAX] {
        assert!(bgr_ref.extract_channel_owned(index).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(alpha);
    drop(bgra);
    drop(gray_channel);
    drop(gray);
    drop(red);
    drop(green);
    drop(blue);
    drop(bgr);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires one extracted channel to have exact grayscale shape and bytes.
fn assert_channel(image: &ffi::OpenCvMat, bytes: &[u8], rows: i64, cols: i64) {
    assert_eq!(
        (image.rows(), image.cols(), image.channels()),
        (rows, cols, 1)
    );
    assert_eq!(
        image
            .to_u8()
            .as_ref()
            .expect("extracted channel bytes")
            .as_slice(),
        bytes
    );
}
