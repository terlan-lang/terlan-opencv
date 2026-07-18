//! Checked image-channel extraction contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, int_arg, matrix,
    NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs channel-order, index rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let bgr_bytes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
    let bgr = matrix(helper, &bgr_bytes, 2, 2, "bgr")?;
    let blue = extract(helper, &bgr, 0)?;
    let green = extract(helper, &bgr, 1)?;
    let red = extract(helper, &bgr, 2)?;
    expect_channel(helper, &blue, &[0, 30, 60, 90], 2, 2)?;
    expect_channel(helper, &green, &[10, 40, 70, 100], 2, 2)?;
    expect_channel(helper, &red, &[20, 50, 80, 110], 2, 2)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[bgr.arg()])?, &bgr_bytes)?;

    let gray = matrix(helper, &[1, 2, 3, 4], 2, 2, "gray")?;
    let gray_channel = extract(helper, &gray, 0)?;
    expect_channel(helper, &gray_channel, &[1, 2, 3, 4], 2, 2)?;

    let bgra = matrix(helper, &[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, "bgra")?;
    let alpha = extract(helper, &bgra, 3)?;
    expect_channel(helper, &alpha, &[4, 8], 1, 2)?;

    for index in [-1, 3, i64::MAX] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.extract_channel",
                &[bgr.arg(), int_arg(index)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.extract_channel",
            &[bgr.arg_with_owner("foreign-worker"), int_arg(0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.extract_channel",
            &[bgr.arg_with_type("other.Package.Resource"), int_arg(0)],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.extract_channel", &[stale.arg(), int_arg(0)])?,
        "stale_handle",
    )?;

    for handle in [
        &alpha,
        &bgra,
        &gray_channel,
        &gray,
        &red,
        &green,
        &blue,
        &bgr,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Extracts one channel and validates fresh matrix ownership.
fn extract(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    channel_index: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.extract_channel",
            &[image.arg(), int_arg(channel_index)],
        )?,
        MAT_TYPE,
    )
}

/// Requires one extracted channel to expose exact grayscale shape and bytes.
fn expect_channel(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    bytes: &[u8],
    rows: i64,
    cols: i64,
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[image.arg()])?, rows)?;
    expect_int(helper.call("opencv.mat.cols", &[image.arg()])?, cols)?;
    expect_int(helper.call("opencv.mat.channels", &[image.arg()])?, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, bytes)
}
