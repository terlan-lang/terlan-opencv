//! Immutable image-channel replacement contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, int_arg, matrix,
    NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs exact replacement, aliasing, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let bgr_bytes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
    let bgr = matrix(helper, &bgr_bytes, 2, 2, "bgr")?;
    let replacement = matrix(helper, &[9, 8, 7, 6], 2, 2, "gray")?;
    let replaced_red = replace(helper, &bgr, &replacement, 2)?;
    expect_image(
        helper,
        &replaced_red,
        2,
        2,
        3,
        &[0, 10, 9, 30, 40, 8, 60, 70, 7, 90, 100, 6],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[bgr.arg()])?, &bgr_bytes)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[replacement.arg()])?,
        &[9, 8, 7, 6],
    )?;

    let bgra = matrix(helper, &[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, "bgra")?;
    let alpha = matrix(helper, &[200, 201], 1, 2, "gray")?;
    let replaced_alpha = replace(helper, &bgra, &alpha, 3)?;
    expect_image(
        helper,
        &replaced_alpha,
        1,
        2,
        4,
        &[1, 2, 3, 200, 5, 6, 7, 201],
    )?;

    let gray = matrix(helper, &[1, 2, 3, 4], 2, 2, "gray")?;
    let aliased = replace(helper, &gray, &gray, 0)?;
    expect_image(helper, &aliased, 2, 2, 1, &[1, 2, 3, 4])?;

    for index in [-1, 3, i64::MAX] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.replace_channel",
                &[bgr.arg(), replacement.arg(), int_arg(index)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let wrong_shape = matrix(helper, &[0; 4], 1, 4, "gray")?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.replace_channel",
            &[bgr.arg(), wrong_shape.arg(), int_arg(0)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.replace_channel",
            &[bgr.arg(), bgr.arg(), int_arg(0)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.replace_channel",
            &[
                bgr.arg(),
                replacement.arg_with_owner("foreign-worker"),
                int_arg(0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.replace_channel",
            &[
                bgr.arg(),
                replacement.arg_with_type("other.Package.Resource"),
                int_arg(0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 4], 2, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.replace_channel",
            &[bgr.arg(), stale.arg(), int_arg(0)],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_shape,
        &aliased,
        &gray,
        &replaced_alpha,
        &alpha,
        &bgra,
        &replaced_red,
        &replacement,
        &bgr,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Replaces one channel through the generated immutable package operation.
fn replace(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    replacement: &NativeHandle,
    channel_index: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.replace_channel",
            &[image.arg(), replacement.arg(), int_arg(channel_index)],
        )?,
        MAT_TYPE,
    )
}

/// Requires one replaced image to have exact shape, layout, and bytes.
fn expect_image(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    rows: i64,
    cols: i64,
    channels: i64,
    bytes: &[u8],
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[image.arg()])?, rows)?;
    expect_int(helper.call("opencv.mat.cols", &[image.arg()])?, cols)?;
    expect_int(
        helper.call("opencv.mat.channels", &[image.arg()])?,
        channels,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, bytes)
}
