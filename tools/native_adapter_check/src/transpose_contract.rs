//! Checked image-transpose contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs exact layout, ownership, receiver rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let gray = matrix(helper, &[1, 2, 3, 4, 5, 6], 2, 3, "gray")?;
    let gray_transposed = transpose(helper, &gray)?;
    expect_shape(helper, &gray_transposed, 3, 2, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_transposed.arg()])?,
        &[1, 4, 2, 5, 3, 6],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray.arg()])?,
        &[1, 2, 3, 4, 5, 6],
    )?;

    let bgr = matrix(
        helper,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        2,
        2,
        "bgr",
    )?;
    let bgr_transposed = transpose(helper, &bgr)?;
    expect_shape(helper, &bgr_transposed, 2, 2, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgr_transposed.arg()])?,
        &[1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12],
    )?;

    let bgra = matrix(helper, &[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, "bgra")?;
    let bgra_transposed = transpose(helper, &bgra)?;
    expect_shape(helper, &bgra_transposed, 2, 1, 4)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_transposed.arg()])?,
        &[1, 2, 3, 4, 5, 6, 7, 8],
    )?;

    expect_error_code(
        helper.call(
            "opencv.imgproc.transpose",
            &[gray.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.transpose",
            &[gray.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.transpose", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &bgra_transposed,
        &bgra,
        &bgr_transposed,
        &bgr,
        &gray_transposed,
        &gray,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Transposes one image and validates fresh matrix ownership.
fn transpose(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.transpose", &[image.arg()])?,
        MAT_TYPE,
    )
}

/// Requires one matrix to expose the expected public shape and layout.
fn expect_shape(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    rows: i64,
    cols: i64,
    channels: i64,
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[image.arg()])?, rows)?;
    expect_int(helper.call("opencv.mat.cols", &[image.arg()])?, cols)?;
    expect_int(
        helper.call("opencv.mat.channels", &[image.arg()])?,
        channels,
    )
}
