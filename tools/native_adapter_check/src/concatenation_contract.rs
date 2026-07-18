//! Checked horizontal and vertical image concatenation contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs exact composition, aliasing, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let horizontal_first = matrix(helper, &[1, 2, 3, 4], 2, 2, "gray")?;
    let horizontal_second = matrix(helper, &[5, 6], 2, 1, "gray")?;
    let horizontal = concatenate(
        helper,
        "opencv.imgproc.concat_horizontal",
        &horizontal_first,
        &horizontal_second,
    )?;
    expect_int(helper.call("opencv.mat.rows", &[horizontal.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[horizontal.arg()])?, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[horizontal.arg()])?,
        &[1, 2, 5, 3, 4, 6],
    )?;

    let vertical_first = matrix(helper, &[1, 2], 1, 2, "gray")?;
    let vertical_second = matrix(helper, &[3, 4, 5, 6], 2, 2, "gray")?;
    let vertical = concatenate(
        helper,
        "opencv.imgproc.concat_vertical",
        &vertical_first,
        &vertical_second,
    )?;
    expect_int(helper.call("opencv.mat.rows", &[vertical.arg()])?, 3)?;
    expect_int(helper.call("opencv.mat.cols", &[vertical.arg()])?, 2)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[vertical.arg()])?,
        &[1, 2, 3, 4, 5, 6],
    )?;

    let bgr = matrix(helper, &[7, 8, 9], 1, 1, "bgr")?;
    let aliased_horizontal = concatenate(helper, "opencv.imgproc.concat_horizontal", &bgr, &bgr)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased_horizontal.arg()])?,
        &[7, 8, 9, 7, 8, 9],
    )?;
    let aliased_vertical = concatenate(helper, "opencv.imgproc.concat_vertical", &bgr, &bgr)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased_vertical.arg()])?,
        &[7, 8, 9, 7, 8, 9],
    )?;
    let bgra_first = matrix(helper, &[1, 2, 3, 4], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[5, 6, 7, 8], 1, 1, "bgra")?;
    let bgra_joined = concatenate(
        helper,
        "opencv.imgproc.concat_horizontal",
        &bgra_first,
        &bgra_second,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_joined.arg()])?,
        &[1, 2, 3, 4, 5, 6, 7, 8],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[horizontal_first.arg()])?,
        &[1, 2, 3, 4],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[vertical_second.arg()])?,
        &[3, 4, 5, 6],
    )?;

    let wrong_rows = matrix(helper, &[0], 1, 1, "gray")?;
    let wrong_columns = matrix(helper, &[0; 2], 2, 1, "gray")?;
    let wrong_layout = matrix(helper, &[0; 6], 2, 1, "bgr")?;
    expect_invalid(
        helper,
        "opencv.imgproc.concat_horizontal",
        &horizontal_first,
        &wrong_rows,
    )?;
    expect_invalid(
        helper,
        "opencv.imgproc.concat_vertical",
        &vertical_first,
        &wrong_columns,
    )?;
    expect_invalid(
        helper,
        "opencv.imgproc.concat_horizontal",
        &horizontal_first,
        &wrong_layout,
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.concat_horizontal",
            &[
                horizontal_first.arg(),
                horizontal_second.arg_with_owner("foreign-worker"),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.concat_vertical",
            &[
                vertical_first.arg(),
                vertical_second.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 2], 1, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.concat_vertical",
            &[vertical_first.arg(), stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_layout,
        &wrong_columns,
        &wrong_rows,
        &bgra_joined,
        &bgra_second,
        &bgra_first,
        &aliased_vertical,
        &aliased_horizontal,
        &bgr,
        &vertical,
        &vertical_second,
        &vertical_first,
        &horizontal,
        &horizontal_second,
        &horizontal_first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Concatenates two images and validates fresh matrix ownership.
fn concatenate(
    helper: &mut NativeHelper,
    operation: &str,
    first: &super::NativeHandle,
    second: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(operation, &[first.arg(), second.arg()])?,
        MAT_TYPE,
    )
}

/// Requires one incompatible concatenation to fail argument validation.
fn expect_invalid(
    helper: &mut NativeHelper,
    operation: &str,
    first: &super::NativeHandle,
    second: &super::NativeHandle,
) -> Result<(), String> {
    expect_error_code(
        helper.call(operation, &[first.arg(), second.arg()])?,
        "opencv.invalid_argument",
    )
}
