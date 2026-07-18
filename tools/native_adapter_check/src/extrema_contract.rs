//! Element-wise image minimum and maximum contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const FIRST: &[u8] = &[0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220];
const SECOND: &[u8] = &[120, 80, 40, 0, 40, 80, 220, 180, 140, 100, 60, 20];

/// Runs exact selection, aliasing, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let first = matrix(helper, FIRST, 2, 2, "bgr")?;
    let second = matrix(helper, SECOND, 2, 2, "bgr")?;
    let minimum = binary_op(helper, "opencv.imgproc.minimum", &first, &second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[minimum.arg()])?,
        &[0, 40, 40, 0, 40, 80, 20, 60, 100, 100, 60, 20],
    )?;
    let maximum = binary_op(helper, "opencv.imgproc.maximum", &first, &second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[maximum.arg()])?,
        &[120, 80, 80, 120, 160, 200, 220, 180, 140, 140, 180, 220],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[first.arg()])?, FIRST)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[second.arg()])?, SECOND)?;

    let alias_minimum = binary_op(helper, "opencv.imgproc.minimum", &first, &first)?;
    let alias_maximum = binary_op(helper, "opencv.imgproc.maximum", &first, &first)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[alias_minimum.arg()])?,
        FIRST,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[alias_maximum.arg()])?,
        FIRST,
    )?;

    let gray_first = matrix(helper, &[250, 5], 1, 2, "gray")?;
    let gray_second = matrix(helper, &[10, 10], 1, 2, "gray")?;
    let gray_minimum = binary_op(helper, "opencv.imgproc.minimum", &gray_first, &gray_second)?;
    let gray_maximum = binary_op(helper, "opencv.imgproc.maximum", &gray_first, &gray_second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_minimum.arg()])?,
        &[10, 5],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_maximum.arg()])?,
        &[250, 10],
    )?;

    let bgra_first = matrix(helper, &[250, 1, 100, 0], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[10, 2, 200, 1], 1, 1, "bgra")?;
    let bgra_minimum = binary_op(helper, "opencv.imgproc.minimum", &bgra_first, &bgra_second)?;
    let bgra_maximum = binary_op(helper, "opencv.imgproc.maximum", &bgra_first, &bgra_second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_minimum.arg()])?,
        &[10, 1, 100, 0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_maximum.arg()])?,
        &[250, 2, 200, 1],
    )?;

    let wrong_shape = matrix(helper, &[0; 6], 1, 2, "bgr")?;
    let wrong_channels = matrix(helper, &[0; 4], 2, 2, "gray")?;
    for operation in ["opencv.imgproc.minimum", "opencv.imgproc.maximum"] {
        for other in [&wrong_shape, &wrong_channels] {
            expect_error_code(
                helper.call(operation, &[first.arg(), other.arg()])?,
                "opencv.invalid_argument",
            )?;
        }
        expect_error_code(
            helper.call(
                operation,
                &[first.arg(), second.arg_with_owner("foreign-worker")],
            )?,
            "cross_owner_handle",
        )?;
        expect_error_code(
            helper.call(
                operation,
                &[first.arg(), second.arg_with_type("other.Package.Resource")],
            )?,
            "handle_type_mismatch",
        )?;
    }
    let stale = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    for operation in ["opencv.imgproc.minimum", "opencv.imgproc.maximum"] {
        expect_error_code(
            helper.call(operation, &[first.arg(), stale.arg()])?,
            "stale_handle",
        )?;
    }

    for handle in [
        &wrong_channels,
        &wrong_shape,
        &bgra_maximum,
        &bgra_minimum,
        &bgra_second,
        &bgra_first,
        &gray_maximum,
        &gray_minimum,
        &gray_second,
        &gray_first,
        &alias_maximum,
        &alias_minimum,
        &maximum,
        &minimum,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Calls one generated binary image operation and requires a matrix handle.
fn binary_op(
    helper: &mut NativeHelper,
    operation: &str,
    first: &NativeHandle,
    second: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(operation, &[first.arg(), second.arg()])?,
        MAT_TYPE,
    )
}
