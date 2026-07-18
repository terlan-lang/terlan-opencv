//! Saturated unsigned-byte image arithmetic contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs exact saturation, aliasing, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let first_bytes = [0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220];
    let second_bytes = [120, 80, 40, 0, 40, 80, 220, 180, 140, 100, 60, 20];
    let first = matrix(helper, &first_bytes, 2, 2, "bgr")?;
    let second = matrix(helper, &second_bytes, 2, 2, "bgr")?;
    let added = binary_op(helper, "opencv.imgproc.add_saturated", &first, &second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[added.arg()])?,
        &[120, 120, 120, 120, 200, 255, 240, 240, 240, 240, 240, 240],
    )?;
    let subtracted = binary_op(helper, "opencv.imgproc.subtract_saturated", &first, &second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[subtracted.arg()])?,
        &[0, 0, 40, 120, 120, 120, 0, 0, 0, 40, 120, 200],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[first.arg()])?,
        &first_bytes,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[second.arg()])?,
        &second_bytes,
    )?;

    let doubled = binary_op(helper, "opencv.imgproc.add_saturated", &first, &first)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[doubled.arg()])?,
        &[0, 80, 160, 240, 255, 255, 40, 120, 200, 255, 255, 255],
    )?;
    let zeroed = binary_op(helper, "opencv.imgproc.subtract_saturated", &first, &first)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[zeroed.arg()])?, &[0; 12])?;

    let gray_first = matrix(helper, &[250, 5], 1, 2, "gray")?;
    let gray_second = matrix(helper, &[10, 10], 1, 2, "gray")?;
    let gray_added = binary_op(
        helper,
        "opencv.imgproc.add_saturated",
        &gray_first,
        &gray_second,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_added.arg()])?,
        &[255, 15],
    )?;
    let gray_subtracted = binary_op(
        helper,
        "opencv.imgproc.subtract_saturated",
        &gray_first,
        &gray_second,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_subtracted.arg()])?,
        &[240, 0],
    )?;

    let bgra_first = matrix(helper, &[250, 1, 100, 0], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[10, 2, 200, 1], 1, 1, "bgra")?;
    let bgra_added = binary_op(
        helper,
        "opencv.imgproc.add_saturated",
        &bgra_first,
        &bgra_second,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_added.arg()])?,
        &[255, 3, 255, 1],
    )?;
    let bgra_subtracted = binary_op(
        helper,
        "opencv.imgproc.subtract_saturated",
        &bgra_first,
        &bgra_second,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_subtracted.arg()])?,
        &[240, 0, 0, 0],
    )?;

    let wrong_shape = matrix(helper, &[0; 6], 1, 2, "bgr")?;
    let wrong_channels = matrix(helper, &[0; 4], 2, 2, "gray")?;
    for operation in [
        "opencv.imgproc.add_saturated",
        "opencv.imgproc.subtract_saturated",
    ] {
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
    for operation in [
        "opencv.imgproc.add_saturated",
        "opencv.imgproc.subtract_saturated",
    ] {
        expect_error_code(
            helper.call(operation, &[first.arg(), stale.arg()])?,
            "stale_handle",
        )?;
    }

    for handle in [
        &wrong_channels,
        &wrong_shape,
        &bgra_subtracted,
        &bgra_added,
        &bgra_second,
        &bgra_first,
        &gray_subtracted,
        &gray_added,
        &gray_second,
        &gray_first,
        &zeroed,
        &doubled,
        &subtracted,
        &added,
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
