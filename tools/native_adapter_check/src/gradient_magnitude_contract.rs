//! Grayscale vector-component magnitude contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const HORIZONTAL: &[u8] = &[0, 3, 6, 255, 5, 8];
const VERTICAL: &[u8] = &[0, 4, 8, 255, 12, 15];

/// Runs exact magnitude, saturation, ownership, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let horizontal = matrix(helper, HORIZONTAL, 2, 3, "gray")?;
    let vertical = matrix(helper, VERTICAL, 2, 3, "gray")?;
    let magnitude = combine(helper, &horizontal, &vertical)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[magnitude.arg()])?,
        &[0, 5, 10, 255, 13, 17],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[horizontal.arg()])?,
        HORIZONTAL,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[vertical.arg()])?,
        VERTICAL,
    )?;

    let zero = matrix(helper, &[0; 6], 2, 3, "gray")?;
    let aliased_zero = combine(helper, &zero, &zero)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased_zero.arg()])?,
        &[0; 6],
    )?;
    let wrong_shape = matrix(helper, &[0; 4], 2, 2, "gray")?;
    let bgr = matrix(helper, &[0; 18], 2, 3, "bgr")?;
    let bgra = matrix(helper, &[0; 24], 2, 3, "bgra")?;
    for other in [&wrong_shape, &bgr, &bgra] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.gradient_magnitude",
                &[horizontal.arg(), other.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.gradient_magnitude",
            &[horizontal.arg(), vertical.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.gradient_magnitude",
            &[
                horizontal.arg(),
                vertical.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, VERTICAL, 2, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.gradient_magnitude",
            &[horizontal.arg(), stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &bgra,
        &bgr,
        &wrong_shape,
        &aliased_zero,
        &zero,
        &magnitude,
        &vertical,
        &horizontal,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Combines two component images through the generated magnitude operation.
fn combine(
    helper: &mut NativeHelper,
    horizontal: &NativeHandle,
    vertical: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.gradient_magnitude",
            &[horizontal.arg(), vertical.arg()],
        )?,
        MAT_TYPE,
    )
}
