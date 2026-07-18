//! Bounded scaled per-element image division contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, float_arg,
    matrix, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const FIRST: &[u8] = &[0, 2, 8, 18, 32, 50, 72, 98, 128, 162, 200, 242];
const SECOND: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Runs exact quotients, saturation, zero-divisor, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST, 2, 2, "bgr")?;
    let second = matrix(helper, SECOND, 2, 2, "bgr")?;
    let fractional = divide(helper, &first, &second, 0.5)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[fractional.arg()])?,
        SECOND,
    )?;
    let zero = divide(helper, &first, &second, 0.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[zero.arg()])?, &[0; 12])?;
    let unscaled = divide(helper, &first, &second, 1.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[unscaled.arg()])?,
        &[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    )?;
    let saturated = divide(helper, &first, &second, 255.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[saturated.arg()])?,
        &[0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    )?;
    let aliased = divide(helper, &first, &first, 255.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased.arg()])?,
        &[0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[first.arg()])?, FIRST)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[second.arg()])?, SECOND)?;

    let gray_first = matrix(helper, &[0, 18], 1, 2, "gray")?;
    let gray_second = matrix(helper, &[0, 3], 1, 2, "gray")?;
    let gray_quotient = divide(helper, &gray_first, &gray_second, 0.5)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_quotient.arg()])?,
        &[0, 3],
    )?;
    let bgra_first = matrix(helper, &[2, 8, 18, 32], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[1, 2, 3, 4], 1, 1, "bgra")?;
    let bgra_quotient = divide(helper, &bgra_first, &bgra_second, 0.5)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_quotient.arg()])?,
        &[1, 2, 3, 4],
    )?;

    for scale in [-0.0001, 255.0001, f64::NAN, f64::INFINITY] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.divide_scaled",
                &[first.arg(), second.arg(), float_arg(scale)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let wrong_shape = matrix(helper, &[0; 6], 1, 2, "bgr")?;
    let wrong_layout = matrix(helper, &[0; 4], 2, 2, "gray")?;
    for other in [&wrong_shape, &wrong_layout] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.divide_scaled",
                &[first.arg(), other.arg(), float_arg(1.0)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.divide_scaled",
            &[
                first.arg(),
                second.arg_with_owner("foreign-worker"),
                float_arg(1.0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.divide_scaled",
            &[
                first.arg(),
                second.arg_with_type("other.Package.Resource"),
                float_arg(1.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, SECOND, 2, 2, "bgr")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.divide_scaled",
            &[first.arg(), stale.arg(), float_arg(1.0)],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_layout,
        &wrong_shape,
        &bgra_quotient,
        &bgra_second,
        &bgra_first,
        &gray_quotient,
        &gray_second,
        &gray_first,
        &aliased,
        &saturated,
        &unscaled,
        &zero,
        &fractional,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Divides two images through the generated bounded-scale operation.
fn divide(
    helper: &mut NativeHelper,
    first: &NativeHandle,
    second: &NativeHandle,
    scale: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.divide_scaled",
            &[first.arg(), second.arg(), float_arg(scale)],
        )?,
        MAT_TYPE,
    )
}
