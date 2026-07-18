//! Bounded scaled per-element image multiplication contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, float_arg,
    matrix, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const FIRST: &[u8] = &[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44];
const SECOND: &[u8] = &[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48];

/// Runs exact products, saturation, aliasing, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST, 2, 2, "bgr")?;
    let second = matrix(helper, SECOND, 2, 2, "bgr")?;
    let fractional = multiply(helper, &first, &second, 0.25)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[fractional.arg()])?,
        &[0, 8, 24, 48, 80, 120, 168, 224, 255, 255, 255, 255],
    )?;
    let zero = multiply(helper, &first, &second, 0.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[zero.arg()])?, &[0; 12])?;
    let saturated = multiply(helper, &first, &second, 1.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[saturated.arg()])?,
        &[0, 32, 96, 192, 255, 255, 255, 255, 255, 255, 255, 255],
    )?;
    let aliased = multiply(helper, &first, &first, 0.25)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased.arg()])?,
        &[0, 4, 16, 36, 64, 100, 144, 196, 255, 255, 255, 255],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[first.arg()])?, FIRST)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[second.arg()])?, SECOND)?;

    let gray_first = matrix(helper, &[10, 20], 1, 2, "gray")?;
    let gray_second = matrix(helper, &[2, 3], 1, 2, "gray")?;
    let gray_product = multiply(helper, &gray_first, &gray_second, 0.5)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_product.arg()])?,
        &[10, 30],
    )?;
    let bgra_first = matrix(helper, &[2, 4, 8, 16], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[4, 8, 16, 32], 1, 1, "bgra")?;
    let bgra_product = multiply(helper, &bgra_first, &bgra_second, 0.5)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_product.arg()])?,
        &[4, 16, 64, 255],
    )?;

    for scale in [-0.0001, 1.0001, f64::NAN, f64::INFINITY] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.multiply_scaled",
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
                "opencv.imgproc.multiply_scaled",
                &[first.arg(), other.arg(), float_arg(1.0)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.multiply_scaled",
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
            "opencv.imgproc.multiply_scaled",
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
            "opencv.imgproc.multiply_scaled",
            &[first.arg(), stale.arg(), float_arg(1.0)],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_layout,
        &wrong_shape,
        &bgra_product,
        &bgra_second,
        &bgra_first,
        &gray_product,
        &gray_second,
        &gray_first,
        &aliased,
        &saturated,
        &zero,
        &fractional,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Multiplies two images through the generated bounded-scale operation.
fn multiply(
    helper: &mut NativeHelper,
    first: &NativeHandle,
    second: &NativeHandle,
    scale: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.multiply_scaled",
            &[first.arg(), second.arg(), float_arg(scale)],
        )?,
        MAT_TYPE,
    )
}
