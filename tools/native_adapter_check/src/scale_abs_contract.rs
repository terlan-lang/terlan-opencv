//! Bounded per-channel scale-and-absolute-value contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, float_arg,
    matrix, NativeHandle, NativeHelper, MAT_TYPE,
};

const SOURCE: &[u8] = &[0, 10, 100, 200, 255];
const TRANSFORMED: &[u8] = &[20, 0, 180, 255, 255];

/// Runs scale/absolute-value behavior, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, SOURCE, 1, 5, "gray")?;
    let transformed = scale_abs(helper, &source, 2.0, -20.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[transformed.arg()])?,
        TRANSFORMED,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[source.arg()])?, SOURCE)?;

    let identity = scale_abs(helper, &source, 1.0, 0.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[identity.arg()])?, SOURCE)?;
    let negative = scale_abs(helper, &source, -2.0, 0.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[negative.arg()])?,
        &[0, 20, 200, 255, 255],
    )?;

    for (layout, channels) in [("bgr", 3_usize), ("bgra", 4_usize)] {
        let color = matrix(helper, &vec![255; channels], 1, 1, layout)?;
        let constant = scale_abs(helper, &color, 0.0, -10.0)?;
        expect_bytes(
            helper.call("opencv.mat.to_u8", &[constant.arg()])?,
            &vec![10; channels],
        )?;
        expect_unit(helper.call("opencv.mat.dispose", &[constant.arg()])?)?;
        expect_unit(helper.call("opencv.mat.dispose", &[color.arg()])?)?;
    }

    for (scale, offset) in [
        (65_536.0, 0.0),
        (-65_536.0, 0.0),
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (1.0, 65_536.0),
        (1.0, -65_536.0),
        (1.0, f64::NAN),
        (1.0, f64::NEG_INFINITY),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.scale_abs",
                &[source.arg(), float_arg(scale), float_arg(offset)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.scale_abs",
            &[
                source.arg_with_owner("foreign-worker"),
                float_arg(1.0),
                float_arg(0.0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.scale_abs",
            &[
                source.arg_with_type("other.Package.Resource"),
                float_arg(1.0),
                float_arg(0.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, SOURCE, 1, 5, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.scale_abs",
            &[stale.arg(), float_arg(1.0), float_arg(0.0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&negative, &identity, &transformed, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Applies one bounded scale/absolute-value operation and checks its type.
fn scale_abs(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    scale: f64,
    offset: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.scale_abs",
            &[image.arg(), float_arg(scale), float_arg(offset)],
        )?,
        MAT_TYPE,
    )
}
