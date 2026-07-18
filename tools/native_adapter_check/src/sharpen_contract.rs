//! Bounded semantic image-sharpening contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, float_arg,
    matrix, NativeHandle, NativeHelper, MAT_TYPE,
};

const SOURCE: &[u8] = &[10, 20, 30];

/// Runs sharpening behavior, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, SOURCE, 1, 3, "gray")?;
    let sharpened = sharpen(helper, &source, 1.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[sharpened.arg()])?,
        &[0, 20, 40],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[source.arg()])?, SOURCE)?;
    let identity = sharpen(helper, &source, 0.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[identity.arg()])?, SOURCE)?;

    for (layout, channels) in [("bgr", 3_usize), ("bgra", 4_usize)] {
        let color = matrix(helper, &vec![10; channels], 1, 1, layout)?;
        let stable = sharpen(helper, &color, 16.0)?;
        expect_bytes(
            helper.call("opencv.mat.to_u8", &[stable.arg()])?,
            &vec![10; channels],
        )?;
        expect_unit(helper.call("opencv.mat.dispose", &[stable.arg()])?)?;
        expect_unit(helper.call("opencv.mat.dispose", &[color.arg()])?)?;
    }

    for amount in [-0.1, 16.1, f64::NAN, f64::INFINITY] {
        expect_error_code(
            helper.call("opencv.imgproc.sharpen", &[source.arg(), float_arg(amount)])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.sharpen",
            &[source.arg_with_owner("foreign-worker"), float_arg(1.0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.sharpen",
            &[
                source.arg_with_type("other.Package.Resource"),
                float_arg(1.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, SOURCE, 1, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.sharpen", &[stale.arg(), float_arg(1.0)])?,
        "stale_handle",
    )?;

    for handle in [&identity, &sharpened, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Applies one bounded sharpening operation and validates its resource type.
fn sharpen(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    amount: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.sharpen", &[image.arg(), float_arg(amount)])?,
        MAT_TYPE,
    )
}
