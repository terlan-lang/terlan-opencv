//! Copied grayscale Harris corner-response contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, float_arg, int_arg,
    matrix, NativeHelper,
};

/// Runs value, bound, ownership, handle, source-integrity, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source_bytes = [0; 25];
    let source = matrix(helper, &source_bytes, 5, 5, "gray")?;
    let response = expect_floats(helper.call(
        "opencv.imgproc.harris_response",
        &[source.arg(), int_arg(3), int_arg(3), float_arg(0.04)],
    )?)?;
    if response != [0.0; 25] || response.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid zero-image Harris response {response:?}"));
    }
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;

    for parameters in [
        (2, 3, 0.04),
        (7, 3, 0.04),
        (3, 1, 0.04),
        (3, 9, 0.04),
        (3, 3, 0.0),
        (3, 3, 0.26),
        (3, 3, f64::NAN),
        (3, 3, f64::INFINITY),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.harris_response",
                &[
                    source.arg(),
                    int_arg(parameters.0),
                    int_arg(parameters.1),
                    float_arg(parameters.2),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let color = matrix(helper, &[0; 75], 5, 5, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.harris_response",
                &[invalid.arg(), int_arg(3), int_arg(3), float_arg(0.04)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.harris_response",
            &[
                source.arg_with_owner("foreign-worker"),
                int_arg(3),
                int_arg(3),
                float_arg(0.04),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.harris_response",
            &[
                source.arg_with_type("other.Package.Resource"),
                int_arg(3),
                int_arg(3),
                float_arg(0.04),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 25], 5, 5, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.harris_response",
            &[stale.arg(), int_arg(3), int_arg(3), float_arg(0.04)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
