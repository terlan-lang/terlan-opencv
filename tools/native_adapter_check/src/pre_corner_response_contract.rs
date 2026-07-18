//! Copied grayscale pre-corner derivative-response contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, int_arg, matrix,
    NativeHelper, CORNER_IMAGE,
};

/// Runs response, source-integrity, aperture, handle, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, CORNER_IMAGE, 7, 7, "gray")?;
    let response = expect_floats(helper.call(
        "opencv.imgproc.pre_corner_response",
        &[source.arg(), int_arg(3)],
    )?)?;
    if response.len() != 49
        || response.iter().any(|value| !value.is_finite())
        || !response.iter().any(|value| value.abs() > 0.0)
    {
        return Err(format!("invalid pre-corner response {response:?}"));
    }
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        CORNER_IMAGE,
    )?;

    let zero = matrix(helper, &[0; 25], 5, 5, "gray")?;
    if expect_floats(helper.call(
        "opencv.imgproc.pre_corner_response",
        &[zero.arg(), int_arg(3)],
    )?)? != [0.0; 25]
    {
        return Err("zero image produced nonzero pre-corner values".to_string());
    }
    for aperture in [1, 9] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.pre_corner_response",
                &[source.arg(), int_arg(aperture)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let color = matrix(helper, &[0; 75], 5, 5, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.pre_corner_response",
                &[invalid.arg(), int_arg(3)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.pre_corner_response",
            &[source.arg_with_owner("foreign-worker"), int_arg(3)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.pre_corner_response",
            &[source.arg_with_type("other.Package.Resource"), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 25], 5, 5, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.pre_corner_response",
            &[stale.arg(), int_arg(3)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &zero, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
