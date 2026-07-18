//! Copied grayscale corner eigensystem contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, int_arg, matrix,
    NativeHelper, CORNER_IMAGE,
};

/// Runs tuple, source-integrity, bound, handle, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, CORNER_IMAGE, 7, 7, "gray")?;
    let response = expect_floats(helper.call(
        "opencv.imgproc.corner_eigensystem",
        &[source.arg(), int_arg(3), int_arg(3)],
    )?)?;
    validate_eigensystems(&response, 49, true)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        CORNER_IMAGE,
    )?;

    let zero = matrix(helper, &[0; 25], 5, 5, "gray")?;
    let zero_response = expect_floats(helper.call(
        "opencv.imgproc.corner_eigensystem",
        &[zero.arg(), int_arg(3), int_arg(3)],
    )?)?;
    validate_eigensystems(&zero_response, 25, false)?;

    for parameters in [(2, 3), (9, 3), (3, 1), (3, 9)] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.corner_eigensystem",
                &[source.arg(), int_arg(parameters.0), int_arg(parameters.1)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let color = matrix(helper, &[0; 75], 5, 5, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.corner_eigensystem",
                &[invalid.arg(), int_arg(3), int_arg(3)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.corner_eigensystem",
            &[
                source.arg_with_owner("foreign-worker"),
                int_arg(3),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.corner_eigensystem",
            &[
                source.arg_with_type("other.Package.Resource"),
                int_arg(3),
                int_arg(3),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 25], 5, 5, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.corner_eigensystem",
            &[stale.arg(), int_arg(3), int_arg(3)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &zero, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Validates flattened finite eigensystem tuples and response presence.
fn validate_eigensystems(
    values: &[f64],
    pixels: usize,
    require_response: bool,
) -> Result<(), String> {
    let valid = values.len() == pixels * 6
        && values.iter().all(|value| value.is_finite())
        && values
            .chunks_exact(6)
            .all(|tuple| tuple[0] >= tuple[1] && tuple[1] >= 0.0)
        && values.chunks_exact(6).any(|tuple| tuple[0] > 0.0) == require_response;
    if valid {
        Ok(())
    } else {
        Err(format!("invalid corner eigensystems {values:?}"))
    }
}
