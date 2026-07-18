//! Copied per-channel mean absolute error contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

const FIRST: &[u8] = &[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
const SECOND: &[u8] = &[10, 8, 25, 20, 60, 40, 65, 90, 70, 100, 80, 120];

/// Runs channel ordering, aliasing, rejection, preservation, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST, 2, 2, "bgr")?;
    let second = matrix(helper, SECOND, 2, 2, "bgr")?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[first.arg(), second.arg()],
        )?)?,
        &[8.75, 15.5, 8.75],
    )?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[first.arg(), first.arg()],
        )?)?,
        &[0.0, 0.0, 0.0],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[first.arg()])?, FIRST)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[second.arg()])?, SECOND)?;

    let gray_first = matrix(helper, &[0, 10, 20, 30], 2, 2, "gray")?;
    let gray_second = matrix(helper, &[0, 20, 10, 50], 2, 2, "gray")?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[gray_first.arg(), gray_second.arg()],
        )?)?,
        &[10.0],
    )?;
    let bgra_first = matrix(helper, &[0, 10, 20, 30, 40, 50, 60, 70], 1, 2, "bgra")?;
    let bgra_second = matrix(helper, &[10, 0, 30, 20, 50, 40, 70, 60], 1, 2, "bgra")?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[bgra_first.arg(), bgra_second.arg()],
        )?)?,
        &[10.0, 10.0, 10.0, 10.0],
    )?;

    let wrong_shape = matrix(helper, &[0; 6], 1, 2, "bgr")?;
    let wrong_layout = matrix(helper, &[0; 4], 2, 2, "gray")?;
    for invalid in [&wrong_shape, &wrong_layout] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.mean_absolute_error_channels",
                &[first.arg(), invalid.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[first.arg(), second.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[first.arg(), second.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mean_absolute_error_channels",
            &[first.arg(), stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_layout,
        &wrong_shape,
        &bgra_second,
        &bgra_first,
        &gray_second,
        &gray_first,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires copied errors to match within bounded floating-point roundoff.
fn expect_close(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-9)
    {
        return Err(format!(
            "expected channel errors {expected:?}, received {actual:?}"
        ));
    }
    Ok(())
}
