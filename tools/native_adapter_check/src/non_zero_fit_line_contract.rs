//! Copied nonzero grayscale-mask least-squares line contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs fitted-line geometry, normalization, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let horizontal_bytes = [2, 7, 9];
    let horizontal = matrix(helper, &horizontal_bytes, 1, 3, "gray")?;
    expect_line(
        expect_floats(helper.call("opencv.imgproc.non_zero_fit_line", &[horizontal.arg()])?)?,
        [1.0, 0.0, 1.0, 0.0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[horizontal.arg()])?,
        &horizontal_bytes,
    )?;

    let vertical = matrix(helper, &[2, 0, 7, 0, 9, 0], 3, 2, "gray")?;
    expect_line(
        expect_floats(helper.call("opencv.imgproc.non_zero_fit_line", &[vertical.arg()])?)?,
        [0.0, 1.0, 0.0, 1.0],
    )?;
    let diagonal = matrix(helper, &[1, 0, 0, 0, 2, 0, 0, 0, 3], 3, 3, "gray")?;
    expect_line(
        expect_floats(helper.call("opencv.imgproc.non_zero_fit_line", &[diagonal.arg()])?)?,
        [2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0, 1.0, 1.0],
    )?;
    let single = matrix(helper, &[0, 0, 0, 0, 1, 0], 2, 3, "gray")?;
    let empty = matrix(helper, &[0; 6], 2, 3, "gray")?;
    for image in [&single, &empty] {
        assert!(
            expect_floats(helper.call("opencv.imgproc.non_zero_fit_line", &[image.arg()])?)?
                .is_empty()
        );
    }

    let bgr = matrix(helper, &[0; 18], 2, 3, "bgr")?;
    let bgra = matrix(helper, &[0; 24], 2, 3, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.non_zero_fit_line", &[image.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_fit_line",
            &[horizontal.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_fit_line",
            &[horizontal.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &horizontal_bytes, 1, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_fit_line", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &oversized,
        &bgra,
        &bgr,
        &empty,
        &single,
        &diagonal,
        &vertical,
        &horizontal,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Compares copied line values with a small native precision tolerance.
fn expect_line(actual: Vec<f64>, expected: [f64; 4]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(left, right)| (left - right).abs() > 1.0e-4)
    {
        return Err(format!("unexpected fitted line {actual:?}"));
    }
    Ok(())
}
