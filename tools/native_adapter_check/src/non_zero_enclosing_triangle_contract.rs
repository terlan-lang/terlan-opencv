//! Copied nonzero grayscale-mask minimum-enclosing-triangle contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs triangle geometry, degenerate normalization, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask_bytes = [0, 1, 0, 0, 9, 0, 2, 0, 3];
    let mask = matrix(helper, &mask_bytes, 3, 3, "gray")?;
    expect_triangle(
        expect_floats(helper.call("opencv.imgproc.non_zero_enclosing_triangle", &[mask.arg()])?)?,
        &[1.0, 0.0, 2.0, 2.0, 0.0, 2.0],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, &mask_bytes)?;

    let line = matrix(helper, &[0, 0, 0, 0, 2, 7, 0, 9], 2, 4, "gray")?;
    expect_triangle(
        expect_floats(helper.call("opencv.imgproc.non_zero_enclosing_triangle", &[line.arg()])?)?,
        &[0.0, 1.0, 3.0, 1.0],
    )?;
    let single = matrix(helper, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, "gray")?;
    expect_triangle(
        expect_floats(helper.call(
            "opencv.imgproc.non_zero_enclosing_triangle",
            &[single.arg()],
        )?)?,
        &[3.0, 2.0],
    )?;
    let empty = matrix(helper, &[0; 12], 3, 4, "gray")?;
    assert!(expect_floats(
        helper.call("opencv.imgproc.non_zero_enclosing_triangle", &[empty.arg()])?
    )?
    .is_empty());

    let bgr = matrix(helper, &[0; 27], 3, 3, "bgr")?;
    let bgra = matrix(helper, &[0; 36], 3, 3, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.non_zero_enclosing_triangle", &[image.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_enclosing_triangle",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_enclosing_triangle",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &mask_bytes, 3, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_enclosing_triangle", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [&oversized, &bgra, &bgr, &empty, &single, &line, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Compares copied triangle coordinates with a small native precision tolerance.
fn expect_triangle(actual: Vec<f64>, expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(left, right)| (left - right).abs() > 1.0e-4)
    {
        return Err(format!("unexpected enclosing triangle {actual:?}"));
    }
    Ok(())
}
