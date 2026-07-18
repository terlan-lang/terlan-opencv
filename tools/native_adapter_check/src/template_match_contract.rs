//! Bounded copied squared-difference template-matching contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs exact scores, aliasing, bounds, handle validation, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let source_bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let pattern_bytes = [1, 2, 4, 5];
    let source = matrix(helper, &source_bytes, 3, 3, "gray")?;
    let pattern = matrix(helper, &pattern_bytes, 2, 2, "gray")?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.match_template_sqdiff",
            &[source.arg(), pattern.arg()],
        )?)?,
        &[0.0, 4.0, 36.0, 64.0],
    )?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.match_template_sqdiff",
            &[source.arg(), source.arg()],
        )?)?,
        &[0.0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[pattern.arg()])?,
        &pattern_bytes,
    )?;

    let bgr_bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let bgr = matrix(helper, &bgr_bytes, 2, 2, "bgr")?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.match_template_sqdiff",
            &[bgr.arg(), bgr.arg()],
        )?)?,
        &[0.0],
    )?;

    let larger = matrix(helper, &[0; 16], 4, 4, "gray")?;
    let color = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    let unit = matrix(helper, &[0], 1, 1, "gray")?;
    let output_heavy = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    let work_heavy = matrix(helper, &vec![0; 512 * 512], 512, 512, "gray")?;
    let work_pattern = matrix(helper, &vec![0; 256 * 256], 256, 256, "gray")?;
    for (image, invalid_pattern) in [
        (&source, &larger),
        (&source, &color),
        (&output_heavy, &unit),
        (&work_heavy, &work_pattern),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.match_template_sqdiff",
                &[image.arg(), invalid_pattern.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.match_template_sqdiff",
            &[source.arg(), pattern.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.match_template_sqdiff",
            &[
                source.arg(),
                pattern.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;

    let stale = matrix(helper, &[1], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.match_template_sqdiff",
            &[source.arg(), stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &work_pattern,
        &work_heavy,
        &output_heavy,
        &unit,
        &color,
        &larger,
        &bgr,
        &pattern,
        &source,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires copied scores to match within bounded floating-point roundoff.
fn expect_close(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-5)
    {
        return Err(format!("expected scores {expected:?}, received {actual:?}"));
    }
    Ok(())
}
