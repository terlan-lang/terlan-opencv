//! Copied nonzero grayscale-mask minimum-area-box contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs box geometry, degenerate normalization, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask_bytes = [1, 0, 0, 2, 0, 9, 0, 0, 3, 0, 0, 4];
    let mask = matrix(helper, &mask_bytes, 3, 4, "gray")?;
    expect_box(
        expect_floats(helper.call("opencv.imgproc.non_zero_min_area_box", &[mask.arg()])?)?,
        &[0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 2.0],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, &mask_bytes)?;

    let diamond = matrix(
        helper,
        &[
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 9, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
        ],
        5,
        5,
        "gray",
    )?;
    expect_box(
        expect_floats(helper.call("opencv.imgproc.non_zero_min_area_box", &[diamond.arg()])?)?,
        &[2.0, 0.0, 4.0, 2.0, 2.0, 4.0, 0.0, 2.0],
    )?;
    let line = matrix(helper, &[0, 0, 0, 0, 2, 7, 0, 9], 2, 4, "gray")?;
    expect_box(
        expect_floats(helper.call("opencv.imgproc.non_zero_min_area_box", &[line.arg()])?)?,
        &[0.0, 1.0, 3.0, 1.0],
    )?;
    let single = matrix(helper, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, "gray")?;
    expect_box(
        expect_floats(helper.call("opencv.imgproc.non_zero_min_area_box", &[single.arg()])?)?,
        &[3.0, 2.0],
    )?;
    let empty = matrix(helper, &[0; 12], 3, 4, "gray")?;
    assert!(
        expect_floats(helper.call("opencv.imgproc.non_zero_min_area_box", &[empty.arg()])?)?
            .is_empty()
    );

    let bgr = matrix(helper, &[0; 36], 3, 4, "bgr")?;
    let bgra = matrix(helper, &[0; 48], 3, 4, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.non_zero_min_area_box", &[image.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_min_area_box",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_min_area_box",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &mask_bytes, 3, 4, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_min_area_box", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &oversized, &bgra, &bgr, &empty, &single, &line, &diamond, &mask,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Compares copied box coordinates with a small native floating-point tolerance.
fn expect_box(actual: Vec<f64>, expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(left, right)| (left - right).abs() > 1.0e-4)
    {
        return Err(format!("unexpected minimum-area box {actual:?}"));
    }
    Ok(())
}
