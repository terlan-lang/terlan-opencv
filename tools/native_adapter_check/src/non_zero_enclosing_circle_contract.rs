//! Copied nonzero grayscale-mask enclosing-circle contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs circle geometry, empty, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask_bytes = [0, 0, 0, 2, 0, 9, 0, 0, 0];
    let mask = matrix(helper, &mask_bytes, 3, 3, "gray")?;
    expect_circle(
        expect_floats(helper.call("opencv.imgproc.non_zero_enclosing_circle", &[mask.arg()])?)?,
        1.0,
        1.0,
        1.0,
        1.001,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, &mask_bytes)?;

    let corners = matrix(helper, &[1, 0, 1, 0, 0, 0, 1, 0, 1], 3, 3, "gray")?;
    expect_circle(
        expect_floats(helper.call("opencv.imgproc.non_zero_enclosing_circle", &[corners.arg()])?)?,
        1.0,
        1.0,
        2.0_f64.sqrt(),
        2.0_f64.sqrt() + 0.001,
    )?;
    let single = matrix(helper, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 3, 4, "gray")?;
    assert_eq!(
        expect_floats(helper.call("opencv.imgproc.non_zero_enclosing_circle", &[single.arg()],)?)?,
        [3.0, 2.0, 0.0]
    );
    let empty = matrix(helper, &[0; 12], 3, 4, "gray")?;
    assert!(expect_floats(
        helper.call("opencv.imgproc.non_zero_enclosing_circle", &[empty.arg()],)?
    )?
    .is_empty());

    let bgr = matrix(helper, &[0; 36], 3, 4, "bgr")?;
    let bgra = matrix(helper, &[0; 48], 3, 4, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.non_zero_enclosing_circle", &[image.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_enclosing_circle",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_enclosing_circle",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &mask_bytes, 3, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_enclosing_circle", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [&oversized, &bgra, &bgr, &empty, &single, &corners, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Validates copied circle coordinates and a bounded native radius interval.
fn expect_circle(
    values: Vec<f64>,
    expected_x: f64,
    expected_y: f64,
    minimum_radius: f64,
    maximum_radius: f64,
) -> Result<(), String> {
    if values.len() != 3
        || (values[0] - expected_x).abs() > 1.0e-5
        || (values[1] - expected_y).abs() > 1.0e-5
        || values[2] < minimum_radius
        || values[2] > maximum_radius
    {
        return Err(format!("unexpected enclosing circle {values:?}"));
    }
    Ok(())
}
