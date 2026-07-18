//! Copied nonzero grayscale-mask least-squares ellipse contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs fitted-ellipse geometry, normalization, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let horizontal_bytes = ellipse_mask(21, 11, &horizontal_ellipse_points());
    let horizontal = matrix(helper, &horizontal_bytes, 11, 21, "gray")?;
    expect_ellipse(
        expect_floats(helper.call("opencv.imgproc.non_zero_fit_ellipse", &[horizontal.arg()])?)?,
        [10.0, 5.0, 20.0, 10.0, 0.0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[horizontal.arg()])?,
        &horizontal_bytes,
    )?;

    let vertical_bytes = ellipse_mask(11, 21, &vertical_ellipse_points());
    let vertical = matrix(helper, &vertical_bytes, 21, 11, "gray")?;
    expect_ellipse(
        expect_floats(helper.call("opencv.imgproc.non_zero_fit_ellipse", &[vertical.arg()])?)?,
        [5.0, 10.0, 20.0, 10.0, 90.0],
    )?;
    let circle_bytes = ellipse_mask(11, 11, &circle_points());
    let circle = matrix(helper, &circle_bytes, 11, 11, "gray")?;
    expect_ellipse(
        expect_floats(helper.call("opencv.imgproc.non_zero_fit_ellipse", &[circle.arg()])?)?,
        [5.0, 5.0, 10.0, 10.0, 0.0],
    )?;

    let insufficient = matrix(helper, &[1, 1, 1, 1, 0, 0], 2, 3, "gray")?;
    let collinear = matrix(helper, &[1, 1, 1, 1, 1], 1, 5, "gray")?;
    for image in [&insufficient, &collinear] {
        assert!(expect_floats(
            helper.call("opencv.imgproc.non_zero_fit_ellipse", &[image.arg()],)?
        )?
        .is_empty());
    }

    let bgr = matrix(helper, &[0; 18], 2, 3, "bgr")?;
    let bgra = matrix(helper, &[0; 24], 2, 3, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.non_zero_fit_ellipse", &[image.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_fit_ellipse",
            &[horizontal.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_fit_ellipse",
            &[horizontal.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &circle_bytes, 11, 11, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_fit_ellipse", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &oversized,
        &bgra,
        &bgr,
        &collinear,
        &insufficient,
        &circle,
        &vertical,
        &horizontal,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Builds a nonbinary grayscale mask from exact coordinate pairs.
fn ellipse_mask(cols: usize, rows: usize, points: &[(usize, usize)]) -> Vec<u8> {
    let mut bytes = vec![0; cols * rows];
    for (index, &(x, y)) in points.iter().enumerate() {
        bytes[y * cols + x] = (index % 254 + 1) as u8;
    }
    bytes
}

/// Returns exact points on a horizontal ellipse centered at ten, five.
fn horizontal_ellipse_points() -> Vec<(usize, usize)> {
    vec![
        (20, 5),
        (0, 5),
        (10, 10),
        (10, 0),
        (16, 9),
        (16, 1),
        (4, 9),
        (4, 1),
        (18, 8),
        (18, 2),
        (2, 8),
        (2, 2),
    ]
}

/// Returns exact points on a vertical ellipse centered at five, ten.
fn vertical_ellipse_points() -> Vec<(usize, usize)> {
    horizontal_ellipse_points()
        .into_iter()
        .map(|(x, y)| (y, x))
        .collect()
}

/// Returns exact points on a circle centered at five, five.
fn circle_points() -> Vec<(usize, usize)> {
    vec![
        (10, 5),
        (0, 5),
        (5, 10),
        (5, 0),
        (8, 9),
        (8, 1),
        (2, 9),
        (2, 1),
        (9, 8),
        (9, 2),
        (1, 8),
        (1, 2),
    ]
}

/// Compares copied ellipse values with a small native precision tolerance.
fn expect_ellipse(actual: Vec<f64>, expected: [f64; 5]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(left, right)| (left - right).abs() > 1.0e-3)
    {
        return Err(format!("unexpected fitted ellipse {actual:?}"));
    }
    Ok(())
}
