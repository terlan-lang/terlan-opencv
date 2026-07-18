//! Copied Douglas-Peucker approximation of a nonzero grayscale-mask hull.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_ints, expect_unit, float_arg, matrix,
    NativeHelper,
};

const OCTAGON: &[u8] = &[
    0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0,
    0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 8, 0, 0,
];

/// Runs exact, reduced, degenerate, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let octagon = matrix(helper, OCTAGON, 7, 7, "gray")?;
    assert_eq!(
        expect_ints(helper.call(
            "opencv.imgproc.non_zero_approximate_hull",
            &[octagon.arg(), float_arg(0.0)],
        )?)?,
        [2, 0, 4, 0, 6, 2, 6, 4, 4, 6, 2, 6, 0, 4, 0, 2]
    );
    let reduced = expect_ints(helper.call(
        "opencv.imgproc.non_zero_approximate_hull",
        &[octagon.arg(), float_arg(1.0)],
    )?)?;
    validate_reduced_hull(&reduced)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[octagon.arg()])?, OCTAGON)?;

    let point = matrix(helper, &[0, 5, 0], 1, 3, "gray")?;
    assert_eq!(
        expect_ints(helper.call(
            "opencv.imgproc.non_zero_approximate_hull",
            &[point.arg(), float_arg(1.0)],
        )?)?,
        [1, 0]
    );
    let line = matrix(helper, &[1, 2, 3], 1, 3, "gray")?;
    assert_eq!(
        expect_ints(helper.call(
            "opencv.imgproc.non_zero_approximate_hull",
            &[line.arg(), float_arg(2.0)],
        )?)?,
        [0, 0, 2, 0]
    );
    let empty = matrix(helper, &[0; 9], 3, 3, "gray")?;
    assert!(expect_ints(helper.call(
        "opencv.imgproc.non_zero_approximate_hull",
        &[empty.arg(), float_arg(0.0)],
    )?)?
    .is_empty());

    for epsilon in [-0.1, 9.0, f64::NAN, f64::INFINITY] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.non_zero_approximate_hull",
                &[octagon.arg(), float_arg(epsilon)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let bgr = matrix(helper, &[0; 27], 3, 3, "bgr")?;
    let bgra = matrix(helper, &[0; 36], 3, 3, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.non_zero_approximate_hull",
                &[image.arg(), float_arg(0.0)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_approximate_hull",
            &[octagon.arg_with_owner("foreign-worker"), float_arg(0.0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_approximate_hull",
            &[
                octagon.arg_with_type("other.Package.Resource"),
                float_arg(0.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, OCTAGON, 7, 7, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_approximate_hull",
            &[stale.arg(), float_arg(0.0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &bgra, &bgr, &empty, &line, &point, &octagon] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Validates that a relaxed approximation is a smaller bounded polygon.
fn validate_reduced_hull(values: &[i64]) -> Result<(), String> {
    if values.len() < 6 || values.len() >= 16 || !values.len().is_multiple_of(2) {
        return Err(format!("unexpected reduced hull {values:?}"));
    }
    if values
        .chunks_exact(2)
        .any(|point| point[0] < 0 || point[0] > 6 || point[1] < 0 || point[1] > 6)
    {
        return Err(format!("out-of-bounds reduced hull {values:?}"));
    }
    Ok(())
}
