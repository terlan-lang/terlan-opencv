//! Immutable approximation contract for retained external contours.

use super::{
    expect_error_code, expect_int, expect_ints, expect_typed_handle, expect_unit, float_arg,
    matrix, NativeHelper,
};

const CONTOURS_TYPE: &str = "opencv.Contours.Contours";
#[rustfmt::skip]
const OCTAGON: &[u8] = &[
    0, 0, 0, 1, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 0, 0,
];

/// Runs exact, simplified, degenerate, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    let image = matrix(helper, OCTAGON, 9, 9, "gray")?;
    let contours = external_contours(helper, &image)?;
    let source_points = expect_ints(helper.call("opencv.contours.points", &[contours.arg()])?)?;

    let exact = approximate(helper, &contours, 0.0)?;
    expect_int(helper.call("opencv.contours.live_count", &[])?, 2)?;
    let exact_points = expect_ints(helper.call("opencv.contours.points", &[exact.arg()])?)?;
    if exact_points != source_points {
        return Err(format!(
            "zero-tolerance approximation changed points: {exact_points:?}"
        ));
    }
    expect_unit(helper.call("opencv.contours.dispose", &[exact.arg()])?)?;
    expect_int(helper.call("opencv.contours.live_count", &[])?, 1)?;

    let reduced = approximate(helper, &contours, 2.0)?;
    let reduced_points = expect_ints(helper.call("opencv.contours.points", &[reduced.arg()])?)?;
    validate_reduced_points(&reduced_points, source_points.len())?;
    if expect_ints(helper.call("opencv.contours.points", &[contours.arg()])?)? != source_points {
        return Err("approximation mutated its source contours".to_owned());
    }

    let point_image = matrix(helper, &[0, 1, 0], 1, 3, "gray")?;
    let point = external_contours(helper, &point_image)?;
    let point_approximation = approximate(helper, &point, 2.0)?;
    if expect_ints(helper.call("opencv.contours.points", &[point_approximation.arg()])?)? != [1, 0]
    {
        return Err("point approximation did not preserve its only point".to_owned());
    }

    let line_image = matrix(helper, &[1, 1, 1], 1, 3, "gray")?;
    let line = external_contours(helper, &line_image)?;
    let line_approximation = approximate(helper, &line, 2.0)?;
    let line_points =
        expect_ints(helper.call("opencv.contours.points", &[line_approximation.arg()])?)?;
    if line_points != [0, 0, 2, 0] {
        return Err(format!(
            "line approximation did not preserve its endpoints: {line_points:?}"
        ));
    }

    let empty_image = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let empty = external_contours(helper, &empty_image)?;
    let empty_approximation = approximate(helper, &empty, 0.0)?;
    expect_int(
        helper.call("opencv.contours.count", &[empty_approximation.arg()])?,
        0,
    )?;

    for epsilon in [-0.1, 12.0, f64::NAN, f64::INFINITY] {
        expect_error_code(
            helper.call(
                "opencv.contours.approximate",
                &[contours.arg(), float_arg(epsilon)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.contours.approximate",
            &[contours.arg_with_owner("foreign-worker"), float_arg(0.0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.approximate",
            &[
                contours.arg_with_type("other.Package.Resource"),
                float_arg(0.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = external_contours(helper, &image)?;
    expect_unit(helper.call("opencv.contours.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.contours.approximate",
            &[stale.arg(), float_arg(0.0)],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &empty_approximation,
        &empty,
        &line_approximation,
        &line,
        &point_approximation,
        &point,
        &reduced,
        &contours,
    ] {
        expect_unit(helper.call("opencv.contours.dispose", &[handle.arg()])?)?;
    }
    for handle in [&empty_image, &line_image, &point_image, &image] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Finds a uniquely owned external-contour analysis for one image handle.
fn external_contours(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[image.arg()])?,
        CONTOURS_TYPE,
    )
}

/// Produces one independently owned contour approximation.
fn approximate(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
    epsilon: f64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.contours.approximate",
            &[contours.arg(), float_arg(epsilon)],
        )?,
        CONTOURS_TYPE,
    )
}

/// Requires a smaller, nonempty, in-bounds flattened polygon.
fn validate_reduced_points(points: &[i64], source_scalar_count: usize) -> Result<(), String> {
    if points.len() < 6
        || points.len() >= source_scalar_count
        || !points.len().is_multiple_of(2)
        || points
            .chunks_exact(2)
            .any(|point| !(0..=8).contains(&point[0]) || !(0..=8).contains(&point[1]))
    {
        return Err(format!("unexpected reduced contour points: {points:?}"));
    }
    Ok(())
}
