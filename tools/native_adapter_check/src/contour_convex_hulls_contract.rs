//! Immutable convex-hull transformation contract for retained contours.

use super::{
    expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle, expect_unit,
    matrix, NativeHelper,
};

const CONTOURS_TYPE: &str = "opencv.Contours.Contours";
#[rustfmt::skip]
const CONCAVE: &[u8] = &[
    0, 0,   0,   0,   0,   0,   0,
    0, 255, 255, 0,   0,   0,   0,
    0, 255, 255, 0,   0,   0,   0,
    0, 255, 255, 0,   0,   0,   0,
    0, 255, 255, 255, 255, 255, 0,
    0, 255, 255, 255, 255, 255, 0,
    0, 0,   0,   0,   0,   0,   0,
];

/// Runs concave, degenerate, empty, ownership, rejection, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    let image = matrix(helper, CONCAVE, 7, 7, "gray")?;
    let contours = external_contours(helper, &image)?;
    let source_points = expect_ints(helper.call("opencv.contours.points", &[contours.arg()])?)?;
    let source_area = only_float(
        expect_floats(helper.call("opencv.contours.areas", &[contours.arg()])?)?,
        "source area",
    )?;
    let source_convex_area = only_float(
        expect_floats(helper.call("opencv.contours.convex_areas", &[contours.arg()])?)?,
        "source convex area",
    )?;

    let hulls = convex_hulls(helper, &contours)?;
    expect_int(helper.call("opencv.contours.live_count", &[])?, 2)?;
    let hull_points = expect_ints(helper.call("opencv.contours.points", &[hulls.arg()])?)?;
    let hull_area = only_float(
        expect_floats(helper.call("opencv.contours.areas", &[hulls.arg()])?)?,
        "hull area",
    )?;
    let hull_convex_area = only_float(
        expect_floats(helper.call("opencv.contours.convex_areas", &[hulls.arg()])?)?,
        "hull convex area",
    )?;
    if hull_points.len() >= source_points.len()
        || source_convex_area <= source_area
        || (hull_area - source_convex_area).abs() > 1.0e-9
        || (hull_convex_area - hull_area).abs() > 1.0e-9
    {
        return Err(format!(
            "unexpected contour hull: source={source_points:?}/{source_area}/{source_convex_area}, hull={hull_points:?}/{hull_area}/{hull_convex_area}"
        ));
    }
    if expect_ints(helper.call("opencv.contours.points", &[contours.arg()])?)? != source_points {
        return Err("convex-hull transformation mutated its source".to_owned());
    }

    let point_image = matrix(helper, &[0, 1, 0], 1, 3, "gray")?;
    let point = external_contours(helper, &point_image)?;
    let point_hull = convex_hulls(helper, &point)?;
    if expect_ints(helper.call("opencv.contours.points", &[point_hull.arg()])?)? != [1, 0] {
        return Err("point hull changed its only point".to_owned());
    }

    let line_image = matrix(helper, &[1, 1, 1], 1, 3, "gray")?;
    let line = external_contours(helper, &line_image)?;
    let line_hull = convex_hulls(helper, &line)?;
    if expect_ints(helper.call("opencv.contours.points", &[line_hull.arg()])?)?.len() != 4 {
        return Err("line hull did not retain two endpoints".to_owned());
    }

    let empty_image = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let empty = external_contours(helper, &empty_image)?;
    let empty_hull = convex_hulls(helper, &empty)?;
    expect_int(
        helper.call("opencv.contours.count", &[empty_hull.arg()])?,
        0,
    )?;

    expect_error_code(
        helper.call(
            "opencv.contours.convex_hulls",
            &[contours.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.convex_hulls",
            &[contours.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = external_contours(helper, &image)?;
    expect_unit(helper.call("opencv.contours.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.contours.convex_hulls", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &empty_hull,
        &empty,
        &line_hull,
        &line,
        &point_hull,
        &point,
        &hulls,
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

/// Finds one uniquely owned external-contour analysis.
fn external_contours(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[image.arg()])?,
        CONTOURS_TYPE,
    )
}

/// Produces independently owned hull geometry for one contour analysis.
fn convex_hulls(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.contours.convex_hulls", &[contours.arg()])?,
        CONTOURS_TYPE,
    )
}

/// Extracts one required scalar projection value.
fn only_float(values: Vec<f64>, kind: &str) -> Result<f64, String> {
    match values.as_slice() {
        [value] if value.is_finite() => Ok(*value),
        _ => Err(format!("unexpected {kind}: {values:?}")),
    }
}
