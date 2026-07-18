//! Indexed convexity-defect projection contract for retained contours.

use super::{
    expect_error_code, expect_floats, expect_int, expect_typed_handle, expect_unit, int_arg,
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

/// Runs concave, convex, degenerate, index, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    let image = matrix(helper, CONCAVE, 7, 7, "gray")?;
    let contours = external_contours(helper, &image)?;
    let defects = project(helper, &contours, 0)?;
    validate_defects(&defects, 7, 7)?;

    let hulls = expect_typed_handle(
        helper.call("opencv.contours.convex_hulls", &[contours.arg()])?,
        CONTOURS_TYPE,
    )?;
    if !project(helper, &hulls, 0)?.is_empty() {
        return Err("convex hull retained convexity defects".to_owned());
    }

    let point_image = matrix(helper, &[0, 1, 0], 1, 3, "gray")?;
    let point = external_contours(helper, &point_image)?;
    if !project(helper, &point, 0)?.is_empty() {
        return Err("point contour produced convexity defects".to_owned());
    }
    let line_image = matrix(helper, &[1, 1, 1], 1, 3, "gray")?;
    let line = external_contours(helper, &line_image)?;
    if !project(helper, &line, 0)?.is_empty() {
        return Err("line contour produced convexity defects".to_owned());
    }

    let empty_image = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let empty = external_contours(helper, &empty_image)?;
    for (handle, index) in [(&contours, -1), (&contours, 1), (&empty, 0)] {
        expect_error_code(
            helper.call(
                "opencv.contours.convexity_defects",
                &[handle.arg(), int_arg(index)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.contours.convexity_defects",
            &[contours.arg_with_owner("foreign-worker"), int_arg(0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.convexity_defects",
            &[contours.arg_with_type("other.Package.Resource"), int_arg(0)],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = external_contours(helper, &image)?;
    expect_unit(helper.call("opencv.contours.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.contours.convexity_defects",
            &[stale.arg(), int_arg(0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&empty, &line, &point, &hulls, &contours] {
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

/// Copies convexity defects for one contour index.
fn project(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
    index: i64,
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.contours.convexity_defects",
        &[contours.arg(), int_arg(index)],
    )?)
}

/// Requires nonempty, finite, bounded seven-scalar defect records.
fn validate_defects(values: &[f64], rows: i64, cols: i64) -> Result<(), String> {
    let maximum_depth = ((rows - 1) as f64).hypot((cols - 1) as f64);
    if values.is_empty()
        || !values.len().is_multiple_of(7)
        || values.chunks_exact(7).any(|record| {
            record.iter().any(|value| !value.is_finite())
                || record[..6].chunks_exact(2).any(|point| {
                    point[0] < 0.0
                        || point[0] >= cols as f64
                        || point[1] < 0.0
                        || point[1] >= rows as f64
                        || point[0].fract() != 0.0
                        || point[1].fract() != 0.0
                })
                || record[6] <= 0.0
                || record[6] > maximum_depth
        })
    {
        return Err(format!("unexpected convexity defects: {values:?}"));
    }
    Ok(())
}
