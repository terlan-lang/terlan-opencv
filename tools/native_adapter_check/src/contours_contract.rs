//! External-contour ownership and copied-projection contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_floats, expect_int, expect_ints,
    expect_typed_handle, expect_unit, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const CONTOURS_TYPE: &str = "opencv.Contours.Contours";
const RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
];
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
#[rustfmt::skip]
const MULTI_SHAPE: &[u8] = &[
    0, 0,   0,   0, 0,   0,   0,   0, 0,
    0, 255, 255, 255, 0,   0,   0,   0, 0,
    0, 255, 255, 255, 0,   0,   0,   0, 0,
    0, 255, 255, 255, 0,   0,   0,   0, 0,
    0, 0,   0,   0, 0,   255, 255, 0, 0,
    0, 0,   0,   0, 0,   255, 255, 0, 0,
    0, 0,   0,   0, 0,   255, 255, 0, 0,
    0, 0,   0,   0, 0,   255, 255, 255, 0,
    0, 0,   0,   0, 0,   0,   0,   0, 0,
];

/// Runs deterministic contour, rejection, and independent-lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    let mask = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(RECTANGLE),
                int_arg(5),
                int_arg(5),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )?;
    let contours = expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[mask.arg()])?,
        CONTOURS_TYPE,
    )?;
    expect_int(helper.call("opencv.contours.live_count", &[])?, 1)?;
    expect_int(helper.call("opencv.contours.count", &[contours.arg()])?, 1)?;
    expect_exact(
        expect_ints(helper.call("opencv.contours.points", &[contours.arg()])?)?,
        &[1, 1, 1, 3, 3, 3, 3, 1],
        "points",
    )?;
    expect_exact(
        expect_ints(helper.call("opencv.contours.offsets", &[contours.arg()])?)?,
        &[0, 4],
        "offsets",
    )?;
    expect_exact(
        expect_ints(helper.call("opencv.contours.bounds", &[contours.arg()])?)?,
        &[1, 1, 3, 3],
        "bounds",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.areas", &[contours.arg()])?)?,
        &[4.0],
        "areas",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.perimeters", &[contours.arg()])?)?,
        &[8.0],
        "perimeters",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.convex_areas", &[contours.arg()])?)?,
        &[4.0],
        "convex areas",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.solidities", &[contours.arg()])?)?,
        &[1.0],
        "solidities",
    )?;
    let rectangle_circularities =
        expect_floats(helper.call("opencv.contours.circularities", &[contours.arg()])?)?;
    if rectangle_circularities.len() != 1
        || (rectangle_circularities[0] - std::f64::consts::FRAC_PI_4).abs() > 1.0e-12
    {
        return Err(format!(
            "unexpected rectangle circularities: {rectangle_circularities:?}"
        ));
    }
    let rectangle_extents =
        expect_floats(helper.call("opencv.contours.extents", &[contours.arg()])?)?;
    if rectangle_extents.len() != 1 || (rectangle_extents[0] - 4.0 / 9.0).abs() > 1.0e-12 {
        return Err(format!(
            "unexpected rectangle extents: {rectangle_extents:?}"
        ));
    }
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.centroids", &[contours.arg()])?)?,
        &[2.0, 2.0],
        "centroids",
    )?;
    let rectangle_hu_moments =
        expect_floats(helper.call("opencv.contours.hu_moments", &[contours.arg()])?)?;
    if rectangle_hu_moments.len() != 7
        || rectangle_hu_moments.iter().any(|value| !value.is_finite())
        || rectangle_hu_moments[0] <= 0.0
    {
        return Err(format!(
            "unexpected rectangle Hu moments: {rectangle_hu_moments:?}"
        ));
    }
    let rectangle_minimum_area_boxes =
        expect_floats(helper.call("opencv.contours.minimum_area_boxes", &[contours.arg()])?)?;
    expect_close_floats(
        &rectangle_minimum_area_boxes,
        &[1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0],
        "minimum-area boxes",
    )?;
    let rectangle_minimum_enclosing_circles = expect_floats(helper.call(
        "opencv.contours.minimum_enclosing_circles",
        &[contours.arg()],
    )?)?;
    expect_circle(
        &rectangle_minimum_enclosing_circles,
        2.0,
        2.0,
        1.414,
        1.415,
        "rectangle minimum enclosing circle",
    )?;
    let rectangle_minimum_enclosing_triangles = expect_floats(helper.call(
        "opencv.contours.minimum_enclosing_triangles",
        &[contours.arg()],
    )?)?;
    expect_area_triangle(
        &rectangle_minimum_enclosing_triangles,
        "rectangle minimum enclosing triangle",
    )?;
    let rectangle_fitted_lines =
        expect_floats(helper.call("opencv.contours.fitted_lines", &[contours.arg()])?)?;
    expect_fitted_line(&rectangle_fitted_lines, "rectangle fitted line")?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.fitted_ellipses", &[contours.arg()])?)?,
        &[2.0, 2.0, 0.0, 0.0, 0.0],
        "rectangle fitted ellipse",
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, RECTANGLE)?;

    let concave = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(CONCAVE), int_arg(7), int_arg(7), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    let concave_contours = expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[concave.arg()])?,
        CONTOURS_TYPE,
    )?;
    let concave_areas =
        expect_floats(helper.call("opencv.contours.areas", &[concave_contours.arg()])?)?;
    let concave_bounds =
        expect_ints(helper.call("opencv.contours.bounds", &[concave_contours.arg()])?)?;
    let concave_perimeters =
        expect_floats(helper.call("opencv.contours.perimeters", &[concave_contours.arg()])?)?;
    let concave_convex_areas =
        expect_floats(helper.call("opencv.contours.convex_areas", &[concave_contours.arg()])?)?;
    let concave_solidities =
        expect_floats(helper.call("opencv.contours.solidities", &[concave_contours.arg()])?)?;
    let concave_circularities =
        expect_floats(helper.call("opencv.contours.circularities", &[concave_contours.arg()])?)?;
    let concave_extents =
        expect_floats(helper.call("opencv.contours.extents", &[concave_contours.arg()])?)?;
    let concave_centroids =
        expect_floats(helper.call("opencv.contours.centroids", &[concave_contours.arg()])?)?;
    let concave_hu_moments =
        expect_floats(helper.call("opencv.contours.hu_moments", &[concave_contours.arg()])?)?;
    let concave_minimum_area_boxes = expect_floats(helper.call(
        "opencv.contours.minimum_area_boxes",
        &[concave_contours.arg()],
    )?)?;
    let concave_minimum_enclosing_circles = expect_floats(helper.call(
        "opencv.contours.minimum_enclosing_circles",
        &[concave_contours.arg()],
    )?)?;
    let concave_minimum_enclosing_triangles = expect_floats(helper.call(
        "opencv.contours.minimum_enclosing_triangles",
        &[concave_contours.arg()],
    )?)?;
    let concave_fitted_lines =
        expect_floats(helper.call("opencv.contours.fitted_lines", &[concave_contours.arg()])?)?;
    let concave_fitted_ellipses =
        expect_floats(helper.call("opencv.contours.fitted_ellipses", &[concave_contours.arg()])?)?;
    if concave_areas.len() != 1
        || concave_bounds.len() != 4
        || concave_perimeters.len() != 1
        || concave_convex_areas.len() != 1
        || concave_solidities.len() != 1
        || concave_circularities.len() != 1
        || concave_extents.len() != 1
        || concave_centroids.len() != 2
        || concave_hu_moments.len() != 7
        || concave_minimum_area_boxes.len() != 8
        || concave_minimum_enclosing_circles.len() != 3
        || concave_minimum_enclosing_triangles.len() != 6
        || concave_fitted_lines.len() != 4
        || concave_fitted_ellipses.len() != 5
        || !concave_areas[0].is_finite()
        || !concave_perimeters[0].is_finite()
        || !concave_convex_areas[0].is_finite()
        || !concave_solidities[0].is_finite()
        || !concave_circularities[0].is_finite()
        || !concave_extents[0].is_finite()
        || !concave_centroids[0].is_finite()
        || !concave_centroids[1].is_finite()
        || concave_hu_moments.iter().any(|value| !value.is_finite())
        || concave_hu_moments == rectangle_hu_moments
        || concave_minimum_area_boxes
            .iter()
            .any(|value| !value.is_finite())
        || concave_minimum_area_boxes == rectangle_minimum_area_boxes
        || concave_minimum_enclosing_circles
            .iter()
            .any(|value| !value.is_finite())
        || concave_minimum_enclosing_circles[2] <= 0.0
        || concave_minimum_enclosing_triangles
            .iter()
            .any(|value| !value.is_finite())
        || triangle_area(&concave_minimum_enclosing_triangles) <= 0.0
        || concave_fitted_lines.iter().any(|value| !value.is_finite())
        || (concave_fitted_lines[0].hypot(concave_fitted_lines[1]) - 1.0).abs() > 1.0e-3
        || concave_fitted_ellipses
            .iter()
            .any(|value| !value.is_finite())
        || concave_fitted_ellipses[2] < concave_fitted_ellipses[3]
        || concave_fitted_ellipses[3] <= 0.0
        || !(0.0..180.0).contains(&concave_fitted_ellipses[4])
        || concave_areas[0] <= 0.0
        || concave_perimeters[0] <= 0.0
        || concave_convex_areas[0] <= concave_areas[0]
        || !(0.0..1.0).contains(&concave_solidities[0])
        || !(0.0..1.0).contains(&concave_circularities[0])
        || !(0.0..1.0).contains(&concave_extents[0])
        || concave_centroids[0] < concave_bounds[0] as f64
        || concave_centroids[0] > (concave_bounds[0] + concave_bounds[2] - 1) as f64
        || concave_centroids[1] < concave_bounds[1] as f64
        || concave_centroids[1] > (concave_bounds[1] + concave_bounds[3] - 1) as f64
        || (concave_solidities[0] - concave_areas[0] / concave_convex_areas[0]).abs() > 1.0e-12
        || (concave_circularities[0]
            - 4.0 * std::f64::consts::PI * concave_areas[0] / concave_perimeters[0].powi(2))
        .abs()
            > 1.0e-12
        || (concave_extents[0] - concave_areas[0] / (concave_bounds[2] * concave_bounds[3]) as f64)
            .abs()
            > 1.0e-12
    {
        return Err(format!(
            "unexpected concave descriptors: area={concave_areas:?}, bounds={concave_bounds:?}, perimeter={concave_perimeters:?}, convex={concave_convex_areas:?}, solidity={concave_solidities:?}, circularity={concave_circularities:?}, extent={concave_extents:?}, centroid={concave_centroids:?}, hu={concave_hu_moments:?}, minimum_box={concave_minimum_area_boxes:?}, minimum_circle={concave_minimum_enclosing_circles:?}, minimum_triangle={concave_minimum_enclosing_triangles:?}, fitted_line={concave_fitted_lines:?}, fitted_ellipse={concave_fitted_ellipses:?}"
        ));
    }
    expect_bytes(helper.call("opencv.mat.to_u8", &[concave.arg()])?, CONCAVE)?;

    let multi_shape = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(MULTI_SHAPE),
                int_arg(9),
                int_arg(9),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )?;
    let multi_contours = expect_typed_handle(
        helper.call(
            "opencv.imgproc.find_external_contours",
            &[multi_shape.arg()],
        )?,
        CONTOURS_TYPE,
    )?;
    expect_int(
        helper.call("opencv.contours.count", &[multi_contours.arg()])?,
        2,
    )?;
    let multi_scores = expect_floats(helper.call(
        "opencv.contours.match_scores",
        &[multi_contours.arg(), int_arg(0)],
    )?)?;
    if multi_scores.len() != 6
        || multi_scores
            .iter()
            .any(|score| !score.is_finite() || *score < 0.0)
        || multi_scores[..3] != [0.0, 0.0, 0.0]
        || multi_scores[3..].iter().all(|score| *score <= 1.0e-6)
    {
        return Err(format!(
            "unexpected multi-contour match scores: {multi_scores:?}"
        ));
    }
    for reference_index in [-1, 2] {
        expect_error_code(
            helper.call(
                "opencv.contours.match_scores",
                &[multi_contours.arg(), int_arg(reference_index)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let point = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(&[0, 0, 0, 0, 255, 0, 0, 0, 0]),
                int_arg(3),
                int_arg(3),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )?;
    let point_contours = expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[point.arg()])?,
        CONTOURS_TYPE,
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.convex_areas", &[point_contours.arg()])?)?,
        &[0.0],
        "point convex areas",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.solidities", &[point_contours.arg()])?)?,
        &[0.0],
        "point solidities",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.circularities", &[point_contours.arg()])?)?,
        &[0.0],
        "point circularities",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.extents", &[point_contours.arg()])?)?,
        &[0.0],
        "point extents",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.centroids", &[point_contours.arg()])?)?,
        &[1.0, 1.0],
        "point centroids",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.hu_moments", &[point_contours.arg()])?)?,
        &[0.0; 7],
        "point Hu moments",
    )?;
    expect_close_floats(
        &expect_floats(helper.call(
            "opencv.contours.minimum_area_boxes",
            &[point_contours.arg()],
        )?)?,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "point minimum-area boxes",
    )?;
    expect_circle(
        &expect_floats(helper.call(
            "opencv.contours.minimum_enclosing_circles",
            &[point_contours.arg()],
        )?)?,
        1.0,
        1.0,
        0.0,
        0.0,
        "point minimum enclosing circle",
    )?;
    expect_close_floats(
        &expect_floats(helper.call(
            "opencv.contours.minimum_enclosing_triangles",
            &[point_contours.arg()],
        )?)?,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "point minimum enclosing triangle",
    )?;
    expect_close_floats(
        &expect_floats(helper.call("opencv.contours.fitted_lines", &[point_contours.arg()])?)?,
        &[0.0, 0.0, 1.0, 1.0],
        "point fitted line",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.fitted_ellipses", &[point_contours.arg()])?)?,
        &[1.0, 1.0, 0.0, 0.0, 0.0],
        "point fitted ellipse",
    )?;

    let line = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(&[0, 0, 0, 255, 255, 255, 0, 0, 0]),
                int_arg(3),
                int_arg(3),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )?;
    let line_contours = expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[line.arg()])?,
        CONTOURS_TYPE,
    )?;
    expect_close_floats(
        &expect_floats(helper.call("opencv.contours.minimum_area_boxes", &[line_contours.arg()])?)?,
        &[0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0],
        "line minimum-area boxes",
    )?;
    expect_circle(
        &expect_floats(helper.call(
            "opencv.contours.minimum_enclosing_circles",
            &[line_contours.arg()],
        )?)?,
        1.0,
        1.0,
        1.0,
        1.001,
        "line minimum enclosing circle",
    )?;
    expect_close_floats(
        &expect_floats(helper.call(
            "opencv.contours.minimum_enclosing_triangles",
            &[line_contours.arg()],
        )?)?,
        &[0.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "line minimum enclosing triangle",
    )?;
    expect_close_floats(
        &expect_floats(helper.call("opencv.contours.fitted_lines", &[line_contours.arg()])?)?,
        &[1.0, 0.0, 1.0, 1.0],
        "line fitted line",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.fitted_ellipses", &[line_contours.arg()])?)?,
        &[1.0, 1.0, 0.0, 0.0, 0.0],
        "line fitted ellipse",
    )?;

    expect_error_code(
        helper.call("opencv.contours.count", &[mask.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.mat.rows", &[contours.arg()])?,
        "handle_type_mismatch",
    )?;
    let empty = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(3), int_arg(3), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    let empty_contours = expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[empty.arg()])?,
        CONTOURS_TYPE,
    )?;
    expect_int(
        helper.call("opencv.contours.count", &[empty_contours.arg()])?,
        0,
    )?;
    expect_exact(
        expect_ints(helper.call("opencv.contours.points", &[empty_contours.arg()])?)?,
        &[],
        "empty points",
    )?;
    expect_exact(
        expect_ints(helper.call("opencv.contours.offsets", &[empty_contours.arg()])?)?,
        &[0],
        "empty offsets",
    )?;
    expect_exact(
        expect_ints(helper.call("opencv.contours.bounds", &[empty_contours.arg()])?)?,
        &[],
        "empty bounds",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.areas", &[empty_contours.arg()])?)?,
        &[],
        "empty areas",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.convex_areas", &[empty_contours.arg()])?)?,
        &[],
        "empty convex areas",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.solidities", &[empty_contours.arg()])?)?,
        &[],
        "empty solidities",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.circularities", &[empty_contours.arg()])?)?,
        &[],
        "empty circularities",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.extents", &[empty_contours.arg()])?)?,
        &[],
        "empty extents",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.centroids", &[empty_contours.arg()])?)?,
        &[],
        "empty centroids",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.hu_moments", &[empty_contours.arg()])?)?,
        &[],
        "empty Hu moments",
    )?;
    expect_exact_floats(
        expect_floats(helper.call(
            "opencv.contours.minimum_area_boxes",
            &[empty_contours.arg()],
        )?)?,
        &[],
        "empty minimum-area boxes",
    )?;
    expect_exact_floats(
        expect_floats(helper.call(
            "opencv.contours.minimum_enclosing_circles",
            &[empty_contours.arg()],
        )?)?,
        &[],
        "empty minimum enclosing circles",
    )?;
    expect_exact_floats(
        expect_floats(helper.call(
            "opencv.contours.minimum_enclosing_triangles",
            &[empty_contours.arg()],
        )?)?,
        &[],
        "empty minimum enclosing triangles",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.fitted_lines", &[empty_contours.arg()])?)?,
        &[],
        "empty fitted lines",
    )?;
    expect_exact_floats(
        expect_floats(helper.call("opencv.contours.fitted_ellipses", &[empty_contours.arg()])?)?,
        &[],
        "empty fitted ellipses",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.match_scores",
            &[empty_contours.arg(), int_arg(0)],
        )?,
        "opencv.invalid_argument",
    )?;

    expect_error_code(
        helper.call("opencv.contours.convex_areas", &[mask.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.match_scores",
            &[contours.arg_with_owner("foreign-worker"), int_arg(0)],
        )?,
        "cross_owner_handle",
    )?;

    reject_invalid_inputs(helper)?;
    for value in [
        &multi_contours,
        &line_contours,
        &point_contours,
        &concave_contours,
        &empty_contours,
        &contours,
    ] {
        expect_unit(helper.call("opencv.contours.dispose", &[value.arg()])?)?;
        expect_error_code(
            helper.call("opencv.contours.match_scores", &[value.arg(), int_arg(0)])?,
            "stale_handle",
        )?;
    }
    for value in [&multi_shape, &line, &point, &concave, &empty, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[value.arg()])?)?;
    }
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires invalid color and oversized inputs to fail before analysis.
fn reject_invalid_inputs(helper: &mut NativeHelper) -> Result<(), String> {
    for (rows, cols, layout) in [(1, 1, "bgr"), (1025, 1024, "gray")] {
        let mat = expect_typed_handle(
            helper.call(
                "opencv.mat.zeros",
                &[int_arg(rows), int_arg(cols), atom_arg(layout)],
            )?,
            MAT_TYPE,
        )?;
        expect_error_code(
            helper.call("opencv.imgproc.find_external_contours", &[mat.arg()])?,
            "opencv.invalid_argument",
        )?;
        expect_unit(helper.call("opencv.mat.dispose", &[mat.arg()])?)?;
    }
    Ok(())
}

/// Requires one exact copied integer projection.
fn expect_exact(actual: Vec<i64>, expected: &[i64], kind: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("unexpected contour {kind}: {actual:?}"))
    }
}

/// Requires one exact copied floating-point projection.
fn expect_exact_floats(actual: Vec<f64>, expected: &[f64], kind: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("unexpected contour {kind}: {actual:?}"))
    }
}

/// Requires one copied floating-point projection within native precision.
fn expect_close_floats(actual: &[f64], expected: &[f64], kind: &str) -> Result<(), String> {
    if actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() <= 1.0e-4)
    {
        Ok(())
    } else {
        Err(format!("unexpected contour {kind}: {actual:?}"))
    }
}

/// Requires one copied circle with a center and bounded native radius.
fn expect_circle(
    actual: &[f64],
    expected_x: f64,
    expected_y: f64,
    minimum_radius: f64,
    maximum_radius: f64,
    kind: &str,
) -> Result<(), String> {
    if actual.len() == 3
        && (actual[0] - expected_x).abs() <= 1.0e-4
        && (actual[1] - expected_y).abs() <= 1.0e-4
        && (minimum_radius..=maximum_radius).contains(&actual[2])
    {
        Ok(())
    } else {
        Err(format!("unexpected contour {kind}: {actual:?}"))
    }
}

/// Returns the absolute area of one flattened three-vertex triangle.
fn triangle_area(points: &[f64]) -> f64 {
    if points.len() != 6 {
        return 0.0;
    }
    ((points[0] * (points[3] - points[5])
        + points[2] * (points[5] - points[1])
        + points[4] * (points[1] - points[3]))
        / 2.0)
        .abs()
}

/// Requires one finite area-bearing copied triangle.
fn expect_area_triangle(actual: &[f64], kind: &str) -> Result<(), String> {
    if actual.len() == 6
        && actual.iter().all(|value| value.is_finite())
        && triangle_area(actual) > 0.0
    {
        Ok(())
    } else {
        Err(format!("unexpected contour {kind}: {actual:?}"))
    }
}

/// Requires one finite canonical unit-direction fitted line.
fn expect_fitted_line(actual: &[f64], kind: &str) -> Result<(), String> {
    if actual.len() == 4
        && actual.iter().all(|value| value.is_finite())
        && (actual[0].hypot(actual[1]) - 1.0).abs() <= 1.0e-3
    {
        Ok(())
    } else {
        Err(format!("unexpected contour {kind}: {actual:?}"))
    }
}
