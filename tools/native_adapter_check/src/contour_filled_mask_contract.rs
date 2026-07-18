//! Contour-to-mask rasterization contract.

use super::{
    expect_bytes, expect_bytes_value, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, int_arg, matrix, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const CONTOURS_TYPE: &str = "opencv.Contours.Contours";
const RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
];
const FILLED_RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0,
];
const OUTLINED_RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 6, 6, 6, 0, 0, 6, 0, 6, 0, 0, 6, 6, 6, 0, 0, 0, 0, 0, 0,
];
const MULTI_POINT: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Runs exact, degenerate, empty, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    let image = matrix(helper, RECTANGLE, 5, 5, "gray")?;
    let contours = external_contours(helper, &image)?;
    let filled = render(helper, &contours, 9)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 2)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filled.arg()])?,
        FILLED_RECTANGLE,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, RECTANGLE)?;

    let zero = render(helper, &contours, 0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[zero.arg()])?, &[0; 25])?;
    let outlined = outline(helper, &contours, 6, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[outlined.arg()])?,
        OUTLINED_RECTANGLE,
    )?;

    let point_image = matrix(helper, &[0, 1, 0], 1, 3, "gray")?;
    let point = external_contours(helper, &point_image)?;
    let point_mask = render(helper, &point, 5)?;
    let point_outline = outline(helper, &point, 5, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[point_mask.arg()])?,
        &[0, 5, 0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[point_outline.arg()])?,
        &[0, 5, 0],
    )?;

    let line_image = matrix(helper, &[1, 1, 1], 1, 3, "gray")?;
    let line = external_contours(helper, &line_image)?;
    let line_mask = render(helper, &line, 7)?;
    let line_outline = outline(helper, &line, 7, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[line_mask.arg()])?,
        &[7, 7, 7],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[line_outline.arg()])?,
        &[7, 7, 7],
    )?;

    let empty_image = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let empty = external_contours(helper, &empty_image)?;
    let empty_mask = render(helper, &empty, 255)?;
    let empty_outline = outline(helper, &empty, 255, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[empty_mask.arg()])?,
        &[0; 9],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[empty_outline.arg()])?,
        &[0; 9],
    )?;

    let multi_image = matrix(helper, MULTI_POINT, 3, 7, "gray")?;
    let multi = external_contours(helper, &multi_image)?;
    expect_int(helper.call("opencv.contours.count", &[multi.arg()])?, 2)?;
    let multi_mask = render(helper, &multi, 11)?;
    let selected_first = select(helper, &multi, 0, 11)?;
    let selected_second = select(helper, &multi, 1, 11)?;
    validate_selection(
        &expect_bytes_value(helper.call("opencv.mat.to_u8", &[multi_mask.arg()])?)?,
        &expect_bytes_value(helper.call("opencv.mat.to_u8", &[selected_first.arg()])?)?,
        &expect_bytes_value(helper.call("opencv.mat.to_u8", &[selected_second.arg()])?)?,
    )?;

    for value in [-1, 256] {
        expect_error_code(
            helper.call(
                "opencv.contours.filled_mask",
                &[contours.arg(), int_arg(value)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (value, thickness) in [(-1, 1), (256, 1), (9, 0), (9, 256)] {
        expect_error_code(
            helper.call(
                "opencv.contours.outline_mask",
                &[contours.arg(), int_arg(value), int_arg(thickness)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (handle, index, value) in [
        (&contours, -1, 9),
        (&contours, 1, 9),
        (&empty, 0, 9),
        (&contours, 0, 256),
    ] {
        expect_error_code(
            helper.call(
                "opencv.contours.selected_filled_mask",
                &[handle.arg(), int_arg(index), int_arg(value)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.contours.filled_mask",
            &[contours.arg_with_owner("foreign-worker"), int_arg(9)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.outline_mask",
            &[
                contours.arg_with_owner("foreign-worker"),
                int_arg(9),
                int_arg(1),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.filled_mask",
            &[contours.arg_with_type("other.Package.Resource"), int_arg(9)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.selected_filled_mask",
            &[
                contours.arg_with_type("other.Package.Resource"),
                int_arg(0),
                int_arg(9),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = external_contours(helper, &image)?;
    expect_unit(helper.call("opencv.contours.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.contours.filled_mask", &[stale.arg(), int_arg(9)])?,
        "stale_handle",
    )?;

    for handle in [&multi, &empty, &line, &point, &contours] {
        expect_unit(helper.call("opencv.contours.dispose", &[handle.arg()])?)?;
    }
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filled.arg()])?,
        FILLED_RECTANGLE,
    )?;
    for handle in [
        &empty_mask,
        &empty_outline,
        &selected_second,
        &selected_first,
        &multi_mask,
        &line_outline,
        &line_mask,
        &point_outline,
        &point_mask,
        &outlined,
        &zero,
        &filled,
        &multi_image,
        &empty_image,
        &line_image,
        &point_image,
        &image,
    ] {
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

/// Renders one contour owner into a fresh grayscale matrix.
fn render(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
    value: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.contours.filled_mask",
            &[contours.arg(), int_arg(value)],
        )?,
        MAT_TYPE,
    )
}

/// Renders every contour outline into a fresh grayscale matrix.
fn outline(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
    value: i64,
    thickness: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.contours.outline_mask",
            &[contours.arg(), int_arg(value), int_arg(thickness)],
        )?,
        MAT_TYPE,
    )
}

/// Renders one selected contour into a fresh grayscale matrix.
fn select(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
    index: i64,
    value: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.contours.selected_filled_mask",
            &[contours.arg(), int_arg(index), int_arg(value)],
        )?,
        MAT_TYPE,
    )
}

/// Requires two disjoint selected masks to reconstruct the complete mask.
fn validate_selection(all: &[u8], first: &[u8], second: &[u8]) -> Result<(), String> {
    if all.len() != first.len()
        || all.len() != second.len()
        || first.iter().filter(|value| **value != 0).count() != 1
        || second.iter().filter(|value| **value != 0).count() != 1
        || first
            .iter()
            .zip(second)
            .any(|(left, right)| *left != 0 && *right != 0)
        || all
            .iter()
            .zip(first.iter().zip(second))
            .any(|(combined, (left, right))| *combined != (*left | *right))
    {
        return Err(format!(
            "unexpected selected masks: all={all:?}, first={first:?}, second={second:?}"
        ));
    }
    Ok(())
}
