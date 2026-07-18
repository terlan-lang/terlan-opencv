//! Immutable bounded grayscale line-drawing contract.

use super::{
    drawing_contract_support::{draw, draw_arguments},
    expect_bytes, expect_error_code, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs rasterization, ownership, validation, handle, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source_bytes = [1; 15];
    let source = matrix(helper, &source_bytes, 3, 5, "gray")?;
    let horizontal = draw(
        helper,
        "opencv.imgproc.draw_line",
        &source,
        [0, 1, 4, 1, 7, 1],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[horizontal.arg()])?,
        &[1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1],
    )?;
    let erased = draw(
        helper,
        "opencv.imgproc.draw_line",
        &source,
        [0, 1, 4, 1, 0, 1],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[erased.arg()])?,
        &[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;

    let diagonal_source = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let diagonal = draw(
        helper,
        "opencv.imgproc.draw_line",
        &diagonal_source,
        [0, 0, 2, 2, 9, 1],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[diagonal.arg()])?,
        &[9, 0, 0, 0, 9, 0, 0, 0, 9],
    )?;
    let point_source = matrix(helper, &[0], 1, 1, "gray")?;
    let point = draw(
        helper,
        "opencv.imgproc.draw_line",
        &point_source,
        [0, 0, 0, 0, 255, 1],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[point.arg()])?, &[255])?;

    for parameters in [
        [-1, 0, 0, 0, 0, 1],
        [5, 0, 0, 0, 0, 1],
        [0, -1, 0, 0, 0, 1],
        [0, 3, 0, 0, 0, 1],
        [0, 0, -1, 0, 0, 1],
        [0, 0, 5, 0, 0, 1],
        [0, 0, 0, -1, 0, 1],
        [0, 0, 0, 3, 0, 1],
        [0, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, 256, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 256],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.draw_line",
                &draw_arguments(source.arg(), parameters),
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let color = matrix(helper, &[0; 45], 3, 5, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.draw_line",
                &draw_arguments(invalid.arg(), [0, 0, 0, 0, 0, 1]),
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.draw_line",
            &draw_arguments(source.arg_with_owner("foreign-worker"), [0, 0, 0, 0, 0, 1]),
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.draw_line",
            &draw_arguments(
                source.arg_with_type("other.Package.Resource"),
                [0, 0, 0, 0, 0, 1],
            ),
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.draw_line",
            &draw_arguments(stale.arg(), [0, 0, 0, 0, 0, 1]),
        )?,
        "stale_handle",
    )?;

    for handle in [
        &oversized,
        &color,
        &point,
        &point_source,
        &diagonal,
        &diagonal_source,
        &erased,
        &horizontal,
        &source,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
