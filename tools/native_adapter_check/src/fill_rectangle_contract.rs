//! Immutable contained grayscale rectangle-fill contract.

use super::{
    drawing_contract_support::{draw, draw_arguments},
    expect_bytes, expect_error_code, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs rectangle fill, validation, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source_bytes = [1; 30];
    let source = matrix(helper, &source_bytes, 5, 6, "gray")?;
    let filled = draw(
        helper,
        "opencv.imgproc.fill_rectangle",
        &source,
        [1, 1, 4, 3, 7],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filled.arg()])?,
        &[
            1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 1, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1,
            1,
        ],
    )?;
    let erased = draw(
        helper,
        "opencv.imgproc.fill_rectangle",
        &source,
        [1, 1, 4, 3, 0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[erased.arg()])?,
        &[
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            1,
        ],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;

    let point_source = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let point = draw(
        helper,
        "opencv.imgproc.fill_rectangle",
        &point_source,
        [2, 1, 1, 1, 9],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[point.arg()])?,
        &[0, 0, 0, 0, 0, 9, 0, 0, 0],
    )?;

    for parameters in [
        [-1, 0, 1, 1, 0],
        [6, 0, 1, 1, 0],
        [0, -1, 1, 1, 0],
        [0, 5, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 7, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 6, 0],
        [5, 0, 2, 1, 0],
        [0, 4, 1, 2, 0],
        [0, 0, 1, 1, -1],
        [0, 0, 1, 1, 256],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.fill_rectangle",
                &draw_arguments(source.arg(), parameters),
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let color = matrix(helper, &[0; 90], 5, 6, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.fill_rectangle",
                &draw_arguments(invalid.arg(), [0, 0, 1, 1, 0]),
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.fill_rectangle",
            &draw_arguments(source.arg_with_owner("foreign-worker"), [0, 0, 1, 1, 0]),
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.fill_rectangle",
            &draw_arguments(
                source.arg_with_type("other.Package.Resource"),
                [0, 0, 1, 1, 0],
            ),
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.fill_rectangle",
            &draw_arguments(stale.arg(), [0, 0, 1, 1, 0]),
        )?,
        "stale_handle",
    )?;

    for handle in [
        &oversized,
        &color,
        &point,
        &point_source,
        &erased,
        &filled,
        &source,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
