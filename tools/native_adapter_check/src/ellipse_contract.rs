//! Immutable fully contained grayscale ellipse contract.

use super::{
    drawing_contract_support::{draw, draw_arguments},
    expect_bytes, expect_error_code, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs ellipse outline/fill, validation, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source_bytes = [1; 25];
    let source = matrix(helper, &source_bytes, 5, 5, "gray")?;
    let outlined = draw(
        helper,
        "opencv.imgproc.draw_ellipse",
        &source,
        [2, 2, 1, 1, 9, 1],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[outlined.arg()])?,
        &[
            1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 9, 1, 9, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1,
        ],
    )?;
    let filled = draw(
        helper,
        "opencv.imgproc.fill_ellipse",
        &source,
        [2, 2, 1, 1, 9],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filled.arg()])?,
        &[
            1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1,
        ],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;

    for parameters in [
        [-1, 2, 1, 1, 0, 1],
        [5, 2, 1, 1, 0, 1],
        [2, -1, 1, 1, 0, 1],
        [2, 5, 1, 1, 0, 1],
        [2, 2, 0, 1, 0, 1],
        [2, 2, 1, 0, 0, 1],
        [2, 2, 5, 1, 0, 1],
        [2, 2, 1, 5, 0, 1],
        [0, 2, 1, 1, 0, 1],
        [2, 0, 1, 1, 0, 1],
        [2, 2, 1, 1, -1, 1],
        [2, 2, 1, 1, 256, 1],
        [2, 2, 1, 1, 0, 0],
        [2, 2, 1, 1, 0, 256],
        [2, 2, 1, 1, 0, 5],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.draw_ellipse",
                &draw_arguments(source.arg(), parameters),
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for parameters in [
        [-1, 2, 1, 1, 0],
        [5, 2, 1, 1, 0],
        [2, -1, 1, 1, 0],
        [2, 5, 1, 1, 0],
        [2, 2, 0, 1, 0],
        [2, 2, 1, 0, 0],
        [2, 2, 5, 1, 0],
        [2, 2, 1, 5, 0],
        [0, 2, 1, 1, 0],
        [2, 0, 1, 1, 0],
        [2, 2, 1, 1, -1],
        [2, 2, 1, 1, 256],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.fill_ellipse",
                &draw_arguments(source.arg(), parameters),
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let color = matrix(helper, &[0; 75], 5, 5, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.draw_ellipse",
                &draw_arguments(invalid.arg(), [2, 2, 1, 1, 0, 1]),
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.imgproc.fill_ellipse",
                &draw_arguments(invalid.arg(), [2, 2, 1, 1, 0]),
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.draw_ellipse",
            &draw_arguments(source.arg_with_owner("foreign-worker"), [2, 2, 1, 1, 0, 1]),
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.fill_ellipse",
            &draw_arguments(
                source.arg_with_type("other.Package.Resource"),
                [2, 2, 1, 1, 0],
            ),
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 9], 3, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.draw_ellipse",
            &draw_arguments(stale.arg(), [1, 1, 1, 1, 0, 1]),
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &filled, &outlined, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
