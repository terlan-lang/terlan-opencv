//! Checked two-dimensional image-tiling contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, int_arg, matrix,
    NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs exact tiling, count rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let gray = matrix(helper, &[1, 2, 3, 4], 2, 2, "gray")?;
    let gray_tiled = tile(helper, &gray, 2, 3)?;
    expect_shape(helper, &gray_tiled, 4, 6, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray_tiled.arg()])?,
        &[
            1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4,
        ],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray.arg()])?,
        &[1, 2, 3, 4],
    )?;

    let bgr = matrix(helper, &[5, 6, 7], 1, 1, "bgr")?;
    let bgr_tiled = tile(helper, &bgr, 2, 2)?;
    expect_shape(helper, &bgr_tiled, 2, 2, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgr_tiled.arg()])?,
        &[5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7],
    )?;

    let bgra = matrix(helper, &[1, 2, 3, 4, 5, 6, 7, 8], 1, 2, "bgra")?;
    let bgra_tiled = tile(helper, &bgra, 2, 1)?;
    expect_shape(helper, &bgra_tiled, 2, 2, 4)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_tiled.arg()])?,
        &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
    )?;

    let identity = tile(helper, &gray, 1, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[identity.arg()])?,
        &[1, 2, 3, 4],
    )?;
    for (vertical_count, horizontal_count) in [(0, 1), (1, 0), (-1, 1), (1, -1)] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.tile",
                &[
                    gray.arg(),
                    int_arg(vertical_count),
                    int_arg(horizontal_count),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.tile",
            &[gray.arg(), int_arg(i64::MAX), int_arg(1)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.tile",
            &[
                gray.arg_with_owner("foreign-worker"),
                int_arg(1),
                int_arg(1),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.tile",
            &[
                gray.arg_with_type("other.Package.Resource"),
                int_arg(1),
                int_arg(1),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.tile",
            &[stale.arg(), int_arg(1), int_arg(1)],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &identity,
        &bgra_tiled,
        &bgra,
        &bgr_tiled,
        &bgr,
        &gray_tiled,
        &gray,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Tiles one image and validates fresh matrix ownership.
fn tile(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    vertical_count: i64,
    horizontal_count: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.tile",
            &[
                image.arg(),
                int_arg(vertical_count),
                int_arg(horizontal_count),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Requires one matrix to expose the expected public shape and layout.
fn expect_shape(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    rows: i64,
    cols: i64,
    channels: i64,
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[image.arg()])?, rows)?;
    expect_int(helper.call("opencv.mat.cols", &[image.arg()])?, cols)?;
    expect_int(
        helper.call("opencv.mat.channels", &[image.arg()])?,
        channels,
    )
}
