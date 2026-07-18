//! Gaussian image-pyramid contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs deterministic pyramid sizing, layout, ownership, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let gray = matrix(helper, &[64; 16], 4, 4, "gray")?;
    let down = pyramid(helper, "opencv.imgproc.pyr_down", &gray)?;
    expect_shape(helper, &down, 2, 2, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[down.arg()])?, &[64; 4])?;
    let restored = pyramid(helper, "opencv.imgproc.pyr_up", &down)?;
    expect_shape(helper, &restored, 4, 4, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[restored.arg()])?,
        &[64; 16],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[gray.arg()])?, &[64; 16])?;

    let odd_bgr = matrix(helper, &[0; 45], 3, 5, "bgr")?;
    let odd_down = pyramid(helper, "opencv.imgproc.pyr_down", &odd_bgr)?;
    expect_shape(helper, &odd_down, 2, 3, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[odd_down.arg()])?,
        &[0; 18],
    )?;
    let bgra = matrix(helper, &[0; 16], 2, 2, "bgra")?;
    let bgra_up = pyramid(helper, "opencv.imgproc.pyr_up", &bgra)?;
    expect_shape(helper, &bgra_up, 4, 4, 4)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[bgra_up.arg()])?, &[0; 64])?;

    for operation in ["opencv.imgproc.pyr_down", "opencv.imgproc.pyr_up"] {
        expect_error_code(
            helper.call(operation, &[gray.arg_with_owner("foreign-worker")])?,
            "cross_owner_handle",
        )?;
        expect_error_code(
            helper.call(operation, &[gray.arg_with_type("other.Package.Resource")])?,
            "handle_type_mismatch",
        )?;
    }
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    for operation in ["opencv.imgproc.pyr_down", "opencv.imgproc.pyr_up"] {
        expect_error_code(helper.call(operation, &[stale.arg()])?, "stale_handle")?;
    }

    for handle in [
        &bgra_up, &bgra, &odd_down, &odd_bgr, &restored, &down, &gray,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Copies bytes into one image with the requested dimensions and layout.
fn matrix(
    helper: &mut NativeHelper,
    bytes: &[u8],
    rows: i64,
    cols: i64,
    layout: &str,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(bytes),
                int_arg(rows),
                int_arg(cols),
                atom_arg(layout),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Applies one pyramid operation and validates its resource type.
fn pyramid(
    helper: &mut NativeHelper,
    operation: &str,
    image: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(helper.call(operation, &[image.arg()])?, MAT_TYPE)
}

/// Requires one matrix to have the expected public dimensions and channels.
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
