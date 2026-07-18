//! Fixed-arity BGR and BGRA channel-merge contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs exact reconstruction, aliasing, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let blue = matrix(helper, &[1, 2, 3, 4], 2, 2, "gray")?;
    let green = matrix(helper, &[10, 20, 30, 40], 2, 2, "gray")?;
    let red = matrix(helper, &[100, 110, 120, 130], 2, 2, "gray")?;
    let alpha = matrix(helper, &[200, 201, 202, 203], 2, 2, "gray")?;
    let bgr = merge_bgr(helper, &blue, &green, &red)?;
    expect_image(
        helper,
        &bgr,
        3,
        &[1, 10, 100, 2, 20, 110, 3, 30, 120, 4, 40, 130],
    )?;
    let bgra = merge_bgra(helper, &blue, &green, &red, &alpha)?;
    expect_image(
        helper,
        &bgra,
        4,
        &[
            1, 10, 100, 200, 2, 20, 110, 201, 3, 30, 120, 202, 4, 40, 130, 203,
        ],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[green.arg()])?,
        &[10, 20, 30, 40],
    )?;

    let aliased = merge_bgr(helper, &blue, &blue, &blue)?;
    expect_image(helper, &aliased, 3, &[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])?;

    let wrong_shape = matrix(helper, &[0; 4], 1, 4, "gray")?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.merge_bgr",
            &[blue.arg(), green.arg(), wrong_shape.arg()],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.merge_bgr",
            &[blue.arg(), bgr.arg(), red.arg()],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.merge_bgr",
            &[
                blue.arg(),
                green.arg_with_owner("foreign-worker"),
                red.arg(),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.merge_bgr",
            &[
                blue.arg(),
                green.arg(),
                red.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 4], 2, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.merge_bgra",
            &[blue.arg(), green.arg(), red.arg(), stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_shape,
        &aliased,
        &bgra,
        &bgr,
        &alpha,
        &red,
        &green,
        &blue,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Merges three grayscale channels into one fresh BGR owner.
fn merge_bgr(
    helper: &mut NativeHelper,
    blue: &super::NativeHandle,
    green: &super::NativeHandle,
    red: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.merge_bgr",
            &[blue.arg(), green.arg(), red.arg()],
        )?,
        MAT_TYPE,
    )
}

/// Merges four grayscale channels into one fresh BGRA owner.
fn merge_bgra(
    helper: &mut NativeHelper,
    blue: &super::NativeHandle,
    green: &super::NativeHandle,
    red: &super::NativeHandle,
    alpha: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.merge_bgra",
            &[blue.arg(), green.arg(), red.arg(), alpha.arg()],
        )?,
        MAT_TYPE,
    )
}

/// Requires one merged image to have exact 2x2 shape, layout, and bytes.
fn expect_image(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    channels: i64,
    bytes: &[u8],
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[image.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[image.arg()])?, 2)?;
    expect_int(
        helper.call("opencv.mat.channels", &[image.arg()])?,
        channels,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, bytes)
}
