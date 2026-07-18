//! Immutable full-layout byte-inversion contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper, MAT_TYPE,
};

/// Runs gray, BGR, BGRA, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let gray_bytes = [0, 1, 127, 128, 254, 255];
    let gray = matrix(helper, &gray_bytes, 2, 3, "gray")?;
    let inverted = invert(helper, &gray)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[inverted.arg()])?,
        &[255, 254, 128, 127, 1, 0],
    )?;
    let restored = invert(helper, &inverted)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[restored.arg()])?,
        &gray_bytes,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[gray.arg()])?, &gray_bytes)?;

    let bgr_bytes = [0, 10, 20, 255, 128, 64];
    let bgr = matrix(helper, &bgr_bytes, 1, 2, "bgr")?;
    let inverted_bgr = invert(helper, &bgr)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[inverted_bgr.arg()])?,
        &[255, 245, 235, 0, 127, 191],
    )?;
    let bgra = matrix(helper, &[0, 1, 2, 3], 1, 1, "bgra")?;
    let inverted_bgra = invert(helper, &bgra)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[inverted_bgra.arg()])?,
        &[255, 254, 253, 252],
    )?;

    expect_error_code(
        helper.call(
            "opencv.imgproc.invert",
            &[gray.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.invert",
            &[gray.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.invert", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &inverted_bgra,
        &bgra,
        &inverted_bgr,
        &bgr,
        &restored,
        &inverted,
        &gray,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Inverts one image and checks its generated resource type.
fn invert(helper: &mut NativeHelper, image: &NativeHandle) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.invert", &[image.arg()])?,
        MAT_TYPE,
    )
}
