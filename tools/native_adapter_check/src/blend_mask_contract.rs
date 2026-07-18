//! Bounded per-pixel byte-mask blending contract.

use super::{
    expect_bytes, expect_error_code, expect_handle, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs exact blending, layout, alias, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first_bytes = [200, 200, 200];
    let second_bytes = [100, 100, 100];
    let weight_bytes = [0, 128, 255];
    let first = matrix(helper, &first_bytes, 1, 3, "gray")?;
    let second = matrix(helper, &second_bytes, 1, 3, "gray")?;
    let weights = matrix(helper, &weight_bytes, 1, 3, "gray")?;
    let blended = blend(helper, &first, &second, &weights)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[blended.arg()])?,
        &[100, 150, 200],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[first.arg()])?,
        &first_bytes,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[second.arg()])?,
        &second_bytes,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[weights.arg()])?,
        &weight_bytes,
    )?;

    let bgr_first = matrix(helper, &[200, 100, 0, 255, 0, 100], 1, 2, "bgr")?;
    let bgr_second = matrix(helper, &[0, 100, 200, 0, 200, 100], 1, 2, "bgr")?;
    let bgr_weights = matrix(helper, &[128, 0], 1, 2, "gray")?;
    let bgr_blended = blend(helper, &bgr_first, &bgr_second, &bgr_weights)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgr_blended.arg()])?,
        &[100, 100, 100, 0, 200, 100],
    )?;
    let bgra_first = matrix(helper, &[10, 20, 30, 40], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[90, 80, 70, 60], 1, 1, "bgra")?;
    let bgra_weights = matrix(helper, &[255], 1, 1, "gray")?;
    let bgra_blended = blend(helper, &bgra_first, &bgra_second, &bgra_weights)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_blended.arg()])?,
        &[10, 20, 30, 40],
    )?;
    let aliased = blend(helper, &first, &first, &weights)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased.arg()])?,
        &first_bytes,
    )?;

    let wrong_shape = matrix(helper, &[0; 4], 2, 2, "gray")?;
    let wrong_layout = matrix(helper, &[0; 9], 1, 3, "bgr")?;
    let color_weights = matrix(helper, &[0; 9], 1, 3, "bgr")?;
    for (invalid_second, invalid_weights) in [
        (&wrong_shape, &weights),
        (&wrong_layout, &weights),
        (&second, &wrong_shape),
        (&second, &color_weights),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.blend_mask",
                &[first.arg(), invalid_second.arg(), invalid_weights.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let oversized_first = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    let oversized_second = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    let oversized_weights = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[
                oversized_first.arg(),
                oversized_second.arg(),
                oversized_weights.arg(),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[
                first.arg(),
                second.arg_with_owner("foreign-worker"),
                weights.arg(),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[
                first.arg(),
                second.arg_with_type("other.Package.Resource"),
                weights.arg(),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[
                first.arg(),
                second.arg(),
                weights.arg_with_owner("foreign-worker"),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[
                first.arg(),
                second.arg(),
                weights.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale_second = matrix(helper, &second_bytes, 1, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale_second.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[first.arg(), stale_second.arg(), weights.arg()],
        )?,
        "stale_handle",
    )?;
    let stale_weights = matrix(helper, &weight_bytes, 1, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale_weights.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.blend_mask",
            &[first.arg(), second.arg(), stale_weights.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &oversized_weights,
        &oversized_second,
        &oversized_first,
        &color_weights,
        &wrong_layout,
        &wrong_shape,
        &aliased,
        &bgra_blended,
        &bgra_weights,
        &bgra_second,
        &bgra_first,
        &bgr_blended,
        &bgr_weights,
        &bgr_second,
        &bgr_first,
        &blended,
        &weights,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Blends three validated handles and extracts one fresh matrix owner.
fn blend(
    helper: &mut NativeHelper,
    first: &super::NativeHandle,
    second: &super::NativeHandle,
    weights: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_handle(helper.call(
        "opencv.imgproc.blend_mask",
        &[first.arg(), second.arg(), weights.arg()],
    )?)
}
