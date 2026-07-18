//! Immutable three-input mask-selection contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const TRUE_BGR: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const FALSE_BGR: &[u8] = &[101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112];

/// Runs exact selection, aliasing, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let when_true = matrix(helper, TRUE_BGR, 2, 2, "bgr")?;
    let when_false = matrix(helper, FALSE_BGR, 2, 2, "bgr")?;
    let mask = matrix(helper, &[255, 0, 1, 0], 2, 2, "gray")?;
    let selected = select(helper, &when_true, &when_false, &mask)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[selected.arg()])?,
        &[1, 2, 3, 104, 105, 106, 7, 8, 9, 110, 111, 112],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[when_true.arg()])?,
        TRUE_BGR,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[when_false.arg()])?,
        FALSE_BGR,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[mask.arg()])?,
        &[255, 0, 1, 0],
    )?;

    let zero_mask = matrix(helper, &[0; 4], 2, 2, "gray")?;
    let full_mask = matrix(helper, &[255; 4], 2, 2, "gray")?;
    let all_false = select(helper, &when_true, &when_false, &zero_mask)?;
    let all_true = select(helper, &when_true, &when_false, &full_mask)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[all_false.arg()])?,
        FALSE_BGR,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[all_true.arg()])?,
        TRUE_BGR,
    )?;
    let aliased_branches = select(helper, &when_true, &when_true, &mask)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased_branches.arg()])?,
        TRUE_BGR,
    )?;

    let gray_true = matrix(helper, &[0, 2], 1, 2, "gray")?;
    let gray_false = matrix(helper, &[101, 102], 1, 2, "gray")?;
    let aliased_mask = select(helper, &gray_true, &gray_false, &gray_true)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased_mask.arg()])?,
        &[101, 2],
    )?;
    let bgra_true = matrix(helper, &[1, 2, 3, 4], 1, 1, "bgra")?;
    let bgra_false = matrix(helper, &[101, 102, 103, 104], 1, 1, "bgra")?;
    let bgra_mask = matrix(helper, &[255], 1, 1, "gray")?;
    let bgra_selected = select(helper, &bgra_true, &bgra_false, &bgra_mask)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_selected.arg()])?,
        &[1, 2, 3, 4],
    )?;

    let wrong_shape_image = matrix(helper, &[0; 6], 1, 2, "bgr")?;
    let wrong_layout_image = matrix(helper, &[0; 4], 2, 2, "gray")?;
    let wrong_shape_mask = matrix(helper, &[0; 2], 1, 2, "gray")?;
    let color_mask = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    for invalid_image in [&wrong_shape_image, &wrong_layout_image] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.select_mask",
                &[when_true.arg(), invalid_image.arg(), mask.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for invalid_mask in [&wrong_shape_mask, &color_mask] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.select_mask",
                &[when_true.arg(), when_false.arg(), invalid_mask.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (other, mask_arg, code) in [
        (
            when_false.arg_with_owner("foreign-worker"),
            mask.arg(),
            "cross_owner_handle",
        ),
        (
            when_false.arg_with_type("other.Package.Resource"),
            mask.arg(),
            "handle_type_mismatch",
        ),
        (
            when_false.arg(),
            mask.arg_with_owner("foreign-worker"),
            "cross_owner_handle",
        ),
        (
            when_false.arg(),
            mask.arg_with_type("other.Package.Resource"),
            "handle_type_mismatch",
        ),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.select_mask",
                &[when_true.arg(), other, mask_arg],
            )?,
            code,
        )?;
    }
    let stale_image = matrix(helper, FALSE_BGR, 2, 2, "bgr")?;
    let stale_mask = matrix(helper, &[0; 4], 2, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale_image.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale_mask.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.select_mask",
            &[when_true.arg(), stale_image.arg(), mask.arg()],
        )?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.select_mask",
            &[when_true.arg(), when_false.arg(), stale_mask.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &color_mask,
        &wrong_shape_mask,
        &wrong_layout_image,
        &wrong_shape_image,
        &bgra_selected,
        &bgra_mask,
        &bgra_false,
        &bgra_true,
        &aliased_mask,
        &gray_false,
        &gray_true,
        &aliased_branches,
        &all_true,
        &all_false,
        &full_mask,
        &zero_mask,
        &selected,
        &mask,
        &when_false,
        &when_true,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Selects between two images through one generated mask operation.
fn select(
    helper: &mut NativeHelper,
    when_true: &NativeHandle,
    when_false: &NativeHandle,
    mask: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.select_mask",
            &[when_true.arg(), when_false.arg(), mask.arg()],
        )?,
        MAT_TYPE,
    )
}
