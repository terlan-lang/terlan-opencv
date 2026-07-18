//! Semantic image-mask and borrowed-handle contract.

use super::{
    atom_arg, expect_bytes, expect_error_code, expect_int, expect_ints, expect_typed_handle,
    expect_unit, int_arg, matrix, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const SOURCE: &[u8] = &[0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0];
const MASK: &[u8] = &[255, 0, 0, 255];
const MASKED: &[u8] = &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0];

/// Runs deterministic masking, aliasing, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let image = matrix(helper, SOURCE, 2, 2, "bgr")?;
    let mask = matrix(helper, MASK, 2, 2, "gray")?;
    let masked = expect_typed_handle(
        helper.call("opencv.imgproc.apply_mask", &[image.arg(), mask.arg()])?,
        MAT_TYPE,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[masked.arg()])?, MASKED)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, SOURCE)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, MASK)?;

    let grayscale = matrix(helper, &[0, 10, 20, 30], 2, 2, "gray")?;
    let aliased = expect_typed_handle(
        helper.call(
            "opencv.imgproc.apply_mask",
            &[grayscale.arg(), grayscale.arg()],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased.arg()])?,
        &[0, 10, 20, 30],
    )?;
    let range_mask = expect_typed_handle(
        helper.call(
            "opencv.imgproc.in_range",
            &[grayscale.arg(), int_arg(10), int_arg(20)],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[range_mask.arg()])?,
        &[0, 255, 255, 0],
    )?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_points", &[range_mask.arg()])?)?,
        [1, 0, 0, 1]
    );
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.non_zero_points", &[mask.arg()])?)?,
        [0, 0, 1, 1]
    );
    let exact_mask = expect_typed_handle(
        helper.call(
            "opencv.imgproc.in_range",
            &[grayscale.arg(), int_arg(20), int_arg(20)],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[exact_mask.arg()])?,
        &[0, 0, 255, 0],
    )?;
    let mask_xor = expect_typed_handle(
        helper.call(
            "opencv.imgproc.mask_xor",
            &[range_mask.arg(), exact_mask.arg()],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[mask_xor.arg()])?,
        &[0, 255, 0, 0],
    )?;
    let mask_and = expect_typed_handle(
        helper.call("opencv.imgproc.mask_and", &[range_mask.arg(), mask.arg()])?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[mask_and.arg()])?,
        &[0, 0, 0, 0],
    )?;
    assert!(
        expect_ints(helper.call("opencv.imgproc.non_zero_points", &[mask_and.arg()])?)?.is_empty()
    );
    let mask_or = expect_typed_handle(
        helper.call("opencv.imgproc.mask_or", &[range_mask.arg(), mask.arg()])?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[mask_or.arg()])?,
        &[255, 255, 255, 255],
    )?;
    let mask_not = expect_typed_handle(
        helper.call("opencv.imgproc.mask_not", &[range_mask.arg()])?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[mask_not.arg()])?,
        &[255, 0, 0, 255],
    )?;
    for operation in ["opencv.imgproc.mask_and", "opencv.imgproc.mask_or"] {
        let same = expect_typed_handle(
            helper.call(operation, &[range_mask.arg(), range_mask.arg()])?,
            MAT_TYPE,
        )?;
        expect_bytes(
            helper.call("opencv.mat.to_u8", &[same.arg()])?,
            &[0, 255, 255, 0],
        )?;
        expect_unit(helper.call("opencv.mat.dispose", &[same.arg()])?)?;
    }
    let same_mask_xor = expect_typed_handle(
        helper.call(
            "opencv.imgproc.mask_xor",
            &[range_mask.arg(), range_mask.arg()],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[same_mask_xor.arg()])?,
        &[0, 0, 0, 0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[grayscale.arg()])?,
        &[0, 10, 20, 30],
    )?;
    let range_selected = expect_typed_handle(
        helper.call(
            "opencv.imgproc.apply_mask",
            &[image.arg(), range_mask.arg()],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[range_selected.arg()])?,
        &[0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 0, 0],
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.in_range",
            &[image.arg(), int_arg(0), int_arg(255)],
        )?,
        "opencv.invalid_argument",
    )?;
    for (lower, upper) in [(-1, 20), (10, 256), (21, 20)] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.in_range",
                &[grayscale.arg(), int_arg(lower), int_arg(upper)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let wrong_shape = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(1), int_arg(2), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    let color_mask = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(2), int_arg(2), atom_arg("bgr")],
        )?,
        MAT_TYPE,
    )?;
    for invalid_mask in [&wrong_shape, &color_mask] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.apply_mask",
                &[image.arg(), invalid_mask.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for operation in [
        "opencv.imgproc.mask_and",
        "opencv.imgproc.mask_or",
        "opencv.imgproc.mask_xor",
    ] {
        for invalid_mask in [&wrong_shape, &color_mask] {
            expect_error_code(
                helper.call(operation, &[range_mask.arg(), invalid_mask.arg()])?,
                "opencv.invalid_argument",
            )?;
        }
    }
    expect_error_code(
        helper.call("opencv.imgproc.mask_not", &[color_mask.arg()])?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_points", &[color_mask.arg()])?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.apply_mask",
            &[image.arg(), mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.apply_mask",
            &[image.arg(), mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;

    let stale = matrix(helper, MASK, 2, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.apply_mask", &[image.arg(), stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mask_and",
            &[range_mask.arg(), mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mask_or",
            &[
                range_mask.arg(),
                mask.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mask_xor",
            &[range_mask.arg(), mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mask_xor",
            &[
                range_mask.arg(),
                mask.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.mask_and", &[range_mask.arg(), stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.mask_xor", &[range_mask.arg(), stale.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_points",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.non_zero_points",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.non_zero_points", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &color_mask,
        &wrong_shape,
        &same_mask_xor,
        &mask_not,
        &mask_or,
        &mask_and,
        &mask_xor,
        &range_selected,
        &exact_mask,
        &range_mask,
        &aliased,
        &grayscale,
        &masked,
        &mask,
        &image,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
