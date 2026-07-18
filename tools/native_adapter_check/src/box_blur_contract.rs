//! Normalized box-filter and stack-blur ownership contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, int_arg, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const IMPULSE: &[u8] = &[0, 0, 0, 0, 90, 0, 0, 0, 0];
const AVERAGED: &[u8] = &[10; 9];

/// Runs box-filter and stack-blur behavior, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let impulse = matrix(helper, IMPULSE, "gray")?;
    let blurred = box_blur(helper, &impulse, 3, 3)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[blurred.arg()])?, AVERAGED)?;
    let stack_identity = stack_blur(helper, &impulse, 1, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[stack_identity.arg()])?,
        IMPULSE,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[impulse.arg()])?, IMPULSE)?;

    let bgr = matrix(helper, &[0; 27], "bgr")?;
    let blurred_bgr = box_blur(helper, &bgr, 1, 3)?;
    let stacked_bgr = stack_blur(helper, &bgr, 3, 3)?;
    expect_int(helper.call("opencv.mat.channels", &[blurred_bgr.arg()])?, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[blurred_bgr.arg()])?,
        &[0; 27],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[stacked_bgr.arg()])?,
        &[0; 27],
    )?;
    let bgra = matrix(helper, &[0; 36], "bgra")?;
    let blurred_bgra = box_blur(helper, &bgra, 3, 1)?;
    let stacked_bgra = stack_blur(helper, &bgra, 3, 3)?;
    expect_int(
        helper.call("opencv.mat.channels", &[blurred_bgra.arg()])?,
        4,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[blurred_bgra.arg()])?,
        &[0; 36],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[stacked_bgra.arg()])?,
        &[0; 36],
    )?;

    for (width, height) in [(0, 3), (2, 3), (3, 0), (3, 4), (257, 3), (3, 257)] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.box_blur",
                &[impulse.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.imgproc.stack_blur",
                &[impulse.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.box_blur",
            &[
                impulse.arg_with_owner("foreign-worker"),
                int_arg(3),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.box_blur",
            &[
                impulse.arg_with_type("other.Package.Resource"),
                int_arg(3),
                int_arg(3),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.stack_blur",
            &[
                impulse.arg_with_owner("foreign-worker"),
                int_arg(3),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.stack_blur",
            &[
                impulse.arg_with_type("other.Package.Resource"),
                int_arg(3),
                int_arg(3),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 9], "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.box_blur",
            &[stale.arg(), int_arg(3), int_arg(3)],
        )?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.stack_blur",
            &[stale.arg(), int_arg(3), int_arg(3)],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &stacked_bgra,
        &blurred_bgra,
        &bgra,
        &stacked_bgr,
        &blurred_bgr,
        &bgr,
        &stack_identity,
        &blurred,
        &impulse,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Creates one three-by-three matrix in a public image layout.
fn matrix(helper: &mut NativeHelper, bytes: &[u8], layout: &str) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(bytes), int_arg(3), int_arg(3), atom_arg(layout)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one box filter and validates its result type.
fn box_blur(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.box_blur",
            &[image.arg(), int_arg(width), int_arg(height)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one stack blur and validates its result type.
fn stack_blur(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    width: i64,
    height: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.stack_blur",
            &[image.arg(), int_arg(width), int_arg(height)],
        )?,
        MAT_TYPE,
    )
}
