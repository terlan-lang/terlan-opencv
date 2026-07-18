//! Adaptive grayscale threshold contract.

use super::{
    atom_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    float_arg, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const ZERO_GRAY: &[u8] = &[0; 9];
const MAX_GRAY: &[u8] = &[200; 9];

/// Runs deterministic adaptive-threshold and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let gray = matrix(helper, "gray")?;
    let mean = adaptive(helper, &gray, 200.0, "mean", "binary_threshold", 3, 0.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mean.arg()])?, ZERO_GRAY)?;
    let gaussian = adaptive(helper, &gray, 200.0, "gaussian", "binary_inverse", 3, 0.0)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gaussian.arg()])?,
        MAX_GRAY,
    )?;
    let shifted = adaptive(helper, &gray, 200.0, "mean", "binary_threshold", 3, 1.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[shifted.arg()])?, MAX_GRAY)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[gray.arg()])?, ZERO_GRAY)?;

    let color = matrix(helper, "bgr")?;
    expect_invalid(helper, &color, 200.0, "mean", "binary_threshold", 3, 0.0)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_threshold",
            &[
                gray.arg(),
                float_arg(200.0),
                atom_arg("invented"),
                atom_arg("binary_threshold"),
                int_arg(3),
                float_arg(0.0),
            ],
        )?,
        "invalid_enum_value",
    )?;
    expect_invalid(helper, &gray, 200.0, "mean", "truncate", 3, 0.0)?;
    for (maximum, block_size, constant) in [
        (-0.1, 3, 0.0),
        (255.1, 3, 0.0),
        (f64::NAN, 3, 0.0),
        (f64::INFINITY, 3, 0.0),
        (200.0, 0, 0.0),
        (200.0, 1, 0.0),
        (200.0, 2, 0.0),
        (200.0, 4, 0.0),
        (200.0, 257, 0.0),
        (200.0, 3, -255.1),
        (200.0, 3, 255.1),
        (200.0, 3, f64::NAN),
        (200.0, 3, f64::INFINITY),
    ] {
        expect_invalid(
            helper,
            &gray,
            maximum,
            "mean",
            "binary_threshold",
            block_size,
            constant,
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_threshold",
            &[
                gray.arg_with_owner("foreign-worker"),
                float_arg(200.0),
                atom_arg("mean"),
                atom_arg("binary_threshold"),
                int_arg(3),
                float_arg(0.0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_threshold",
            &[
                gray.arg_with_type("other.Package.Resource"),
                float_arg(200.0),
                atom_arg("mean"),
                atom_arg("binary_threshold"),
                int_arg(3),
                float_arg(0.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_threshold",
            &[
                stale.arg(),
                float_arg(200.0),
                atom_arg("mean"),
                atom_arg("binary_threshold"),
                int_arg(3),
                float_arg(0.0),
            ],
        )?,
        "stale_handle",
    )?;

    for handle in [&color, &shifted, &gaussian, &mean, &gray] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Creates one zero-filled three-by-three image in the requested layout.
fn matrix(helper: &mut NativeHelper, layout: &str) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(3), int_arg(3), atom_arg(layout)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one adaptive threshold and validates its returned matrix type.
fn adaptive(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    maximum: f64,
    method: &str,
    mode: &str,
    block_size: i64,
    constant: f64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.adaptive_threshold",
            &[
                image.arg(),
                float_arg(maximum),
                atom_arg(method),
                atom_arg(mode),
                int_arg(block_size),
                float_arg(constant),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Requires one adaptive-threshold call to fail as an OpenCV argument error.
fn expect_invalid(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    maximum: f64,
    method: &str,
    mode: &str,
    block_size: i64,
    constant: f64,
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_threshold",
            &[
                image.arg(),
                float_arg(maximum),
                atom_arg(method),
                atom_arg(mode),
                int_arg(block_size),
                float_arg(constant),
            ],
        )?,
        "opencv.invalid_argument",
    )
}
