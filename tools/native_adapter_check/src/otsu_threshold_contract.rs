//! Automatic Otsu binarization contract.

use super::{
    atom_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    float_arg, matrix, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const SOURCE: &[u8] = &[0, 0, 0, 0, 255, 255, 255, 255];
const BINARY: &[u8] = &[0, 0, 0, 0, 200, 200, 200, 200];
const INVERSE: &[u8] = &[200, 200, 200, 200, 0, 0, 0, 0];

/// Runs exact automatic-threshold, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, SOURCE, 2, 4, "gray")?;
    let binary = apply(helper, &source, 200.0, "binary_threshold")?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[binary.arg()])?, BINARY)?;
    let inverse = apply(helper, &source, 200.0, "binary_inverse")?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[inverse.arg()])?, INVERSE)?;
    let zero_maximum = apply(helper, &source, 0.0, "binary_threshold")?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[zero_maximum.arg()])?,
        &[0; 8],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[source.arg()])?, SOURCE)?;

    let color = matrix(helper, &[0; 24], 2, 4, "bgr")?;
    expect_invalid(helper, &color, 200.0, "binary_threshold")?;
    for maximum in [-0.1, 255.1, f64::NAN, f64::INFINITY] {
        expect_invalid(helper, &source, maximum, "binary_threshold")?;
    }
    for mode in ["truncate", "to_zero", "to_zero_inverse"] {
        expect_invalid(helper, &source, 200.0, mode)?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.otsu_threshold",
            &[source.arg(), float_arg(200.0), atom_arg("invented_mode")],
        )?,
        "invalid_enum_value",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.otsu_threshold",
            &[
                source.arg_with_owner("foreign-worker"),
                float_arg(200.0),
                atom_arg("binary_threshold"),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.otsu_threshold",
            &[
                source.arg_with_type("other.Package.Resource"),
                float_arg(200.0),
                atom_arg("binary_threshold"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, SOURCE, 2, 4, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.otsu_threshold",
            &[stale.arg(), float_arg(200.0), atom_arg("binary_threshold")],
        )?,
        "stale_handle",
    )?;

    for handle in [&color, &zero_maximum, &inverse, &binary, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Applies automatic Otsu thresholding and validates fresh matrix ownership.
fn apply(
    helper: &mut NativeHelper,
    source: &super::NativeHandle,
    maximum: f64,
    mode: &str,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.otsu_threshold",
            &[source.arg(), float_arg(maximum), atom_arg(mode)],
        )?,
        MAT_TYPE,
    )
}

/// Requires one automatic Otsu threshold call to fail argument validation.
fn expect_invalid(
    helper: &mut NativeHelper,
    source: &super::NativeHandle,
    maximum: f64,
    mode: &str,
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.imgproc.otsu_threshold",
            &[source.arg(), float_arg(maximum), atom_arg(mode)],
        )?,
        "opencv.invalid_argument",
    )
}
