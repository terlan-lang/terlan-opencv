//! Typed grayscale image-comparison mask contract.

use super::{
    atom_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    matrix, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const FIRST: &[u8] = &[0, 10, 20, 30];
const SECOND: &[u8] = &[0, 20, 10, 30];

/// Runs all comparison modes, rejection paths, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST, 2, 2, "gray")?;
    let second = matrix(helper, SECOND, 2, 2, "gray")?;
    for (mode, expected) in [
        ("equal", [255, 0, 0, 255]),
        ("not_equal", [0, 255, 255, 0]),
        ("less", [0, 255, 0, 0]),
        ("less_or_equal", [255, 255, 0, 255]),
        ("greater", [0, 0, 255, 0]),
        ("greater_or_equal", [255, 0, 255, 255]),
    ] {
        let mask = compare(helper, &first, &second, mode)?;
        expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, &expected)?;
        expect_unit(helper.call("opencv.mat.dispose", &[mask.arg()])?)?;
    }
    expect_bytes(helper.call("opencv.mat.to_u8", &[first.arg()])?, FIRST)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[second.arg()])?, SECOND)?;

    let alias_equal = compare(helper, &first, &first, "equal")?;
    let alias_not_equal = compare(helper, &first, &first, "not_equal")?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[alias_equal.arg()])?,
        &[255; 4],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[alias_not_equal.arg()])?,
        &[0; 4],
    )?;

    expect_error_code(
        helper.call(
            "opencv.imgproc.compare_mask",
            &[first.arg(), second.arg(), atom_arg("invented")],
        )?,
        "invalid_enum_value",
    )?;
    let wrong_shape = matrix(helper, &[0; 2], 1, 2, "gray")?;
    let color = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    for other in [&wrong_shape, &color] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.compare_mask",
                &[first.arg(), other.arg(), atom_arg("equal")],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.compare_mask",
            &[
                first.arg(),
                second.arg_with_owner("foreign-worker"),
                atom_arg("equal"),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.compare_mask",
            &[
                first.arg(),
                second.arg_with_type("other.Package.Resource"),
                atom_arg("equal"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 4], 2, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.compare_mask",
            &[first.arg(), stale.arg(), atom_arg("equal")],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &color,
        &wrong_shape,
        &alias_not_equal,
        &alias_equal,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Compares two images through one generated semantic mode.
fn compare(
    helper: &mut NativeHelper,
    first: &NativeHandle,
    second: &NativeHandle,
    mode: &str,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.compare_mask",
            &[first.arg(), second.arg(), atom_arg(mode)],
        )?,
        MAT_TYPE,
    )
}
