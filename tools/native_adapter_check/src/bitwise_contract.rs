//! Immutable full-layout binary bitwise contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper, MAT_TYPE,
};

const FIRST_BYTES: &[u8] = &[0, 15, 240, 170, 85, 255];
const SECOND_BYTES: &[u8] = &[255, 51, 15, 85, 170, 0];

/// Runs bitwise truth-table, compatibility, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST_BYTES, 2, 3, "gray")?;
    let second = matrix(helper, SECOND_BYTES, 2, 3, "gray")?;
    let and_result = bitwise(helper, "opencv.imgproc.bitwise_and", &first, &second)?;
    let or_result = bitwise(helper, "opencv.imgproc.bitwise_or", &first, &second)?;
    let xor_result = bitwise(helper, "opencv.imgproc.bitwise_xor", &first, &second)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[and_result.arg()])?,
        &[0, 3, 0, 0, 0, 0],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[or_result.arg()])?,
        &[255, 63, 255, 255, 255, 255],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[xor_result.arg()])?,
        &[255, 60, 255, 255, 255, 255],
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[first.arg()])?,
        FIRST_BYTES,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[second.arg()])?,
        SECOND_BYTES,
    )?;

    for operation in ["opencv.imgproc.bitwise_and", "opencv.imgproc.bitwise_or"] {
        let identity = bitwise(helper, operation, &first, &first)?;
        expect_bytes(
            helper.call("opencv.mat.to_u8", &[identity.arg()])?,
            FIRST_BYTES,
        )?;
        expect_unit(helper.call("opencv.mat.dispose", &[identity.arg()])?)?;
    }
    let self_xor = bitwise(helper, "opencv.imgproc.bitwise_xor", &first, &first)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[self_xor.arg()])?, &[0; 6])?;

    let bgr_first = matrix(helper, &[0, 15, 240], 1, 1, "bgr")?;
    let bgr_second = matrix(helper, &[255, 51, 15], 1, 1, "bgr")?;
    let bgr_xor = bitwise(
        helper,
        "opencv.imgproc.bitwise_xor",
        &bgr_first,
        &bgr_second,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgr_xor.arg()])?,
        &[255, 60, 255],
    )?;
    let bgra_first = matrix(helper, &[170, 85, 255, 0], 1, 1, "bgra")?;
    let bgra_second = matrix(helper, &[85, 170, 0, 255], 1, 1, "bgra")?;
    let bgra_and = bitwise(
        helper,
        "opencv.imgproc.bitwise_and",
        &bgra_first,
        &bgra_second,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[bgra_and.arg()])?, &[0; 4])?;

    let wrong_shape = matrix(helper, &[0; 4], 2, 2, "gray")?;
    let wrong_layout = matrix(helper, &[0; 18], 2, 3, "bgr")?;
    for operation in [
        "opencv.imgproc.bitwise_and",
        "opencv.imgproc.bitwise_or",
        "opencv.imgproc.bitwise_xor",
    ] {
        for invalid in [&wrong_shape, &wrong_layout] {
            expect_error_code(
                helper.call(operation, &[first.arg(), invalid.arg()])?,
                "opencv.invalid_argument",
            )?;
        }
        expect_error_code(
            helper.call(
                operation,
                &[first.arg(), second.arg_with_owner("foreign-worker")],
            )?,
            "cross_owner_handle",
        )?;
        expect_error_code(
            helper.call(
                operation,
                &[first.arg(), second.arg_with_type("other.Package.Resource")],
            )?,
            "handle_type_mismatch",
        )?;
    }
    let stale = matrix(helper, &[0; 6], 2, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.bitwise_and", &[first.arg(), stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_layout,
        &wrong_shape,
        &bgra_and,
        &bgra_second,
        &bgra_first,
        &bgr_xor,
        &bgr_second,
        &bgr_first,
        &self_xor,
        &xor_result,
        &or_result,
        &and_result,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Applies one generated binary bitwise operation and checks its result type.
fn bitwise(
    helper: &mut NativeHelper,
    operation: &str,
    first: &NativeHandle,
    second: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(operation, &[first.arg(), second.arg()])?,
        MAT_TYPE,
    )
}
