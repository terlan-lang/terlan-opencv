//! Copied grayscale integral-image contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_ints, expect_unit, matrix, NativeHelper,
};

/// Runs exact values, bounds, handle validation, source integrity, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;

    let source_bytes = [1, 2, 3, 4, 5, 6];
    let source = matrix(helper, &source_bytes, 2, 3, "gray")?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.integral_values", &[source.arg()])?)?,
        [0, 0, 0, 0, 0, 1, 3, 6, 0, 5, 12, 21]
    );
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;

    let zero = matrix(helper, &[0, 0, 0, 0], 2, 2, "gray")?;
    assert_eq!(
        expect_ints(helper.call("opencv.imgproc.integral_values", &[zero.arg()])?)?,
        [0; 9]
    );

    let color = matrix(helper, &[1, 2, 3], 1, 1, "bgr")?;
    expect_error_code(
        helper.call("opencv.imgproc.integral_values", &[color.arg()])?,
        "opencv.invalid_argument",
    )?;

    let oversized_bytes = vec![0; 1025 * 1024];
    let oversized = matrix(helper, &oversized_bytes, 1025, 1024, "gray")?;
    expect_error_code(
        helper.call("opencv.imgproc.integral_values", &[oversized.arg()])?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.integral_values",
            &[source.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.integral_values",
            &[source.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;

    let stale = matrix(helper, &[255], 1, 1, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.integral_values", &[stale.arg()])?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &zero, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
