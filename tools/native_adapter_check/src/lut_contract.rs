//! Copied unsigned-byte lookup-table contract.

use super::{
    bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    matrix, NativeHandle, NativeHelper, MAT_TYPE,
};

const SOURCE: &[u8] = &[0, 1, 2, 127, 128, 129, 253, 254, 255];
const INVERTED: &[u8] = &[255, 254, 253, 128, 127, 126, 2, 1, 0];

/// Runs lookup-table behavior, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, SOURCE, 3, 3, "gray")?;
    let mut table = inversion_table();
    let transformed = apply_lut(helper, &source, &table)?;
    table.fill(0);
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[transformed.arg()])?,
        INVERTED,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[source.arg()])?, SOURCE)?;

    for (layout, bytes) in [("bgr", 3_usize), ("bgra", 4_usize)] {
        let color = matrix(helper, &vec![0; bytes], 1, 1, layout)?;
        let inverted = apply_lut(helper, &color, &inversion_table())?;
        expect_bytes(
            helper.call("opencv.mat.to_u8", &[inverted.arg()])?,
            &vec![255; bytes],
        )?;
        expect_unit(helper.call("opencv.mat.dispose", &[inverted.arg()])?)?;
        expect_unit(helper.call("opencv.mat.dispose", &[color.arg()])?)?;
    }

    for table in [vec![], vec![0; 255], vec![0; 257]] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.apply_lut",
                &[source.arg(), bytes_arg(&table)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.apply_lut",
            &[
                source.arg_with_owner("foreign-worker"),
                bytes_arg(&inversion_table()),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.apply_lut",
            &[
                source.arg_with_type("other.Package.Resource"),
                bytes_arg(&inversion_table()),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, SOURCE, 3, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.apply_lut",
            &[stale.arg(), bytes_arg(&inversion_table())],
        )?,
        "stale_handle",
    )?;

    expect_unit(helper.call("opencv.mat.dispose", &[transformed.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Applies one copied inversion table and validates the returned resource type.
fn apply_lut(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    table: &[u8],
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.apply_lut", &[image.arg(), bytes_arg(table)])?,
        MAT_TYPE,
    )
}

/// Builds the complete unsigned-byte inversion table.
fn inversion_table() -> Vec<u8> {
    (0_u8..=u8::MAX).rev().collect()
}
