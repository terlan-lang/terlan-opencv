//! Bounded canonical probabilistic Hough line-segment contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_ints, expect_unit, int_arg, matrix,
    NativeHelper,
};

#[rustfmt::skip]
const LINE_MASK: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
];

/// Runs deterministic segment, validation, handle, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    assert_eq!(LINE_MASK.len(), 49);
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask = matrix(helper, LINE_MASK, 7, 7, "gray")?;
    let segments = expect_ints(helper.call(
        "opencv.imgproc.line_segments",
        &[mask.arg(), int_arg(3), int_arg(4), int_arg(0)],
    )?)?;
    expect_segments(&segments, &[1, 3, 5, 3])?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, LINE_MASK)?;

    let binary_bytes = LINE_MASK
        .iter()
        .map(|value| if *value == 0 { 0 } else { 255 })
        .collect::<Vec<_>>();
    let binary = matrix(helper, &binary_bytes, 7, 7, "gray")?;
    let binary_segments = expect_ints(helper.call(
        "opencv.imgproc.line_segments",
        &[binary.arg(), int_arg(3), int_arg(4), int_arg(0)],
    )?)?;
    expect_segments(&binary_segments, &segments)?;

    let empty = matrix(helper, &[0; 49], 7, 7, "gray")?;
    if !expect_ints(helper.call(
        "opencv.imgproc.line_segments",
        &[empty.arg(), int_arg(1), int_arg(0), int_arg(0)],
    )?)?
    .is_empty()
    {
        return Err("empty mask produced line segments".to_owned());
    }

    for arguments in [
        vec![mask.arg(), int_arg(0), int_arg(0), int_arg(0)],
        vec![mask.arg(), int_arg(8), int_arg(0), int_arg(0)],
        vec![mask.arg(), int_arg(1), int_arg(-1), int_arg(0)],
        vec![mask.arg(), int_arg(1), int_arg(9), int_arg(0)],
        vec![mask.arg(), int_arg(1), int_arg(0), int_arg(-1)],
        vec![mask.arg(), int_arg(1), int_arg(0), int_arg(9)],
    ] {
        expect_error_code(
            helper.call("opencv.imgproc.line_segments", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    let bgr = matrix(helper, &[0; 147], 7, 7, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&bgr, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.line_segments",
                &[invalid.arg(), int_arg(1), int_arg(0), int_arg(0)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.line_segments",
            &[
                mask.arg_with_owner("foreign-worker"),
                int_arg(1),
                int_arg(0),
                int_arg(0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.line_segments",
            &[
                mask.arg_with_type("other.Package.Resource"),
                int_arg(1),
                int_arg(0),
                int_arg(0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, LINE_MASK, 7, 7, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.line_segments",
            &[stale.arg(), int_arg(1), int_arg(0), int_arg(0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &bgr, &empty, &binary, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires canonical flattened segment endpoints.
fn expect_segments(actual: &[i64], expected: &[i64]) -> Result<(), String> {
    if actual == expected && actual.len().is_multiple_of(4) {
        Ok(())
    } else {
        Err(format!("unexpected line segments: {actual:?}"))
    }
}
