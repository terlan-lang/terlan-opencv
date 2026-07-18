//! Bounded copied standard Hough-line contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, int_arg, matrix,
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

/// Runs output, validation, handle, source-integrity, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask = matrix(helper, LINE_MASK, 7, 7, "gray")?;
    let lines =
        expect_floats(helper.call("opencv.imgproc.detect_lines", &[mask.arg(), int_arg(3)])?)?;
    validate_lines(&lines, 7, 7)?;
    if !lines.chunks_exact(2).any(|line| {
        (line[0] - 3.0).abs() <= 1.0 && (line[1] - std::f64::consts::FRAC_PI_2).abs() <= 0.15
    }) {
        return Err(format!("expected horizontal line, received {lines:?}"));
    }
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, LINE_MASK)?;

    let blank = matrix(helper, &[0; 49], 7, 7, "gray")?;
    if !expect_floats(helper.call("opencv.imgproc.detect_lines", &[blank.arg(), int_arg(1)])?)?
        .is_empty()
    {
        return Err("blank image produced lines".to_owned());
    }
    for threshold in [0, 8] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.detect_lines",
                &[mask.arg(), int_arg(threshold)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    let color = matrix(helper, &[0; 147], 7, 7, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.detect_lines", &[invalid.arg(), int_arg(1)])?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.detect_lines",
            &[mask.arg_with_owner("foreign-worker"), int_arg(1)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.detect_lines",
            &[mask.arg_with_type("other.Package.Resource"), int_arg(1)],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, LINE_MASK, 7, 7, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.detect_lines", &[stale.arg(), int_arg(1)])?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &blank, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires finite bounded rho/theta pairs in canonical theta/rho order.
fn validate_lines(lines: &[f64], rows: i64, columns: i64) -> Result<(), String> {
    let maximum_rho = (((rows - 1).pow(2) + (columns - 1).pow(2)) as f64).sqrt() + 1.0;
    let pairs = lines.chunks_exact(2).collect::<Vec<_>>();
    let valid_values = !pairs.is_empty()
        && lines.len().is_multiple_of(2)
        && pairs.iter().all(|line| {
            line[0].is_finite()
                && line[0].abs() <= maximum_rho
                && line[1].is_finite()
                && (0.0..std::f64::consts::PI).contains(&line[1])
        });
    let canonical = pairs.windows(2).all(|pair| {
        pair[0][1] < pair[1][1] || (pair[0][1] == pair[1][1] && pair[0][0] <= pair[1][0])
    });
    if valid_values && canonical {
        Ok(())
    } else {
        Err(format!("invalid canonical lines: {lines:?}"))
    }
}
