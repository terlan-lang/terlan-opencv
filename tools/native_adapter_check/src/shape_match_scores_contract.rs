//! Fixed binary-mask Hu-invariant shape-match score contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

const FIRST_SHAPE: &[u8] = &[1, 0, 0, 0, 2, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0];
const EQUIVALENT_SHAPE: &[u8] = &[255, 0, 0, 0, 127, 0, 0, 0, 63, 31, 15, 0, 0, 0, 0, 0];
const DIFFERENT_SHAPE: &[u8] = &[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0];
const TRANSLATED_SHAPE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];

/// Runs identical, intensity-independent, dissimilar, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST_SHAPE, 4, 4, "gray")?;
    assert_eq!(scores(helper, &first, &first)?, [0.0, 0.0, 0.0]);

    let equivalent = matrix(helper, EQUIVALENT_SHAPE, 4, 4, "gray")?;
    assert_eq!(scores(helper, &first, &equivalent)?, [0.0, 0.0, 0.0]);
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[first.arg()])?,
        FIRST_SHAPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[equivalent.arg()])?,
        EQUIVALENT_SHAPE,
    )?;

    let different = matrix(helper, DIFFERENT_SHAPE, 4, 4, "gray")?;
    let different_scores = scores(helper, &first, &different)?;
    if different_scores.iter().all(|score| *score <= 1.0e-6) {
        return Err(format!(
            "different masks produced zero shape scores {different_scores:?}"
        ));
    }
    let translated = matrix(helper, TRANSLATED_SHAPE, 6, 6, "gray")?;
    validate_scores(&scores(helper, &first, &translated)?)?;

    let empty = matrix(helper, &[0; 16], 4, 4, "gray")?;
    for arguments in [[first.arg(), empty.arg()], [empty.arg(), first.arg()]] {
        expect_error_code(
            helper.call("opencv.imgproc.shape_match_scores", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    let bgr = matrix(helper, &[0; 48], 4, 4, "bgr")?;
    let bgra = matrix(helper, &[0; 64], 4, 4, "bgra")?;
    let oversized = matrix(helper, &vec![1; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&bgr, &bgra, &oversized] {
        for arguments in [[first.arg(), image.arg()], [image.arg(), first.arg()]] {
            expect_error_code(
                helper.call("opencv.imgproc.shape_match_scores", &arguments)?,
                "opencv.invalid_argument",
            )?;
        }
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.shape_match_scores",
            &[first.arg(), equivalent.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.shape_match_scores",
            &[
                first.arg(),
                equivalent.arg_with_type("other.Package.Resource"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, EQUIVALENT_SHAPE, 4, 4, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.shape_match_scores",
            &[first.arg(), stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &oversized,
        &bgra,
        &bgr,
        &empty,
        &translated,
        &different,
        &equivalent,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Copies and validates one fixed three-score comparison.
fn scores(
    helper: &mut NativeHelper,
    first: &super::NativeHandle,
    second: &super::NativeHandle,
) -> Result<Vec<f64>, String> {
    let values = expect_floats(helper.call(
        "opencv.imgproc.shape_match_scores",
        &[first.arg(), second.arg()],
    )?)?;
    validate_scores(&values)?;
    Ok(values)
}

/// Requires exactly three finite nonnegative shape-match scores.
fn validate_scores(values: &[f64]) -> Result<(), String> {
    if values.len() != 3
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(format!("invalid shape-match scores {values:?}"));
    }
    Ok(())
}
