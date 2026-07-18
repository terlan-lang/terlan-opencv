//! Precise copied Euclidean mask distance-transform contract.

use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, matrix, NativeHelper,
};

/// Runs exact distance, copied-output, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask_bytes = [
        0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 255, 5, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0,
    ];
    let mask = matrix(helper, &mask_bytes, 5, 5, "gray")?;
    expect_close(
        &expect_floats(helper.call("opencv.imgproc.euclidean_distance_transform", &[mask.arg()])?)?,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, &mask_bytes)?;

    let diagonal_bytes = [0, 255, 255, 255, 255, 255, 255, 255, 255];
    let diagonal = matrix(helper, &diagonal_bytes, 3, 3, "gray")?;
    expect_close(
        &expect_floats(helper.call(
            "opencv.imgproc.euclidean_distance_transform",
            &[diagonal.arg()],
        )?)?,
        &[
            0.0,
            1.0,
            2.0,
            1.0,
            2.0_f64.sqrt(),
            5.0_f64.sqrt(),
            2.0,
            5.0_f64.sqrt(),
            8.0_f64.sqrt(),
        ],
    )?;
    let zero = matrix(helper, &[0; 9], 3, 3, "gray")?;
    expect_close(
        &expect_floats(helper.call("opencv.imgproc.euclidean_distance_transform", &[zero.arg()])?)?,
        &[0.0; 9],
    )?;

    let all_nonzero = matrix(helper, &[255; 9], 3, 3, "gray")?;
    let bgr = matrix(helper, &[0; 27], 3, 3, "bgr")?;
    let bgra = matrix(helper, &[0; 36], 3, 3, "bgra")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for image in [&all_nonzero, &bgr, &bgra, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.euclidean_distance_transform",
                &[image.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.euclidean_distance_transform",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.euclidean_distance_transform",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &mask_bytes, 5, 5, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.euclidean_distance_transform",
            &[stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &oversized,
        &bgra,
        &bgr,
        &all_nonzero,
        &zero,
        &diagonal,
        &mask,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires copied distances to match within bounded single-precision roundoff.
fn expect_close(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-5)
    {
        return Err(format!(
            "expected distances {expected:?}, received {actual:?}"
        ));
    }
    Ok(())
}
