//! Immutable adaptive grayscale histogram-equalization contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, float_arg,
    int_arg, matrix, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs CLAHE shape, validation, ownership, handle, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source_bytes = core::array::from_fn::<_, 64, _>(|index| (index * 4) as u8);
    let source = matrix(helper, &source_bytes, 8, 8, "gray")?;
    let equalized = expect_typed_handle(
        helper.call(
            "opencv.imgproc.adaptive_equalize_hist",
            &[source.arg(), float_arg(2.0), int_arg(2), int_arg(2)],
        )?,
        MAT_TYPE,
    )?;
    expect_int(helper.call("opencv.mat.rows", &[equalized.arg()])?, 8)?;
    expect_int(helper.call("opencv.mat.cols", &[equalized.arg()])?, 8)?;
    expect_int(helper.call("opencv.mat.channels", &[equalized.arg()])?, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        &source_bytes,
    )?;

    for parameters in [
        (0.0, 1, 1),
        (-1.0, 1, 1),
        (257.0, 1, 1),
        (f64::NAN, 1, 1),
        (f64::INFINITY, 1, 1),
        (2.0, 0, 1),
        (2.0, 1, 0),
        (2.0, 9, 1),
        (2.0, 1, 9),
        (2.0, 65, 1),
        (2.0, 1, 65),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.adaptive_equalize_hist",
                &[
                    source.arg(),
                    float_arg(parameters.0),
                    int_arg(parameters.1),
                    int_arg(parameters.2),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let color = matrix(helper, &[0; 8 * 8 * 3], 8, 8, "bgr")?;
    let oversized = matrix(helper, &vec![0; 1025 * 1024], 1025, 1024, "gray")?;
    for invalid in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.adaptive_equalize_hist",
                &[invalid.arg(), float_arg(2.0), int_arg(1), int_arg(1)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_equalize_hist",
            &[
                source.arg_with_owner("foreign-worker"),
                float_arg(2.0),
                int_arg(1),
                int_arg(1),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_equalize_hist",
            &[
                source.arg_with_type("other.Package.Resource"),
                float_arg(2.0),
                int_arg(1),
                int_arg(1),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, &[0; 4], 2, 2, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.adaptive_equalize_hist",
            &[stale.arg(), float_arg(2.0), int_arg(1), int_arg(1)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &equalized, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
