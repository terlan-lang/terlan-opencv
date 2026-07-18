//! Multi-resource image arithmetic and borrowed-handle contract.

use super::{
    atom_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    float_arg, int_arg, matrix, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const FIRST: &[u8] = &[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
const SECOND: &[u8] = &[10, 8, 25, 20, 60, 40, 65, 90, 70, 100, 80, 120];
const DIFFERENCE: &[u8] = &[10, 2, 5, 10, 20, 10, 5, 20, 10, 10, 20, 10];
const WEIGHTED_FIRST: &[u8] = &[0, 40, 80, 120, 160, 200, 20, 60, 100, 140, 180, 220];
const WEIGHTED_SECOND: &[u8] = &[120, 80, 40, 0, 40, 80, 220, 180, 140, 100, 60, 20];
const WEIGHTED_RESULT: &[u8] = &[100, 80, 60, 40, 80, 120, 180, 160, 140, 120, 100, 80];

/// Runs deterministic arithmetic, aliasing, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let first = matrix(helper, FIRST, 2, 2, "bgr")?;
    let second = matrix(helper, SECOND, 2, 2, "bgr")?;
    let difference = expect_typed_handle(
        helper.call("opencv.imgproc.abs_diff", &[first.arg(), second.arg()])?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[difference.arg()])?,
        DIFFERENCE,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[first.arg()])?, FIRST)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[second.arg()])?, SECOND)?;

    let aliased = expect_typed_handle(
        helper.call("opencv.imgproc.abs_diff", &[first.arg(), first.arg()])?,
        MAT_TYPE,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[aliased.arg()])?, &[0; 12])?;

    let weighted_first = matrix(helper, WEIGHTED_FIRST, 2, 2, "bgr")?;
    let weighted_second = matrix(helper, WEIGHTED_SECOND, 2, 2, "bgr")?;
    let weighted = expect_typed_handle(
        helper.call(
            "opencv.imgproc.add_weighted",
            &[
                weighted_first.arg(),
                weighted_second.arg(),
                float_arg(0.25),
                float_arg(0.75),
                float_arg(10.0),
            ],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[weighted.arg()])?,
        WEIGHTED_RESULT,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[weighted_first.arg()])?,
        WEIGHTED_FIRST,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[weighted_second.arg()])?,
        WEIGHTED_SECOND,
    )?;
    let aliased_weighted = expect_typed_handle(
        helper.call(
            "opencv.imgproc.add_weighted",
            &[
                weighted_first.arg(),
                weighted_first.arg(),
                float_arg(0.5),
                float_arg(0.5),
                float_arg(0.0),
            ],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[aliased_weighted.arg()])?,
        WEIGHTED_FIRST,
    )?;
    let saturated_weighted = expect_typed_handle(
        helper.call(
            "opencv.imgproc.add_weighted",
            &[
                weighted_first.arg(),
                weighted_second.arg(),
                float_arg(1.0),
                float_arg(1.0),
                float_arg(100.0),
            ],
        )?,
        MAT_TYPE,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[saturated_weighted.arg()])?,
        &[220, 220, 220, 220, 255, 255, 255, 255, 255, 255, 255, 255],
    )?;

    let wrong_shape = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(1), int_arg(2), atom_arg("bgr")],
        )?,
        MAT_TYPE,
    )?;
    let wrong_channels = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(2), int_arg(2), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    for other in [&wrong_shape, &wrong_channels] {
        expect_error_code(
            helper.call("opencv.imgproc.abs_diff", &[first.arg(), other.arg()])?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.imgproc.add_weighted",
                &[
                    first.arg(),
                    other.arg(),
                    float_arg(0.5),
                    float_arg(0.5),
                    float_arg(0.0),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.abs_diff",
            &[first.arg(), second.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.abs_diff",
            &[first.arg(), second.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;

    let stale = matrix(helper, FIRST, 2, 2, "bgr")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call("opencv.imgproc.abs_diff", &[first.arg(), stale.arg()])?,
        "stale_handle",
    )?;
    for (first_weight, second_weight, offset) in [
        (16.01, 0.0, 0.0),
        (0.0, -16.01, 0.0),
        (f64::NAN, 0.0, 0.0),
        (0.0, f64::INFINITY, 0.0),
        (0.0, 0.0, 255.01),
        (0.0, 0.0, f64::NEG_INFINITY),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.add_weighted",
                &[
                    first.arg(),
                    second.arg(),
                    float_arg(first_weight),
                    float_arg(second_weight),
                    float_arg(offset),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.add_weighted",
            &[
                first.arg(),
                second.arg_with_owner("foreign-worker"),
                float_arg(0.5),
                float_arg(0.5),
                float_arg(0.0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.add_weighted",
            &[
                first.arg(),
                stale.arg(),
                float_arg(0.5),
                float_arg(0.5),
                float_arg(0.0),
            ],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &wrong_channels,
        &wrong_shape,
        &saturated_weighted,
        &aliased_weighted,
        &weighted,
        &weighted_second,
        &weighted_first,
        &aliased,
        &difference,
        &second,
        &first,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}
