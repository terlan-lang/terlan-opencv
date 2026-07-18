//! Bounded pyramid mean-shift filtering contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, float_arg, int_arg, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const CONSTANT_BGR: &[u8] = &[
    10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30,
    10, 20, 30,
];

/// Runs mean-shift behavior, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let source = matrix(helper, CONSTANT_BGR, "bgr")?;
    let filtered = mean_shift(helper, &source, 1.0, 1.0, 0)?;
    expect_int(helper.call("opencv.mat.channels", &[filtered.arg()])?, 3)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filtered.arg()])?,
        CONSTANT_BGR,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[source.arg()])?,
        CONSTANT_BGR,
    )?;

    let gray = matrix(helper, &[0; 9], "gray")?;
    let bgra = matrix(helper, &[0; 36], "bgra")?;
    for image in [&gray, &bgra] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.mean_shift_filter",
                &[image.arg(), float_arg(1.0), float_arg(1.0), int_arg(0)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (spatial, color, level) in [
        (0.0, 1.0, 0),
        (-1.0, 1.0, 0),
        (65_536.0, 1.0, 0),
        (f64::NAN, 1.0, 0),
        (f64::INFINITY, 1.0, 0),
        (1.0, 0.0, 0),
        (1.0, -1.0, 0),
        (1.0, 65_536.0, 0),
        (1.0, f64::NAN, 0),
        (1.0, f64::INFINITY, 0),
        (1.0, 1.0, -1),
        (1.0, 1.0, 9),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.mean_shift_filter",
                &[
                    source.arg(),
                    float_arg(spatial),
                    float_arg(color),
                    int_arg(level),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.mean_shift_filter",
            &[
                source.arg_with_owner("foreign-worker"),
                float_arg(1.0),
                float_arg(1.0),
                int_arg(0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mean_shift_filter",
            &[
                source.arg_with_type("other.Package.Resource"),
                float_arg(1.0),
                float_arg(1.0),
                int_arg(0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, CONSTANT_BGR, "bgr")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.mean_shift_filter",
            &[stale.arg(), float_arg(1.0), float_arg(1.0), int_arg(0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&bgra, &gray, &filtered, &source] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Creates one three-by-three matrix in the requested public layout.
fn matrix(helper: &mut NativeHelper, bytes: &[u8], layout: &str) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(bytes), int_arg(3), int_arg(3), atom_arg(layout)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one mean-shift filter and validates its result type.
fn mean_shift(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    spatial_radius: f64,
    color_radius: f64,
    max_level: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.mean_shift_filter",
            &[
                image.arg(),
                float_arg(spatial_radius),
                float_arg(color_radius),
                int_arg(max_level),
            ],
        )?,
        MAT_TYPE,
    )
}
