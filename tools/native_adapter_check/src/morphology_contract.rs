//! Compound morphology contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle,
    expect_unit, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const IMPULSE: &[u8] = &[0, 0, 0, 0, 255, 0, 0, 0, 0];
const ZERO: &[u8] = &[0; 9];
const FULL: &[u8] = &[255; 9];
const BLACK_HAT: &[u8] = &[255, 255, 255, 255, 0, 255, 255, 255, 255];

/// Runs exact compound morphology, rejection, and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let impulse = matrix(helper, IMPULSE, "gray")?;
    let opened = morphology(helper, &impulse, "open", 3, 3, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[opened.arg()])?, ZERO)?;
    let closed = morphology(helper, &impulse, "close", 3, 3, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[closed.arg()])?, FULL)?;
    let gradient = morphology(helper, &impulse, "gradient", 3, 3, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[gradient.arg()])?, FULL)?;
    let top_hat = morphology(helper, &impulse, "top_hat", 3, 3, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[top_hat.arg()])?, IMPULSE)?;
    let black_hat = morphology(helper, &impulse, "black_hat", 3, 3, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[black_hat.arg()])?,
        BLACK_HAT,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[impulse.arg()])?, IMPULSE)?;

    let bgr = matrix(helper, &[0; 27], "bgr")?;
    let bgr_opened = morphology(helper, &bgr, "open", 3, 3, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgr_opened.arg()])?,
        &[0; 27],
    )?;
    let bgra = matrix(helper, &[0; 36], "bgra")?;
    let bgra_closed = morphology(helper, &bgra, "close", 3, 3, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[bgra_closed.arg()])?,
        &[0; 36],
    )?;

    expect_error_code(
        helper.call(
            "opencv.imgproc.morphology",
            &[
                impulse.arg(),
                atom_arg("invented"),
                int_arg(3),
                int_arg(3),
                int_arg(1),
            ],
        )?,
        "invalid_enum_value",
    )?;
    for (width, height, iterations) in [
        (0, 3, 1),
        (2, 3, 1),
        (257, 3, 1),
        (3, 0, 1),
        (3, 2, 1),
        (3, 257, 1),
        (3, 3, 0),
        (3, 3, 65),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.morphology",
                &[
                    impulse.arg(),
                    atom_arg("open"),
                    int_arg(width),
                    int_arg(height),
                    int_arg(iterations),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.morphology",
            &[
                impulse.arg_with_owner("foreign-worker"),
                atom_arg("open"),
                int_arg(3),
                int_arg(3),
                int_arg(1),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.morphology",
            &[
                impulse.arg_with_type("other.Package.Resource"),
                atom_arg("open"),
                int_arg(3),
                int_arg(3),
                int_arg(1),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, ZERO, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.morphology",
            &[
                stale.arg(),
                atom_arg("open"),
                int_arg(3),
                int_arg(3),
                int_arg(1),
            ],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &bgra_closed,
        &bgra,
        &bgr_opened,
        &bgr,
        &black_hat,
        &top_hat,
        &gradient,
        &closed,
        &opened,
        &impulse,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Copies bytes into one three-by-three image.
fn matrix(
    helper: &mut NativeHelper,
    bytes: &[u8],
    layout: &str,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(bytes), int_arg(3), int_arg(3), atom_arg(layout)],
        )?,
        MAT_TYPE,
    )
}

/// Applies one compound morphology operation and checks its resource type.
fn morphology(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    operation: &str,
    width: i64,
    height: i64,
    iterations: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.morphology",
            &[
                image.arg(),
                atom_arg(operation),
                int_arg(width),
                int_arg(height),
                int_arg(iterations),
            ],
        )?,
        MAT_TYPE,
    )
}
