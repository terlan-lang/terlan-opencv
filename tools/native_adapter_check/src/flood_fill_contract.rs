//! Immutable exact-value flood-fill contract.

use super::{
    atom_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    int_arg, matrix, NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const GRAY: &[u8] = &[1, 0, 1, 0, 1, 0, 1, 0, 1];

/// Runs connectivity, layout, validation, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let gray = matrix(helper, GRAY, 3, 3, "gray")?;
    let four = fill(helper, &gray, 1, 1, 9, "four")?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[four.arg()])?,
        &[1, 0, 1, 0, 9, 0, 1, 0, 1],
    )?;
    let eight = fill(helper, &gray, 1, 1, 9, "eight")?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[eight.arg()])?,
        &[9, 0, 9, 0, 9, 0, 9, 0, 9],
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[gray.arg()])?, GRAY)?;

    let bgr = matrix(helper, &[1, 2, 3, 1, 2, 3], 1, 2, "bgr")?;
    let filled_bgr = fill(helper, &bgr, 0, 0, 7, "four")?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[filled_bgr.arg()])?,
        &[7, 7, 7, 7, 7, 7],
    )?;
    let bgra = matrix(helper, &[1, 2, 3, 4], 1, 1, "bgra")?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.flood_fill",
            &[
                bgra.arg(),
                int_arg(0),
                int_arg(0),
                int_arg(8),
                atom_arg("eight"),
            ],
        )?,
        "opencv.invalid_argument",
    )?;

    for args in [
        [int_arg(-1), int_arg(0), int_arg(0), atom_arg("four")],
        [int_arg(3), int_arg(0), int_arg(0), atom_arg("four")],
        [int_arg(0), int_arg(-1), int_arg(0), atom_arg("four")],
        [int_arg(0), int_arg(3), int_arg(0), atom_arg("four")],
        [int_arg(0), int_arg(0), int_arg(-1), atom_arg("four")],
        [int_arg(0), int_arg(0), int_arg(256), atom_arg("four")],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.flood_fill",
                &[
                    gray.arg(),
                    args[0].clone(),
                    args[1].clone(),
                    args[2].clone(),
                    args[3].clone(),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.flood_fill",
            &[
                gray.arg(),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                atom_arg("unsupported"),
            ],
        )?,
        "invalid_enum_value",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.flood_fill",
            &[
                gray.arg_with_owner("foreign-worker"),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                atom_arg("four"),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.flood_fill",
            &[
                gray.arg_with_type("other.Package.Resource"),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                atom_arg("four"),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, GRAY, 3, 3, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.flood_fill",
            &[
                stale.arg(),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                atom_arg("four"),
            ],
        )?,
        "stale_handle",
    )?;

    for handle in [&bgra, &filled_bgr, &bgr, &eight, &four, &gray] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Flood-fills one image through the generated immutable operation.
fn fill(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    seed_x: i64,
    seed_y: i64,
    fill_value: i64,
    connectivity: &str,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.imgproc.flood_fill",
            &[
                image.arg(),
                int_arg(seed_x),
                int_arg(seed_y),
                int_arg(fill_value),
                atom_arg(connectivity),
            ],
        )?,
        MAT_TYPE,
    )
}
