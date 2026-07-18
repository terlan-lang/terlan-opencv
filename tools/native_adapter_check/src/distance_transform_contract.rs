//! Saturated Manhattan mask distance-transform contract.

use super::{
    expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit, matrix,
    NativeHandle, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const MASK: &[u8] = &[
    0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 255, 5, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0,
];
const DISTANCES: &[u8] = &[
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
];

/// Runs exact distance, nonbinary-mask, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    let mask = matrix(helper, MASK, 5, 5, "gray")?;
    let distances = transform(helper, &mask)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[distances.arg()])?,
        DISTANCES,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, MASK)?;

    let mut wide_mask = vec![255; 513];
    wide_mask[0] = 0;
    wide_mask[512] = 0;
    let wide = matrix(helper, &wide_mask, 1, 513, "gray")?;
    let wide_distances = transform(helper, &wide)?;
    let expected_wide: Vec<u8> = (0..513)
        .map(|column| column.min(512 - column).min(255) as u8)
        .collect();
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[wide_distances.arg()])?,
        &expected_wide,
    )?;

    let zero = matrix(helper, &[0; 9], 3, 3, "gray")?;
    let zero_distances = transform(helper, &zero)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[zero_distances.arg()])?,
        &[0; 9],
    )?;
    let bgr = matrix(helper, &[0; 75], 5, 5, "bgr")?;
    let bgra = matrix(helper, &[0; 100], 5, 5, "bgra")?;
    for image in [&bgr, &bgra] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.manhattan_distance_transform",
                &[image.arg()],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.manhattan_distance_transform",
            &[mask.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.manhattan_distance_transform",
            &[mask.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = matrix(helper, MASK, 5, 5, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.manhattan_distance_transform",
            &[stale.arg()],
        )?,
        "stale_handle",
    )?;

    for handle in [
        &bgra,
        &bgr,
        &zero_distances,
        &zero,
        &wide_distances,
        &wide,
        &distances,
        &mask,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Applies the generated Manhattan mask distance transform.
fn transform(helper: &mut NativeHelper, mask: &NativeHandle) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.manhattan_distance_transform", &[mask.arg()])?,
        MAT_TYPE,
    )
}
