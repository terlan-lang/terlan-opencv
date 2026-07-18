//! Connected-component ownership and projection contract.

use super::{
    atom_arg, bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints,
    expect_typed_handle, expect_unit, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const COMPONENTS_TYPE: &str = "opencv.Components.Components";

/// Runs deterministic four/eight-neighbor analysis and lifecycle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.components.live_count", &[])?, 0)?;
    let mask = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(&[0, 255, 0, 0, 255, 0, 0, 0, 255]),
                int_arg(3),
                int_arg(3),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )?;
    let four = expect_typed_handle(
        helper.call(
            "opencv.imgproc.connected_components",
            &[mask.arg(), atom_arg("four")],
        )?,
        COMPONENTS_TYPE,
    )?;
    let eight = expect_typed_handle(
        helper.call(
            "opencv.imgproc.connected_components",
            &[mask.arg(), atom_arg("eight")],
        )?,
        COMPONENTS_TYPE,
    )?;
    expect_int(helper.call("opencv.components.live_count", &[])?, 2)?;
    expect_int(
        helper.call("opencv.components.foreground_count", &[four.arg()])?,
        2,
    )?;
    expect_ints(helper.call("opencv.components.labels", &[four.arg()])?)
        .and_then(|actual| expect_exact(actual, &[0, 1, 0, 0, 1, 0, 0, 0, 2], "four labels"))?;
    expect_ints(helper.call("opencv.components.areas", &[four.arg()])?)
        .and_then(|actual| expect_exact(actual, &[2, 1], "four areas"))?;
    expect_ints(helper.call("opencv.components.bounds", &[four.arg()])?)
        .and_then(|actual| expect_exact(actual, &[1, 0, 1, 2, 2, 2, 1, 1], "four bounds"))?;
    expect_floats(helper.call("opencv.components.centroids", &[four.arg()])?)
        .and_then(|actual| expect_close(&actual, &[1.0, 0.5, 2.0, 2.0], "four centroids"))?;
    expect_int(
        helper.call("opencv.components.foreground_count", &[eight.arg()])?,
        1,
    )?;
    expect_ints(helper.call("opencv.components.labels", &[eight.arg()])?)
        .and_then(|actual| expect_exact(actual, &[0, 1, 0, 0, 1, 0, 0, 0, 1], "eight labels"))?;
    expect_ints(helper.call("opencv.components.areas", &[eight.arg()])?)
        .and_then(|actual| expect_exact(actual, &[3], "eight areas"))?;
    expect_ints(helper.call("opencv.components.bounds", &[eight.arg()])?)
        .and_then(|actual| expect_exact(actual, &[1, 0, 2, 3], "eight bounds"))?;
    expect_floats(helper.call("opencv.components.centroids", &[eight.arg()])?)
        .and_then(|actual| expect_close(&actual, &[4.0 / 3.0, 1.0], "eight centroids"))?;

    let empty_mask = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(2), int_arg(2), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    let empty = expect_typed_handle(
        helper.call(
            "opencv.imgproc.connected_components",
            &[empty_mask.arg(), atom_arg("four")],
        )?,
        COMPONENTS_TYPE,
    )?;
    expect_int(
        helper.call("opencv.components.foreground_count", &[empty.arg()])?,
        0,
    )?;
    if !expect_floats(helper.call("opencv.components.centroids", &[empty.arg()])?)?.is_empty() {
        return Err("empty analysis returned foreground centroids".to_owned());
    }

    expect_error_code(
        helper.call("opencv.components.foreground_count", &[mask.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.mat.rows", &[four.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.connected_components",
            &[mask.arg(), atom_arg("invented")],
        )?,
        "invalid_enum_value",
    )?;

    let color = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(&[0, 0, 0]),
                int_arg(1),
                int_arg(1),
                atom_arg("bgr"),
            ],
        )?,
        MAT_TYPE,
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.connected_components",
            &[color.arg(), atom_arg("four")],
        )?,
        "opencv.invalid_argument",
    )?;
    let oversized = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(1025), int_arg(1024), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.connected_components",
            &[oversized.arg(), atom_arg("eight")],
        )?,
        "opencv.invalid_argument",
    )?;

    for components in [&four, &eight, &empty] {
        expect_unit(helper.call("opencv.components.dispose", &[components.arg()])?)?;
        expect_error_code(
            helper.call("opencv.components.foreground_count", &[components.arg()])?,
            "stale_handle",
        )?;
    }
    for mat in [&oversized, &color, &empty_mask, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[mat.arg()])?)?;
    }
    expect_int(helper.call("opencv.components.live_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Requires copied floats to agree within native centroid precision.
fn expect_close(actual: &[f64], expected: &[f64], kind: &str) -> Result<(), String> {
    if actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() <= 1.0e-9)
    {
        Ok(())
    } else {
        Err(format!("unexpected {kind}: {actual:?}"))
    }
}

/// Requires one exact copied integer projection.
fn expect_exact(actual: Vec<i64>, expected: &[i64], kind: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("unexpected {kind}: {actual:?}"))
    }
}
