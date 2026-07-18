//! Signed point-to-contour distance projection contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_floats, expect_int,
    expect_typed_handle, expect_unit, float_arg, int_arg, NativeHelper,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
const CONTOURS_TYPE: &str = "opencv.Contours.Contours";
const RECTANGLE: &[u8] = &[
    0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0,
];

/// Runs signed geometry, degeneracy, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    let image = matrix(helper, RECTANGLE, 5, 5)?;
    let contours = external_contours(helper, &image)?;
    expect_close(&project(helper, &contours, 2.0, 2.0)?, &[1.0], "inside")?;
    expect_close(&project(helper, &contours, 1.0, 2.0)?, &[0.0], "boundary")?;
    expect_close(
        &project(helper, &contours, 0.0, 0.0)?,
        &[-2.0_f64.sqrt()],
        "outside",
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, RECTANGLE)?;

    let point_image = matrix(helper, &[0, 1, 0], 1, 3)?;
    let point = external_contours(helper, &point_image)?;
    expect_close(&project(helper, &point, 1.0, 0.0)?, &[0.0], "point hit")?;
    expect_close(&project(helper, &point, 0.0, 0.0)?, &[-1.0], "point miss")?;

    let line_image = matrix(helper, &[1, 1, 1], 1, 3)?;
    let line = external_contours(helper, &line_image)?;
    expect_close(&project(helper, &line, 1.0, 0.0)?, &[0.0], "line hit")?;

    let empty_image = matrix(helper, &[0; 9], 3, 3)?;
    let empty = external_contours(helper, &empty_image)?;
    if !project(helper, &empty, 1.0, 1.0)?.is_empty() {
        return Err("empty contours produced signed distances".to_owned());
    }

    for (x, y) in [
        (-0.1, 0.0),
        (5.0, 0.0),
        (0.0, -0.1),
        (0.0, 5.0),
        (f64::NAN, 0.0),
        (0.0, f64::INFINITY),
    ] {
        expect_error_code(
            helper.call(
                "opencv.contours.signed_distances",
                &[contours.arg(), float_arg(x), float_arg(y)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.contours.signed_distances",
            &[
                contours.arg_with_owner("foreign-worker"),
                float_arg(0.0),
                float_arg(0.0),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.contours.signed_distances",
            &[
                contours.arg_with_type("other.Package.Resource"),
                float_arg(0.0),
                float_arg(0.0),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    let stale = external_contours(helper, &image)?;
    expect_unit(helper.call("opencv.contours.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.contours.signed_distances",
            &[stale.arg(), float_arg(0.0), float_arg(0.0)],
        )?,
        "stale_handle",
    )?;

    for handle in [&empty, &line, &point, &contours] {
        expect_unit(helper.call("opencv.contours.dispose", &[handle.arg()])?)?;
    }
    for handle in [&empty_image, &line_image, &point_image, &image] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.contours.live_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Creates one bounded grayscale matrix.
fn matrix(
    helper: &mut NativeHelper,
    bytes: &[u8],
    rows: i64,
    cols: i64,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(bytes),
                int_arg(rows),
                int_arg(cols),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )
}

/// Finds one uniquely owned external-contour analysis.
fn external_contours(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
) -> Result<super::NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.imgproc.find_external_contours", &[image.arg()])?,
        CONTOURS_TYPE,
    )
}

/// Copies signed distances from one point to every retained contour.
fn project(
    helper: &mut NativeHelper,
    contours: &super::NativeHandle,
    x: f64,
    y: f64,
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.contours.signed_distances",
        &[contours.arg(), float_arg(x), float_arg(y)],
    )?)
}

/// Requires one copied projection within native floating-point precision.
fn expect_close(actual: &[f64], expected: &[f64], kind: &str) -> Result<(), String> {
    if actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (left - right).abs() <= 1.0e-9)
    {
        Ok(())
    } else {
        Err(format!("unexpected {kind} signed distances: {actual:?}"))
    }
}
