//! Bounded Shi-Tomasi feature-detection contract.

use super::{
    atom_arg, bytes_arg, expect_bytes, expect_error_code, expect_floats, expect_int,
    expect_typed_handle, expect_unit, float_arg, int_arg, NativeHelper, CORNER_IMAGE,
};

const MAT_TYPE: &str = "opencv.Mat.Mat";
/// Runs point-copy, source-integrity, bound, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let mask = expect_typed_handle(
        helper.call(
            "opencv.mat.from_u8",
            &[
                bytes_arg(CORNER_IMAGE),
                int_arg(7),
                int_arg(7),
                atom_arg("gray"),
            ],
        )?,
        MAT_TYPE,
    )?;
    let coordinates = expect_floats(helper.call(
        "opencv.imgproc.good_features_to_track",
        &[mask.arg(), int_arg(8), float_arg(0.01), float_arg(1.0)],
    )?)?;
    validate_coordinates(&coordinates, 1, 8, 7.0, 7.0)?;
    let moments = expect_floats(helper.call("opencv.imgproc.hu_moments", &[mask.arg()])?)?;
    validate_hu_moments(&moments, false)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[mask.arg()])?,
        CORNER_IMAGE,
    )?;

    let one = expect_floats(helper.call(
        "opencv.imgproc.good_features_to_track",
        &[mask.arg(), int_arg(1), float_arg(0.01), float_arg(0.0)],
    )?)?;
    validate_coordinates(&one, 1, 1, 7.0, 7.0)?;

    let empty = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(7), int_arg(7), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    let empty_coordinates = expect_floats(helper.call(
        "opencv.imgproc.good_features_to_track",
        &[empty.arg(), int_arg(8), float_arg(0.01), float_arg(1.0)],
    )?)?;
    if !empty_coordinates.is_empty() {
        return Err(format!(
            "zero mask produced feature coordinates {empty_coordinates:?}"
        ));
    }
    let empty_moments = expect_floats(helper.call("opencv.imgproc.hu_moments", &[empty.arg()])?)?;
    validate_hu_moments(&empty_moments, true)?;

    let color = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(2), int_arg(2), atom_arg("bgr")],
        )?,
        MAT_TYPE,
    )?;
    let oversized = expect_typed_handle(
        helper.call(
            "opencv.mat.zeros",
            &[int_arg(1025), int_arg(1024), atom_arg("gray")],
        )?,
        MAT_TYPE,
    )?;
    for handle in [&color, &oversized] {
        expect_error_code(
            helper.call("opencv.imgproc.hu_moments", &[handle.arg()])?,
            "opencv.invalid_argument",
        )?;
    }
    for arguments in [
        vec![color.arg(), int_arg(1), float_arg(0.01), float_arg(1.0)],
        vec![oversized.arg(), int_arg(1), float_arg(0.01), float_arg(1.0)],
        vec![mask.arg(), int_arg(0), float_arg(0.01), float_arg(1.0)],
        vec![mask.arg(), int_arg(4097), float_arg(0.01), float_arg(1.0)],
        vec![mask.arg(), int_arg(1), float_arg(0.0), float_arg(1.0)],
        vec![mask.arg(), int_arg(1), float_arg(1.01), float_arg(1.0)],
        vec![mask.arg(), int_arg(1), float_arg(f64::NAN), float_arg(1.0)],
        vec![
            mask.arg(),
            int_arg(1),
            float_arg(f64::INFINITY),
            float_arg(1.0),
        ],
        vec![mask.arg(), int_arg(1), float_arg(0.01), float_arg(-0.1)],
        vec![
            mask.arg(),
            int_arg(1),
            float_arg(0.01),
            float_arg(1_000_001.0),
        ],
        vec![mask.arg(), int_arg(1), float_arg(0.01), float_arg(f64::NAN)],
        vec![
            mask.arg(),
            int_arg(1),
            float_arg(0.01),
            float_arg(f64::INFINITY),
        ],
    ] {
        expect_error_code(
            helper.call("opencv.imgproc.good_features_to_track", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }

    for handle in [&oversized, &color, &empty, &mask] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Validates flattened finite coordinates without relying on detector ordering.
fn validate_coordinates(
    coordinates: &[f64],
    minimum_points: usize,
    maximum_points: usize,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let point_count = coordinates.len() / 2;
    if !coordinates.len().is_multiple_of(2)
        || !(minimum_points..=maximum_points).contains(&point_count)
    {
        return Err(format!(
            "expected {minimum_points}..={maximum_points} feature points, received {coordinates:?}"
        ));
    }
    for point in coordinates.chunks_exact(2) {
        if !point[0].is_finite()
            || !point[1].is_finite()
            || point[0] < 0.0
            || point[0] >= width
            || point[1] < 0.0
            || point[1] >= height
        {
            return Err(format!("invalid feature coordinates {coordinates:?}"));
        }
    }
    Ok(())
}

/// Validates the fixed Hu-invariant shape and empty-mask behavior.
fn validate_hu_moments(values: &[f64], empty: bool) -> Result<(), String> {
    if values.len() != 7 || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("invalid Hu moments {values:?}"));
    }
    if empty && values.iter().any(|value| *value != 0.0) {
        return Err(format!("empty mask produced Hu moments {values:?}"));
    }
    if !empty && (values[0] <= 0.0 || values[0] >= 1.0) {
        return Err(format!("nonempty mask produced Hu moments {values:?}"));
    }
    Ok(())
}
