//! Bounded three-dimensional camera projection contract.

use super::{
    camera_intrinsics_arg, distortion_arg, expect_error_code, expect_floats, float_list_arg,
    vector3_arg, NativeHelper,
};

const OBJECT_POINTS: [f64; 9] = [0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0];
const ZERO_VECTOR: [f64; 3] = [0.0; 3];
const INTRINSICS: [f64; 4] = [2.0, 2.0, 1.0, 1.0];
const ZERO_DISTORTION: [f64; 5] = [0.0; 5];

/// Runs successful projection, deterministic output, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let identity = project(
        helper,
        &OBJECT_POINTS,
        ZERO_VECTOR,
        ZERO_VECTOR,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&identity, &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0])?;
    let rotated = project(
        helper,
        &OBJECT_POINTS,
        [0.0, 0.0, std::f64::consts::FRAC_PI_2],
        ZERO_VECTOR,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&rotated, &[1.0, 1.0, 1.0, 2.0, 0.0, 1.0])?;
    let translated = project(
        helper,
        &OBJECT_POINTS,
        ZERO_VECTOR,
        [0.0, 0.0, 2.0],
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let repeated = project(
        helper,
        &OBJECT_POINTS,
        ZERO_VECTOR,
        [0.0, 0.0, 2.0],
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&translated, &[1.0, 1.0, 1.5, 1.0, 1.0, 1.5])?;
    if translated != repeated {
        return Err("3D point projection was nondeterministic".to_owned());
    }
    if !project(
        helper,
        &[],
        ZERO_VECTOR,
        ZERO_VECTOR,
        INTRINSICS,
        ZERO_DISTORTION,
    )?
    .is_empty()
    {
        return Err("empty 3D projection did not remain empty".to_owned());
    }

    for points in [vec![0.0], vec![0.0, 0.0], vec![0.0; 4]] {
        expect_invalid(
            helper,
            &points,
            ZERO_VECTOR,
            ZERO_VECTOR,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
    }
    expect_invalid(
        helper,
        &vec![0.0; 4097 * 3],
        ZERO_VECTOR,
        ZERO_VECTOR,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut points = OBJECT_POINTS;
        points[0] = invalid;
        expect_invalid(
            helper,
            &points,
            ZERO_VECTOR,
            ZERO_VECTOR,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        expect_invalid(
            helper,
            &OBJECT_POINTS,
            [invalid, 0.0, 0.0],
            ZERO_VECTOR,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        expect_invalid(
            helper,
            &OBJECT_POINTS,
            ZERO_VECTOR,
            [0.0, 0.0, invalid],
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
    }
    expect_invalid(
        helper,
        &OBJECT_POINTS,
        ZERO_VECTOR,
        ZERO_VECTOR,
        [0.0, 2.0, 1.0, 1.0],
        ZERO_DISTORTION,
    )?;
    expect_invalid(
        helper,
        &OBJECT_POINTS,
        ZERO_VECTOR,
        ZERO_VECTOR,
        INTRINSICS,
        [11.0, 0.0, 0.0, 0.0, 0.0],
    )?;
    expect_invalid(
        helper,
        &[1_000_000.0, 0.0, 0.000001],
        ZERO_VECTOR,
        ZERO_VECTOR,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.project_points",
            &arguments(
                "li:0,0,2".to_owned(),
                vector3_arg(ZERO_VECTOR),
                vector3_arg(ZERO_VECTOR),
                camera_intrinsics_arg(INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ),
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.project_points",
            &arguments(
                "lf:1,,2".to_owned(),
                vector3_arg(ZERO_VECTOR),
                vector3_arg(ZERO_VECTOR),
                camera_intrinsics_arg(INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ),
        )?,
        "invalid_argument",
    )
}

/// Projects one copied object-point sequence through the helper protocol.
pub(super) fn project(
    helper: &mut NativeHelper,
    points: &[f64],
    rotation: [f64; 3],
    translation: [f64; 3],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.calib3d.project_points",
        &arguments(
            float_list_arg(points),
            vector3_arg(rotation),
            vector3_arg(translation),
            camera_intrinsics_arg(intrinsics),
            distortion_arg(distortion),
        ),
    )?)
}

/// Requires one invalid projection request to fail finitely.
fn expect_invalid(
    helper: &mut NativeHelper,
    points: &[f64],
    rotation: [f64; 3],
    translation: [f64; 3],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.project_points",
            &arguments(
                float_list_arg(points),
                vector3_arg(rotation),
                vector3_arg(translation),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ),
        )?,
        "opencv.invalid_argument",
    )
}

/// Builds arguments in the public operation order.
fn arguments(
    points: String,
    rotation: String,
    translation: String,
    intrinsics: String,
    distortion: String,
) -> [String; 5] {
    [points, rotation, translation, intrinsics, distortion]
}

/// Requires two copied float vectors to match within projection tolerance.
fn expect_near(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-9)
    {
        Ok(())
    } else {
        Err(format!(
            "expected projected points {expected:?}, received {actual:?}"
        ))
    }
}
