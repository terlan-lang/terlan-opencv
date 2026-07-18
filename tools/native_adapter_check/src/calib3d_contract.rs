//! Bounded camera-undistortion native-helper contract.

use super::{
    affine_arg, camera_intrinsics_arg, distortion_arg, expect_bytes, expect_error_code,
    expect_float_record, expect_floats, expect_handle, expect_int, expect_unit, float_list_arg,
    matrix, perspective_arg, NativeHandle, NativeHelper,
};

const IDENTITY_INTRINSICS: [f64; 4] = [8.0, 8.0, 4.0, 4.0];
const ZERO_DISTORTION: [f64; 5] = [0.0; 5];

/// Runs successful correction, ownership, limit, and forged-input checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let fixture = grayscale_fixture();
    let source = matrix(helper, &fixture, 9, 9, "gray")?;
    let identity = undistort(helper, &source, IDENTITY_INTRINSICS, ZERO_DISTORTION)?;
    assert_shape(helper, &identity, 9, 9, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[identity.arg()])?,
        &fixture,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[source.arg()])?, &fixture)?;

    let distorted = undistort(
        helper,
        &source,
        IDENTITY_INTRINSICS,
        [-0.2, 0.03, 0.01, -0.01, 0.0],
    )?;
    let repeated = undistort(
        helper,
        &source,
        IDENTITY_INTRINSICS,
        [-0.2, 0.03, 0.01, -0.01, 0.0],
    )?;
    assert_shape(helper, &distorted, 9, 9, 1)?;
    let distorted_bytes = copied_bytes(helper, &distorted)?;
    if distorted_bytes != copied_bytes(helper, &repeated)? || distorted_bytes.len() != fixture.len()
    {
        return Err("undistortion output was nondeterministic or malformed".to_owned());
    }
    let optimal = undistort_optimal(
        helper,
        &source,
        IDENTITY_INTRINSICS,
        [-0.2, 0.03, 0.01, -0.01, 0.0],
        0.0,
        false,
    )?;
    let optimal_repeated = undistort_optimal(
        helper,
        &source,
        IDENTITY_INTRINSICS,
        [-0.2, 0.03, 0.01, -0.01, 0.0],
        0.0,
        false,
    )?;
    let optimal_centered = undistort_optimal(
        helper,
        &source,
        IDENTITY_INTRINSICS,
        [-0.2, 0.03, 0.01, -0.01, 0.0],
        1.0,
        true,
    )?;
    assert_shape(helper, &optimal, 9, 9, 1)?;
    assert_shape(helper, &optimal_centered, 9, 9, 1)?;
    if copied_bytes(helper, &optimal)? != copied_bytes(helper, &optimal_repeated)? {
        return Err("optimal undistortion output was nondeterministic".to_owned());
    }

    let color = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    let color_result = undistort(helper, &color, [2.0, 2.0, 0.5, 0.5], ZERO_DISTORTION)?;
    let color_optimal = undistort_optimal(
        helper,
        &color,
        [2.0, 2.0, 0.5, 0.5],
        ZERO_DISTORTION,
        0.5,
        true,
    )?;
    assert_shape(helper, &color_result, 2, 2, 3)?;
    assert_shape(helper, &color_optimal, 2, 2, 3)?;

    validate_point_undistortion(helper)?;
    validate_perspective_point_transform(helper)?;
    validate_affine_point_transform(helper)?;
    validate_perspective_transform_estimation(helper)?;
    validate_affine_transform_estimation(helper)?;

    reject_camera_models(helper, &source)?;
    reject_distortion_models(helper, &source)?;
    for alpha in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        expect_optimal_invalid(helper, &source, IDENTITY_INTRINSICS, ZERO_DISTORTION, alpha)?;
    }
    for (argument, code) in [
        (
            source.arg_with_owner("foreign-worker"),
            "cross_owner_handle",
        ),
        (
            source.arg_with_type("other.Package.Resource"),
            "handle_type_mismatch",
        ),
    ] {
        expect_error_code(
            helper.call(
                "opencv.calib3d.undistort",
                &[
                    argument.clone(),
                    camera_intrinsics_arg(IDENTITY_INTRINSICS),
                    distortion_arg(ZERO_DISTORTION),
                ],
            )?,
            code,
        )?;
        expect_error_code(
            helper.call(
                "opencv.calib3d.undistort_optimal",
                &[
                    argument,
                    camera_intrinsics_arg(IDENTITY_INTRINSICS),
                    distortion_arg(ZERO_DISTORTION),
                    super::float_arg(0.5),
                    super::bool_arg(false),
                ],
            )?,
            code,
        )?;
    }
    let oversized = matrix(helper, &[0; 1025 * 1024], 1025, 1024, "gray")?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort",
            &[
                oversized.arg(),
                camera_intrinsics_arg([1000.0, 1000.0, 512.0, 512.0]),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_optimal_invalid(
        helper,
        &oversized,
        [1000.0, 1000.0, 512.0, 512.0],
        ZERO_DISTORTION,
        0.5,
    )?;

    expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[identity.arg()])?,
        &fixture,
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort",
            &[
                source.arg(),
                camera_intrinsics_arg(IDENTITY_INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort_optimal",
            &[
                source.arg(),
                camera_intrinsics_arg(IDENTITY_INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
                super::float_arg(0.5),
                super::bool_arg(false),
            ],
        )?,
        "stale_handle",
    )?;
    for handle in [
        identity,
        distorted,
        repeated,
        optimal,
        optimal_repeated,
        optimal_centered,
        color_optimal,
        color_result,
        color,
        oversized,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Validates copied point correction, determinism, empty input, and rejection policy.
fn validate_point_undistortion(helper: &mut NativeHelper) -> Result<(), String> {
    let points = [0.0, 0.0, 4.0, 4.0, 8.0, 8.0];
    let identity = undistort_points(helper, &points, IDENTITY_INTRINSICS, ZERO_DISTORTION)?;
    expect_near(&identity, &points)?;
    let distortion = [-0.2, 0.03, 0.01, -0.01, 0.0];
    let corrected = undistort_points(helper, &points, IDENTITY_INTRINSICS, distortion)?;
    let repeated = undistort_points(helper, &points, IDENTITY_INTRINSICS, distortion)?;
    if corrected != repeated || corrected.len() != points.len() {
        return Err("point undistortion was nondeterministic or malformed".to_owned());
    }
    if !undistort_points(helper, &[], IDENTITY_INTRINSICS, ZERO_DISTORTION)?.is_empty() {
        return Err("empty point undistortion produced coordinates".to_owned());
    }

    for invalid in [
        vec![1.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
    ] {
        expect_points_invalid(helper, &invalid, IDENTITY_INTRINSICS, ZERO_DISTORTION)?;
    }
    expect_points_invalid(
        helper,
        &vec![0.0; 4097 * 2],
        IDENTITY_INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_points_invalid(helper, &points, [0.0, 8.0, 4.0, 4.0], ZERO_DISTORTION)?;
    expect_points_invalid(
        helper,
        &points,
        IDENTITY_INTRINSICS,
        [10.1, 0.0, 0.0, 0.0, 0.0],
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort_points",
            &[
                "li:1,2".to_owned(),
                camera_intrinsics_arg(IDENTITY_INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort_points",
            &[
                "lf:1,,2".to_owned(),
                camera_intrinsics_arg(IDENTITY_INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "invalid_argument",
    )
}

/// Validates copied perspective point transforms and bounded rejection policy.
fn validate_perspective_point_transform(helper: &mut NativeHelper) -> Result<(), String> {
    const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let points = [0.0, 0.0, 4.0, 4.0];
    expect_near(&perspective_points(helper, &points, IDENTITY)?, &points)?;
    let translated = [1.0, 0.0, 2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
    let first = perspective_points(helper, &points, translated)?;
    let repeated = perspective_points(helper, &points, translated)?;
    expect_near(&first, &[2.0, -1.0, 6.0, 3.0])?;
    if first != repeated {
        return Err("perspective point transform was nondeterministic".to_owned());
    }
    if !perspective_points(helper, &[], IDENTITY)?.is_empty() {
        return Err("empty perspective point transform produced coordinates".to_owned());
    }
    for invalid in [
        vec![1.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
        vec![0.0; 4097 * 2],
    ] {
        expect_perspective_points_invalid(helper, &invalid, IDENTITY)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut transform = IDENTITY;
        transform[0] = invalid;
        expect_perspective_points_invalid(helper, &points, transform)?;
    }
    expect_perspective_points_invalid(
        helper,
        &[1.0, 0.0],
        [1.0, 0.0, 1_000_000.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.perspective_transform_points",
            &["li:1,2".to_owned(), perspective_arg(IDENTITY)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.perspective_transform_points",
            &["lf:1,,2".to_owned(), perspective_arg(IDENTITY)],
        )?,
        "invalid_argument",
    )
}

/// Requires one perspective point transform to fail at the native boundary.
fn expect_perspective_points_invalid(
    helper: &mut NativeHelper,
    points: &[f64],
    transform: [f64; 9],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.perspective_transform_points",
            &[float_list_arg(points), perspective_arg(transform)],
        )?,
        "opencv.invalid_argument",
    )
}

/// Invokes the generated copied perspective point transform.
fn perspective_points(
    helper: &mut NativeHelper,
    points: &[f64],
    transform: [f64; 9],
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.calib3d.perspective_transform_points",
        &[float_list_arg(points), perspective_arg(transform)],
    )?)
}

/// Validates copied affine point transforms and bounded rejection policy.
fn validate_affine_point_transform(helper: &mut NativeHelper) -> Result<(), String> {
    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let points = [0.0, 0.0, 4.0, 4.0];
    expect_near(&affine_points(helper, &points, IDENTITY)?, &points)?;
    let translated = [1.0, 0.0, 2.0, 0.0, 1.0, -1.0];
    let first = affine_points(helper, &points, translated)?;
    let repeated = affine_points(helper, &points, translated)?;
    expect_near(&first, &[2.0, -1.0, 6.0, 3.0])?;
    if first != repeated {
        return Err("affine point transform was nondeterministic".to_owned());
    }
    if !affine_points(helper, &[], IDENTITY)?.is_empty() {
        return Err("empty affine point transform produced coordinates".to_owned());
    }
    for invalid in [
        vec![1.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
        vec![0.0; 4097 * 2],
    ] {
        expect_affine_points_invalid(helper, &invalid, IDENTITY)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut transform = IDENTITY;
        transform[0] = invalid;
        expect_affine_points_invalid(helper, &points, transform)?;
    }
    expect_affine_points_invalid(helper, &[1.0, 0.0], [1.0, 0.0, 1_000_000.0, 0.0, 1.0, 0.0])?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.affine_transform_points",
            &["li:1,2".to_owned(), affine_arg(IDENTITY)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.affine_transform_points",
            &["lf:1,,2".to_owned(), affine_arg(IDENTITY)],
        )?,
        "invalid_argument",
    )
}

/// Requires one affine point transform to fail at the native boundary.
fn expect_affine_points_invalid(
    helper: &mut NativeHelper,
    points: &[f64],
    transform: [f64; 6],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.affine_transform_points",
            &[float_list_arg(points), affine_arg(transform)],
        )?,
        "opencv.invalid_argument",
    )
}

/// Invokes the generated copied affine point transform.
fn affine_points(
    helper: &mut NativeHelper,
    points: &[f64],
    transform: [f64; 6],
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.calib3d.affine_transform_points",
        &[float_list_arg(points), affine_arg(transform)],
    )?)
}

/// Validates perspective estimation, copied-record projection, and rejection policy.
fn validate_perspective_transform_estimation(helper: &mut NativeHelper) -> Result<(), String> {
    const SOURCE: [f64; 8] = [0.0, 0.0, 4.0, 0.0, 4.0, 4.0, 0.0, 4.0];
    const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let identity = estimate_perspective_transform(helper, &SOURCE, &SOURCE)?;
    expect_near(&identity, &IDENTITY)?;
    let destination = [2.0, -1.0, 6.0, -1.0, 6.0, 3.0, 2.0, 3.0];
    let translated = estimate_perspective_transform(helper, &SOURCE, &destination)?;
    let repeated = estimate_perspective_transform(helper, &SOURCE, &destination)?;
    expect_near(&translated, &[1.0, 0.0, 2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0])?;
    if translated != repeated {
        return Err("perspective transform estimation was nondeterministic".to_owned());
    }
    expect_near(
        &perspective_points(
            helper,
            &SOURCE,
            translated
                .clone()
                .try_into()
                .map_err(|_| "estimated transform did not have nine fields")?,
        )?,
        &destination,
    )?;

    for (source, destination) in [
        (vec![], SOURCE.to_vec()),
        (vec![0.0, 0.0], SOURCE.to_vec()),
        (SOURCE[..6].to_vec(), SOURCE.to_vec()),
        ([SOURCE.as_slice(), &[5.0, 5.0]].concat(), SOURCE.to_vec()),
        (SOURCE.to_vec(), vec![]),
        (SOURCE.to_vec(), SOURCE[..6].to_vec()),
        (
            vec![0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 4.0],
            SOURCE.to_vec(),
        ),
        (
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 4.0],
            SOURCE.to_vec(),
        ),
    ] {
        expect_estimation_invalid(helper, &source, &destination)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut source = SOURCE;
        source[0] = invalid;
        expect_estimation_invalid(helper, &source, &SOURCE)?;
        let mut destination = SOURCE;
        destination[7] = invalid;
        expect_estimation_invalid(helper, &SOURCE, &destination)?;
    }
    expect_error_code(
        helper.call(
            "opencv.calib3d.estimate_perspective_transform",
            &["li:1,2,3,4,5,6,7,8".to_owned(), float_list_arg(&SOURCE)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.estimate_perspective_transform",
            &["lf:1,,2".to_owned(), float_list_arg(&SOURCE)],
        )?,
        "invalid_argument",
    )
}

/// Requires one invalid perspective estimation call to fail finitely.
fn expect_estimation_invalid(
    helper: &mut NativeHelper,
    source: &[f64],
    destination: &[f64],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.estimate_perspective_transform",
            &[float_list_arg(source), float_list_arg(destination)],
        )?,
        "opencv.invalid_argument",
    )
}

/// Estimates and decodes one copied perspective transform record.
fn estimate_perspective_transform(
    helper: &mut NativeHelper,
    source: &[f64],
    destination: &[f64],
) -> Result<Vec<f64>, String> {
    expect_float_record(
        helper.call(
            "opencv.calib3d.estimate_perspective_transform",
            &[float_list_arg(source), float_list_arg(destination)],
        )?,
        "PerspectiveTransform",
        &[
            "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22",
        ],
    )
}

/// Validates copied affine estimation and rejects unusable correspondences.
fn validate_affine_transform_estimation(helper: &mut NativeHelper) -> Result<(), String> {
    const SOURCE: [f64; 6] = [0.0, 0.0, 4.0, 0.0, 0.0, 4.0];
    const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let identity = estimate_affine_transform(helper, &SOURCE, &SOURCE)?;
    expect_near(&identity, &IDENTITY)?;
    let destination = [2.0, -1.0, 6.0, -1.0, 2.0, 3.0];
    let translated = estimate_affine_transform(helper, &SOURCE, &destination)?;
    let repeated = estimate_affine_transform(helper, &SOURCE, &destination)?;
    expect_near(&translated, &[1.0, 0.0, 2.0, 0.0, 1.0, -1.0])?;
    if translated != repeated {
        return Err("affine transform estimation was nondeterministic".to_owned());
    }
    expect_near(
        &affine_points(
            helper,
            &SOURCE,
            translated
                .clone()
                .try_into()
                .map_err(|_| "estimated affine transform did not have six fields")?,
        )?,
        &destination,
    )?;

    for (source, destination) in [
        (vec![], SOURCE.to_vec()),
        (vec![0.0, 0.0], SOURCE.to_vec()),
        (SOURCE[..4].to_vec(), SOURCE.to_vec()),
        ([SOURCE.as_slice(), &[5.0, 5.0]].concat(), SOURCE.to_vec()),
        (SOURCE.to_vec(), vec![]),
        (SOURCE.to_vec(), SOURCE[..4].to_vec()),
        (vec![0.0, 0.0, 0.0, 0.0, 0.0, 4.0], SOURCE.to_vec()),
        (vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0], SOURCE.to_vec()),
        (SOURCE.to_vec(), vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0]),
    ] {
        expect_affine_estimation_invalid(helper, &source, &destination)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut source = SOURCE;
        source[0] = invalid;
        expect_affine_estimation_invalid(helper, &source, &SOURCE)?;
        let mut destination = SOURCE;
        destination[5] = invalid;
        expect_affine_estimation_invalid(helper, &SOURCE, &destination)?;
    }
    expect_error_code(
        helper.call(
            "opencv.calib3d.estimate_affine_transform",
            &["li:1,2,3,4,5,6".to_owned(), float_list_arg(&SOURCE)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.estimate_affine_transform",
            &["lf:1,,2".to_owned(), float_list_arg(&SOURCE)],
        )?,
        "invalid_argument",
    )
}

/// Requires one invalid affine estimation call to fail finitely.
fn expect_affine_estimation_invalid(
    helper: &mut NativeHelper,
    source: &[f64],
    destination: &[f64],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.estimate_affine_transform",
            &[float_list_arg(source), float_list_arg(destination)],
        )?,
        "opencv.invalid_argument",
    )
}

/// Estimates and decodes one copied affine transform record.
fn estimate_affine_transform(
    helper: &mut NativeHelper,
    source: &[f64],
    destination: &[f64],
) -> Result<Vec<f64>, String> {
    expect_float_record(
        helper.call(
            "opencv.calib3d.estimate_affine_transform",
            &[float_list_arg(source), float_list_arg(destination)],
        )?,
        "AffineTransform",
        &["m00", "m01", "m02", "m10", "m11", "m12"],
    )
}

/// Requires one point-correction call to fail at the native boundary.
fn expect_points_invalid(
    helper: &mut NativeHelper,
    points: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort_points",
            &[
                float_list_arg(points),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Invokes generated copied point correction.
fn undistort_points(
    helper: &mut NativeHelper,
    points: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.calib3d.undistort_points",
        &[
            float_list_arg(points),
            camera_intrinsics_arg(intrinsics),
            distortion_arg(distortion),
        ],
    )?)
}

/// Compares copied coordinates with native floating-point tolerance.
fn expect_near(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-9)
    {
        return Ok(());
    }
    Err(format!(
        "unexpected undistorted points: actual={actual:?}, expected={expected:?}"
    ))
}

/// Rejects non-finite, non-positive, and out-of-image camera parameters.
fn reject_camera_models(helper: &mut NativeHelper, source: &NativeHandle) -> Result<(), String> {
    for intrinsics in [
        [0.0, 8.0, 4.0, 4.0],
        [-1.0, 8.0, 4.0, 4.0],
        [8.0, 0.0, 4.0, 4.0],
        [8.0, -1.0, 4.0, 4.0],
        [f64::NAN, 8.0, 4.0, 4.0],
        [8.0, f64::INFINITY, 4.0, 4.0],
        [1_000_001.0, 8.0, 4.0, 4.0],
        [8.0, 1_000_001.0, 4.0, 4.0],
        [8.0, 8.0, -1.0, 4.0],
        [8.0, 8.0, 9.0, 4.0],
        [8.0, 8.0, 4.0, -1.0],
        [8.0, 8.0, 4.0, 9.0],
    ] {
        expect_invalid(helper, source, intrinsics, ZERO_DISTORTION)?;
    }
    Ok(())
}

/// Rejects every non-finite or excessive lens-distortion coefficient.
fn reject_distortion_models(
    helper: &mut NativeHelper,
    source: &NativeHandle,
) -> Result<(), String> {
    for index in 0..5 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 10.1, -10.1] {
            let mut distortion = ZERO_DISTORTION;
            distortion[index] = invalid;
            expect_invalid(helper, source, IDENTITY_INTRINSICS, distortion)?;
        }
    }
    Ok(())
}

/// Requires one invalid camera invocation to fail at the native boundary.
fn expect_invalid(
    helper: &mut NativeHelper,
    source: &NativeHandle,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort",
            &[
                source.arg(),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Requires one optimal-camera invocation to fail at the native boundary.
fn expect_optimal_invalid(
    helper: &mut NativeHelper,
    source: &NativeHandle,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
    alpha: f64,
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.undistort_optimal",
            &[
                source.arg(),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
                super::float_arg(alpha),
                super::bool_arg(false),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Invokes the generated undistortion operation.
fn undistort(
    helper: &mut NativeHelper,
    source: &NativeHandle,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<NativeHandle, String> {
    expect_handle(helper.call(
        "opencv.calib3d.undistort",
        &[
            source.arg(),
            camera_intrinsics_arg(intrinsics),
            distortion_arg(distortion),
        ],
    )?)
}

/// Invokes the generated optimal-camera undistortion operation.
fn undistort_optimal(
    helper: &mut NativeHelper,
    source: &NativeHandle,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
    alpha: f64,
    center_principal_point: bool,
) -> Result<NativeHandle, String> {
    expect_handle(helper.call(
        "opencv.calib3d.undistort_optimal",
        &[
            source.arg(),
            camera_intrinsics_arg(intrinsics),
            distortion_arg(distortion),
            super::float_arg(alpha),
            super::bool_arg(center_principal_point),
        ],
    )?)
}

/// Requires one matrix to preserve the expected image geometry.
fn assert_shape(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    rows: i64,
    cols: i64,
    channels: i64,
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[image.arg()])?, rows)?;
    expect_int(helper.call("opencv.mat.cols", &[image.arg()])?, cols)?;
    expect_int(
        helper.call("opencv.mat.channels", &[image.arg()])?,
        channels,
    )
}

/// Copies one matrix through the generated helper protocol.
fn copied_bytes(helper: &mut NativeHelper, image: &NativeHandle) -> Result<Vec<u8>, String> {
    super::expect_bytes_value(helper.call("opencv.mat.to_u8", &[image.arg()])?)
}

/// Builds an asymmetric grayscale fixture that exposes remapping errors.
fn grayscale_fixture() -> Vec<u8> {
    (0..81)
        .map(|index| ((index * 37 + 11) % 256) as u8)
        .collect()
}
