//! Bounded camera-pose estimation contract.

use super::{
    calib3d_projection_contract::project, camera_intrinsics_arg, camera_pose_arg, distortion_arg,
    expect_error_code, expect_float_record, float_arg, float_list_arg, vector3_arg, NativeHelper,
};

const OBJECT_POINTS: [f64; 24] = [
    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, -0.5, -0.5, 1.0, 0.5, -0.5,
    1.0, -0.5, 0.5, 1.0, 0.5, 0.5, 1.0,
];
const MINIMAL_OBJECT_POINTS: [f64; 12] = [
    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0,
];
const PLANAR_OBJECT_POINTS: [f64; 18] = [
    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0,
];
const SQUARE_OBJECT_POINTS: [f64; 12] = [
    -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0,
];
const ROTATION: [f64; 3] = [0.1, -0.2, 0.05];
const TRANSLATION: [f64; 3] = [0.25, -0.4, 8.0];
const INTRINSICS: [f64; 4] = [100.0, 100.0, 50.0, 50.0];
const ZERO_DISTORTION: [f64; 5] = [0.0; 5];

/// Runs round-trip, deterministic, malformed-input, and boundedness checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let image_points = project(
        helper,
        &OBJECT_POINTS,
        ROTATION,
        TRANSLATION,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let pose = solve(
        helper,
        &OBJECT_POINTS,
        &image_points,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let repeated = solve(
        helper,
        &OBJECT_POINTS,
        &image_points,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    if pose != repeated {
        return Err("camera pose estimation was nondeterministic".to_owned());
    }
    expect_near(&pose[..3], &ROTATION, 1.0e-5)?;
    expect_near(&pose[3..], &TRANSLATION, 1.0e-5)?;
    let robust_pose = solve_robust(
        helper,
        &OBJECT_POINTS,
        &image_points,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&robust_pose[..3], &ROTATION, 1.0e-5)?;
    expect_near(&robust_pose[3..], &TRANSLATION, 1.0e-5)?;
    let mut image_points_with_outlier = image_points.clone();
    image_points_with_outlier[0] += 100.0;
    image_points_with_outlier[1] -= 100.0;
    let robust_outlier_pose = solve_robust(
        helper,
        &OBJECT_POINTS,
        &image_points_with_outlier,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let robust_outlier_repeated = solve_robust(
        helper,
        &OBJECT_POINTS,
        &image_points_with_outlier,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&robust_outlier_pose[..3], &ROTATION, 1.0e-4)?;
    expect_near(&robust_outlier_pose[3..], &TRANSLATION, 1.0e-4)?;
    expect_near(&robust_outlier_repeated, &robust_outlier_pose, 1.0e-12)?;
    let minimal_images = project(
        helper,
        &MINIMAL_OBJECT_POINTS,
        ROTATION,
        TRANSLATION,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let minimal_pose = solve_minimal(
        helper,
        &MINIMAL_OBJECT_POINTS,
        &minimal_images,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let minimal_repeated = solve_minimal(
        helper,
        &MINIMAL_OBJECT_POINTS,
        &minimal_images,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&minimal_pose[..3], &ROTATION, 1.0e-4)?;
    expect_near(&minimal_pose[3..], &TRANSLATION, 1.0e-4)?;
    expect_near(&minimal_repeated, &minimal_pose, 1.0e-12)?;
    let planar_images = project(
        helper,
        &PLANAR_OBJECT_POINTS,
        ROTATION,
        TRANSLATION,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let planar_pose = solve_planar(
        helper,
        &PLANAR_OBJECT_POINTS,
        &planar_images,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let planar_repeated = solve_planar(
        helper,
        &PLANAR_OBJECT_POINTS,
        &planar_images,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&planar_pose[..3], &ROTATION, 1.0e-4)?;
    expect_near(&planar_pose[3..], &TRANSLATION, 1.0e-4)?;
    expect_near(&planar_repeated, &planar_pose, 1.0e-12)?;
    let square_images = project(
        helper,
        &SQUARE_OBJECT_POINTS,
        ROTATION,
        TRANSLATION,
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    let square_pose =
        solve_square_marker(helper, &square_images, 2.0, INTRINSICS, ZERO_DISTORTION)?;
    let square_repeated =
        solve_square_marker(helper, &square_images, 2.0, INTRINSICS, ZERO_DISTORTION)?;
    expect_near(&square_pose[..3], &ROTATION, 1.0e-4)?;
    expect_near(&square_pose[3..], &TRANSLATION, 1.0e-4)?;
    expect_near(&square_repeated, &square_pose, 1.0e-12)?;
    let reprojected = project(
        helper,
        &OBJECT_POINTS,
        [pose[0], pose[1], pose[2]],
        [pose[3], pose[4], pose[5]],
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    expect_near(&reprojected, &image_points, 1.0e-6)?;

    let initial_pose = [0.15, -0.1, 0.1, 0.5, -0.2, 7.5];
    for operation in [
        "opencv.calib3d.refine_pose",
        "opencv.calib3d.refine_pose_vvs",
    ] {
        let refined = refine(
            helper,
            operation,
            &OBJECT_POINTS,
            &image_points,
            initial_pose,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        let refined_repeated = refine(
            helper,
            operation,
            &OBJECT_POINTS,
            &image_points,
            initial_pose,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        if refined != refined_repeated {
            return Err(format!("{operation} was nondeterministic"));
        }
        expect_near(&refined[..3], &ROTATION, 1.0e-5)?;
        expect_near(&refined[3..], &TRANSLATION, 1.0e-5)?;
    }

    let identity = [0.0; 6];
    expect_near(
        &compose(helper, identity, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0])?,
        &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        1.0e-12,
    )?;
    let first = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0];
    let second = [0.0, 0.0, 0.2, 0.0, 2.0, 0.0];
    let composed = compose(helper, first, second)?;
    let composed_repeated = compose(helper, first, second)?;
    if composed != composed_repeated {
        return Err("camera pose composition was nondeterministic".to_owned());
    }
    expect_near(&composed[..3], &[0.0, 0.0, 0.3], 1.0e-12)?;
    expect_near(
        &composed[3..],
        &[0.2_f64.cos(), 2.0 + 0.2_f64.sin(), 0.0],
        1.0e-12,
    )?;
    let reversed = compose(helper, second, first)?;
    expect_near(&reversed[..3], &[0.0, 0.0, 0.3], 1.0e-12)?;
    expect_near(
        &reversed[3..],
        &[1.0 - 2.0 * 0.1_f64.sin(), 2.0 * 0.1_f64.cos(), 0.0],
        1.0e-12,
    )?;
    if composed == reversed {
        return Err("camera pose composition ignored application order".to_owned());
    }

    for (objects, images) in [
        (vec![], vec![]),
        (OBJECT_POINTS[..15].to_vec(), image_points[..10].to_vec()),
        (OBJECT_POINTS.to_vec(), image_points[..14].to_vec()),
        (OBJECT_POINTS[..23].to_vec(), image_points.to_vec()),
        (OBJECT_POINTS.to_vec(), image_points[..15].to_vec()),
    ] {
        expect_invalid(helper, &objects, &images, INTRINSICS, ZERO_DISTORTION)?;
    }
    expect_invalid(
        helper,
        &vec![0.0; 4097 * 3],
        &vec![0.0; 4097 * 2],
        INTRINSICS,
        ZERO_DISTORTION,
    )?;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut objects = OBJECT_POINTS;
        objects[0] = invalid;
        expect_invalid(helper, &objects, &image_points, INTRINSICS, ZERO_DISTORTION)?;
        let mut images = image_points.clone();
        images[0] = invalid;
        expect_invalid(helper, &OBJECT_POINTS, &images, INTRINSICS, ZERO_DISTORTION)?;
    }
    expect_invalid(
        helper,
        &OBJECT_POINTS,
        &image_points,
        [0.0, 100.0, 50.0, 50.0],
        ZERO_DISTORTION,
    )?;
    expect_invalid(
        helper,
        &OBJECT_POINTS,
        &image_points,
        INTRINSICS,
        [11.0, 0.0, 0.0, 0.0, 0.0],
    )?;
    for (objects, images) in [
        (
            MINIMAL_OBJECT_POINTS[..9].to_vec(),
            minimal_images[..6].to_vec(),
        ),
        (
            [MINIMAL_OBJECT_POINTS.to_vec(), vec![0.5, 0.5, 0.5]].concat(),
            [minimal_images.clone(), vec![50.0, 50.0]].concat(),
        ),
        (MINIMAL_OBJECT_POINTS.to_vec(), minimal_images[..6].to_vec()),
        (vec![0.0; 12], minimal_images.clone()),
        (
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            minimal_images.clone(),
        ),
        (
            MINIMAL_OBJECT_POINTS.to_vec(),
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
        ),
    ] {
        expect_minimal_invalid(helper, &objects, &images, INTRINSICS, ZERO_DISTORTION)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut objects = MINIMAL_OBJECT_POINTS;
        objects[0] = invalid;
        expect_minimal_invalid(
            helper,
            &objects,
            &minimal_images,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        let mut images = minimal_images.clone();
        images[0] = invalid;
        expect_minimal_invalid(
            helper,
            &MINIMAL_OBJECT_POINTS,
            &images,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut objects = PLANAR_OBJECT_POINTS;
        objects[0] = invalid;
        expect_planar_invalid(
            helper,
            &objects,
            &planar_images,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        let mut images = planar_images.clone();
        images[0] = invalid;
        expect_planar_invalid(
            helper,
            &PLANAR_OBJECT_POINTS,
            &images,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
    }
    for (objects, images) in [
        (
            PLANAR_OBJECT_POINTS[..9].to_vec(),
            planar_images[..6].to_vec(),
        ),
        (MINIMAL_OBJECT_POINTS.to_vec(), minimal_images.clone()),
        (vec![0.0; 18], planar_images.clone()),
        (
            PLANAR_OBJECT_POINTS.to_vec(),
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0],
        ),
        (PLANAR_OBJECT_POINTS.to_vec(), planar_images[..10].to_vec()),
    ] {
        expect_planar_invalid(helper, &objects, &images, INTRINSICS, ZERO_DISTORTION)?;
    }
    for images in [
        square_images[..6].to_vec(),
        [square_images.clone(), vec![50.0, 50.0]].concat(),
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
    ] {
        expect_square_invalid(helper, &images, 2.0, INTRINSICS, ZERO_DISTORTION)?;
    }
    for marker_size in [
        0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1_000_001.0,
    ] {
        expect_square_invalid(
            helper,
            &square_images,
            marker_size,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut images = square_images.clone();
        images[0] = invalid;
        expect_square_invalid(helper, &images, 2.0, INTRINSICS, ZERO_DISTORTION)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut pose = initial_pose;
        pose[0] = invalid;
        for operation in [
            "opencv.calib3d.refine_pose",
            "opencv.calib3d.refine_pose_vvs",
        ] {
            expect_refine_invalid(
                helper,
                operation,
                &OBJECT_POINTS,
                &image_points,
                pose,
                INTRINSICS,
                ZERO_DISTORTION,
            )?;
        }
        expect_error_code(
            helper.call(
                "opencv.calib3d.compose_pose",
                &[camera_pose_arg(pose), camera_pose_arg(identity)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.calib3d.compose_pose",
                &[camera_pose_arg(identity), camera_pose_arg(pose)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for operation in [
        "opencv.calib3d.refine_pose",
        "opencv.calib3d.refine_pose_vvs",
    ] {
        expect_refine_invalid(
            helper,
            operation,
            &OBJECT_POINTS[..15],
            &image_points[..10],
            initial_pose,
            INTRINSICS,
            ZERO_DISTORTION,
        )?;
        expect_error_code(
            helper.call(
                operation,
                &[
                    float_list_arg(&OBJECT_POINTS),
                    float_list_arg(&image_points),
                    vector3_arg([0.0; 3]),
                    camera_intrinsics_arg(INTRINSICS),
                    distortion_arg(ZERO_DISTORTION),
                ],
            )?,
            "record_type_mismatch",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.calib3d.solve_pose_minimal",
            &[
                "li:0,0,0,1,0,0,0,1,0,0,0,1".to_owned(),
                float_list_arg(&minimal_images),
                camera_intrinsics_arg(INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.solve_pose_planar",
            &[
                "li:0,0,0,1,0,0,0,1,0,1,1,0".to_owned(),
                float_list_arg(&planar_images[..8]),
                camera_intrinsics_arg(INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.calib3d.solve_square_marker_pose",
            &[
                "li:0,0,1,0,1,1,0,1".to_owned(),
                float_arg(2.0),
                camera_intrinsics_arg(INTRINSICS),
                distortion_arg(ZERO_DISTORTION),
            ],
        )?,
        "invalid_arguments",
    )?;
    for operation in [
        "opencv.calib3d.solve_pose",
        "opencv.calib3d.solve_pose_robust",
    ] {
        expect_error_code(
            helper.call(
                operation,
                &[
                    "li:0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1".to_owned(),
                    float_list_arg(&image_points[..12]),
                    camera_intrinsics_arg(INTRINSICS),
                    distortion_arg(ZERO_DISTORTION),
                ],
            )?,
            "invalid_arguments",
        )?;
        expect_error_code(
            helper.call(
                operation,
                &[
                    "lf:1,,2".to_owned(),
                    float_list_arg(&image_points),
                    camera_intrinsics_arg(INTRINSICS),
                    distortion_arg(ZERO_DISTORTION),
                ],
            )?,
            "invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.calib3d.compose_pose",
            &[vector3_arg([0.0; 3]), camera_pose_arg(identity)],
        )?,
        "record_type_mismatch",
    )
}

/// Composes and decodes two copied camera poses.
fn compose(
    helper: &mut NativeHelper,
    first: [f64; 6],
    second: [f64; 6],
) -> Result<Vec<f64>, String> {
    expect_float_record(
        helper.call(
            "opencv.calib3d.compose_pose",
            &[camera_pose_arg(first), camera_pose_arg(second)],
        )?,
        "CameraPose",
        &["rx", "ry", "rz", "tx", "ty", "tz"],
    )
}

/// Refines and decodes one copied camera pose.
fn refine(
    helper: &mut NativeHelper,
    operation: &str,
    objects: &[f64],
    images: &[f64],
    initial_pose: [f64; 6],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    expect_float_record(
        helper.call(
            operation,
            &[
                float_list_arg(objects),
                float_list_arg(images),
                camera_pose_arg(initial_pose),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "CameraPose",
        &["rx", "ry", "rz", "tx", "ty", "tz"],
    )
}

/// Estimates and decodes one copied camera pose.
fn solve(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    solve_with_operation(
        helper,
        "opencv.calib3d.solve_pose",
        objects,
        images,
        intrinsics,
        distortion,
    )
}

/// Estimates and decodes one copied pose with fixed robust outlier rejection.
fn solve_robust(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    solve_with_operation(
        helper,
        "opencv.calib3d.solve_pose_robust",
        objects,
        images,
        intrinsics,
        distortion,
    )
}

/// Estimates and decodes the best copied four-correspondence pose.
fn solve_minimal(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    solve_with_operation(
        helper,
        "opencv.calib3d.solve_pose_minimal",
        objects,
        images,
        intrinsics,
        distortion,
    )
}

/// Estimates and decodes the lowest-error copied planar pose.
fn solve_planar(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    solve_with_operation(
        helper,
        "opencv.calib3d.solve_pose_planar",
        objects,
        images,
        intrinsics,
        distortion,
    )
}

/// Estimates and decodes one fixed-order square-marker pose.
fn solve_square_marker(
    helper: &mut NativeHelper,
    images: &[f64],
    marker_size: f64,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    expect_float_record(
        helper.call(
            "opencv.calib3d.solve_square_marker_pose",
            &[
                float_list_arg(images),
                float_arg(marker_size),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "CameraPose",
        &["rx", "ry", "rz", "tx", "ty", "tz"],
    )
}

/// Estimates and decodes one copied pose through the selected operation.
fn solve_with_operation(
    helper: &mut NativeHelper,
    operation: &str,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<Vec<f64>, String> {
    expect_float_record(
        helper.call(
            operation,
            &[
                float_list_arg(objects),
                float_list_arg(images),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "CameraPose",
        &["rx", "ry", "rz", "tx", "ty", "tz"],
    )
}

/// Requires malformed pose correspondences to fail before returning a value.
fn expect_invalid(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    for operation in [
        "opencv.calib3d.solve_pose",
        "opencv.calib3d.solve_pose_robust",
    ] {
        expect_error_code(
            helper.call(
                operation,
                &[
                    float_list_arg(objects),
                    float_list_arg(images),
                    camera_intrinsics_arg(intrinsics),
                    distortion_arg(distortion),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    Ok(())
}

/// Requires malformed minimal-pose input to fail before returning a value.
fn expect_minimal_invalid(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.solve_pose_minimal",
            &[
                float_list_arg(objects),
                float_list_arg(images),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Requires malformed planar-pose input to fail before returning a value.
fn expect_planar_invalid(
    helper: &mut NativeHelper,
    objects: &[f64],
    images: &[f64],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.solve_pose_planar",
            &[
                float_list_arg(objects),
                float_list_arg(images),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Requires malformed square-marker input to fail before returning a value.
fn expect_square_invalid(
    helper: &mut NativeHelper,
    images: &[f64],
    marker_size: f64,
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.calib3d.solve_square_marker_pose",
            &[
                float_list_arg(images),
                float_arg(marker_size),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Requires malformed refinement input to fail before returning a value.
fn expect_refine_invalid(
    helper: &mut NativeHelper,
    operation: &str,
    objects: &[f64],
    images: &[f64],
    initial_pose: [f64; 6],
    intrinsics: [f64; 4],
    distortion: [f64; 5],
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            operation,
            &[
                float_list_arg(objects),
                float_list_arg(images),
                camera_pose_arg(initial_pose),
                camera_intrinsics_arg(intrinsics),
                distortion_arg(distortion),
            ],
        )?,
        "opencv.invalid_argument",
    )
}

/// Requires two float slices to agree within one explicit tolerance.
fn expect_near(actual: &[f64], expected: &[f64], tolerance: f64) -> Result<(), String> {
    if actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
    {
        Ok(())
    } else {
        Err(format!("expected {expected:?}, received {actual:?}"))
    }
}
