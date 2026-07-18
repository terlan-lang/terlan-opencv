//! Direct camera-pose estimation tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

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

/// Proves projection-to-solver round trips and deterministic copied output.
#[test]
fn solves_projected_camera_pose() {
    let images = project(&OBJECT_POINTS, ROTATION, TRANSLATION);
    let first = solve(&OBJECT_POINTS, &images);
    let repeated = solve(&OBJECT_POINTS, &images);
    assert_pose(&first, &[0.1, -0.2, 0.05, 0.25, -0.4, 8.0]);
    assert_pose(&repeated, &pose_values(&first));
    assert_pose(&solve_robust(&OBJECT_POINTS, &images), &pose_values(&first));

    let mut images_with_outlier = images.clone();
    images_with_outlier[0] += 100.0;
    images_with_outlier[1] -= 100.0;
    let robust = solve_robust(&OBJECT_POINTS, &images_with_outlier);
    assert_pose(&robust, &[0.1, -0.2, 0.05, 0.25, -0.4, 8.0]);
    assert_pose(
        &solve_robust(&OBJECT_POINTS, &images_with_outlier),
        &pose_values(&robust),
    );

    for vvs in [false, true] {
        let refined = refine(
            &OBJECT_POINTS,
            &images,
            [0.15, -0.1, 0.1, 0.5, -0.2, 7.5],
            vvs,
        );
        assert_pose(&refined, &[0.1, -0.2, 0.05, 0.25, -0.4, 8.0]);
    }
}

/// Proves copied poses compose in documented application order.
#[test]
fn composes_camera_poses() {
    let first = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0];
    let second = [0.0, 0.0, 0.2, 0.0, 2.0, 0.0];
    let composed = compose(first, second);
    assert_pose(
        &composed,
        &[0.0, 0.0, 0.3, 0.2_f64.cos(), 2.0 + 0.2_f64.sin(), 0.0],
    );
    assert_pose(&compose(first, second), &pose_values(&composed));
    let reversed = compose(second, first);
    assert_pose(
        &reversed,
        &[
            0.0,
            0.0,
            0.3,
            1.0 - 2.0 * 0.1_f64.sin(),
            2.0 * 0.1_f64.cos(),
            0.0,
        ],
    );
    assert_ne!(pose_values(&composed), pose_values(&reversed));
}

/// Proves four non-coplanar observations select the lowest-error AP3P pose.
#[test]
fn solves_minimal_camera_pose() {
    let images = project(&MINIMAL_OBJECT_POINTS, ROTATION, TRANSLATION);
    let pose = solve_minimal(&MINIMAL_OBJECT_POINTS, &images);
    assert_pose(&pose, &[0.1, -0.2, 0.05, 0.25, -0.4, 8.0]);
    assert_pose(
        &solve_minimal(&MINIMAL_OBJECT_POINTS, &images),
        &pose_values(&pose),
    );
}

/// Proves planar observations select the lowest-error fixed-IPPE pose.
#[test]
fn solves_planar_camera_pose() {
    let images = project(&PLANAR_OBJECT_POINTS, ROTATION, TRANSLATION);
    let pose = solve_planar(&PLANAR_OBJECT_POINTS, &images);
    assert_pose(&pose, &[0.1, -0.2, 0.05, 0.25, -0.4, 8.0]);
    assert_pose(
        &solve_planar(&PLANAR_OBJECT_POINTS, &images),
        &pose_values(&pose),
    );
}

/// Proves one ordered square marker selects the lowest-error IPPE pose.
#[test]
fn solves_square_marker_camera_pose() {
    let images = project(&SQUARE_OBJECT_POINTS, ROTATION, TRANSLATION);
    let pose = solve_square_marker(&images, 2.0);
    assert_pose(&pose, &[0.1, -0.2, 0.05, 0.25, -0.4, 8.0]);
    assert_pose(&solve_square_marker(&images, 2.0), &pose_values(&pose));
}

/// Proves malformed, mismatched, excessive, and non-finite inputs are rejected.
#[test]
fn rejects_invalid_pose_correspondences() {
    let images = project(&OBJECT_POINTS, ROTATION, TRANSLATION);
    for (objects, images) in [
        (vec![], vec![]),
        (OBJECT_POINTS[..15].to_vec(), images[..10].to_vec()),
        (OBJECT_POINTS.to_vec(), images[..14].to_vec()),
        (OBJECT_POINTS[..23].to_vec(), images.clone()),
        (OBJECT_POINTS.to_vec(), images[..15].to_vec()),
        (vec![0.0; 4097 * 3], vec![0.0; 4097 * 2]),
    ] {
        assert_rejected(&objects, &images);
        assert_refinement_rejected(&objects, &images);
    }
    let minimal_images = project(&MINIMAL_OBJECT_POINTS, ROTATION, TRANSLATION);
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
        assert!(solve_minimal(&objects, &images).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let planar_images = project(&PLANAR_OBJECT_POINTS, ROTATION, TRANSLATION);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut objects = PLANAR_OBJECT_POINTS;
        objects[0] = invalid;
        assert!(solve_planar(&objects, &planar_images).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        let mut images = planar_images.clone();
        images[0] = invalid;
        assert!(solve_planar(&PLANAR_OBJECT_POINTS, &images).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut objects = MINIMAL_OBJECT_POINTS;
        objects[0] = invalid;
        assert!(solve_minimal(&objects, &minimal_images).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        let mut images = minimal_images.clone();
        images[0] = invalid;
        assert!(solve_minimal(&MINIMAL_OBJECT_POINTS, &images).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
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
        assert!(solve_planar(&objects, &images).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let square_images = project(&SQUARE_OBJECT_POINTS, ROTATION, TRANSLATION);
    for images in [
        square_images[..6].to_vec(),
        [square_images.clone(), vec![50.0, 50.0]].concat(),
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
    ] {
        assert!(solve_square_marker(&images, 2.0).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for marker_size in [
        0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1_000_001.0,
    ] {
        assert!(solve_square_marker(&square_images, marker_size).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut invalid_images = square_images.clone();
        invalid_images[0] = invalid;
        assert!(solve_square_marker(&invalid_images, 2.0).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut objects = OBJECT_POINTS;
        objects[0] = invalid;
        assert_rejected(&objects, &images);
        let mut invalid_images = images.clone();
        invalid_images[0] = invalid;
        assert_rejected(&OBJECT_POINTS, &invalid_images);
        let mut pose = [0.1, -0.2, 0.05, 0.25, -0.4, 8.0];
        pose[0] = invalid;
        for vvs in [false, true] {
            assert!(refine(&OBJECT_POINTS, &images, pose, vvs).is_null());
            assert_eq!(ffi::take_last_failure(), 1);
        }
        assert!(compose(pose, [0.0; 6]).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(compose([0.0; 6], pose).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
}

/// Produces synthetic image observations through the direct projection bridge.
fn project(objects: &[f64], rotation: [f64; 3], translation: [f64; 3]) -> Vec<f64> {
    ffi::project_points_owned(
        objects,
        rotation[0],
        rotation[1],
        rotation[2],
        translation[0],
        translation[1],
        translation[2],
        100.0,
        100.0,
        50.0,
        50.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    .as_ref()
    .expect("synthetic image points")
    .iter()
    .copied()
    .collect()
}

/// Invokes the fixed-policy direct pose solver.
fn solve(objects: &[f64], images: &[f64]) -> cxx::UniquePtr<ffi::CameraPose> {
    ffi::solve_pose_owned(
        objects, images, 100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
}

/// Invokes the fixed-policy direct robust pose solver.
fn solve_robust(objects: &[f64], images: &[f64]) -> cxx::UniquePtr<ffi::CameraPose> {
    ffi::solve_pose_robust_owned(
        objects, images, 100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
}

/// Invokes direct fixed-policy four-correspondence pose solving.
fn solve_minimal(objects: &[f64], images: &[f64]) -> cxx::UniquePtr<ffi::CameraPose> {
    ffi::solve_pose_minimal_owned(
        objects, images, 100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
}

/// Invokes direct fixed-policy planar pose solving.
fn solve_planar(objects: &[f64], images: &[f64]) -> cxx::UniquePtr<ffi::CameraPose> {
    ffi::solve_pose_planar_owned(
        objects, images, 100.0, 100.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
}

/// Invokes direct fixed-policy square-marker pose solving.
fn solve_square_marker(images: &[f64], marker_size: f64) -> cxx::UniquePtr<ffi::CameraPose> {
    ffi::solve_square_marker_pose_owned(
        images,
        marker_size,
        100.0,
        100.0,
        50.0,
        50.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
}

/// Invokes the fixed-policy direct pose refiner.
fn refine(
    objects: &[f64],
    images: &[f64],
    pose: [f64; 6],
    vvs: bool,
) -> cxx::UniquePtr<ffi::CameraPose> {
    if vvs {
        ffi::refine_pose_vvs_owned(
            objects, images, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], 100.0, 100.0,
            50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
    } else {
        ffi::refine_pose_owned(
            objects, images, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], 100.0, 100.0,
            50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
    }
}

/// Invokes direct copied-pose composition.
fn compose(first: [f64; 6], second: [f64; 6]) -> cxx::UniquePtr<ffi::CameraPose> {
    ffi::compose_pose_owned(
        first[0], first[1], first[2], first[3], first[4], first[5], second[0], second[1],
        second[2], second[3], second[4], second[5],
    )
}

/// Requires one malformed direct call to fail finitely.
fn assert_rejected(objects: &[f64], images: &[f64]) {
    assert!(solve(objects, images).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(solve_robust(objects, images).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Requires malformed correspondences to fail under both refinement policies.
fn assert_refinement_rejected(objects: &[f64], images: &[f64]) {
    for vvs in [false, true] {
        assert!(refine(objects, images, [0.1, -0.2, 0.05, 0.25, -0.4, 8.0], vvs).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
}

/// Copies all pose fields from one non-null native result.
fn pose_values(pose: &cxx::UniquePtr<ffi::CameraPose>) -> [f64; 6] {
    let pose = pose.as_ref().expect("camera pose");
    [
        pose.rx_value(),
        pose.ry_value(),
        pose.rz_value(),
        pose.tx_value(),
        pose.ty_value(),
        pose.tz_value(),
    ]
}

/// Requires pose values to agree within solver tolerance.
fn assert_pose(actual: &cxx::UniquePtr<ffi::CameraPose>, expected: &[f64; 6]) {
    for (actual, expected) in pose_values(actual).iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }
}
