//! Bounded FAST keypoint detection contract.

use super::{
    bool_arg, detector_fixture, expect_bytes, expect_error_code, expect_floats, expect_int,
    expect_unit, int_arg, matrix, validate_detector_points, NativeHelper,
};

/// Runs detection, ordering, truncation, rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let fixture = detector_fixture();
    let image = matrix(helper, &fixture, 32, 32, "gray")?;
    let points = detect(helper, &image, 10, true, 8)?;
    validate_detector_points("FAST", &points, 1, 8, 32.0, 32.0)?;
    let one = detect(helper, &image, 10, true, 1)?;
    validate_detector_points("FAST", &one, 1, 1, 32.0, 32.0)?;
    let unsuppressed = detect(helper, &image, 10, false, 8)?;
    validate_detector_points("FAST", &unsuppressed, 1, 8, 32.0, 32.0)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, &fixture)?;

    let empty = matrix(helper, &[0; 1024], 32, 32, "gray")?;
    let empty_points = detect(helper, &empty, 10, true, 8)?;
    if !empty_points.is_empty() {
        return Err(format!("zero image produced FAST points {empty_points:?}"));
    }

    let color = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    let oversized = matrix(helper, &[0; 1025 * 1024], 1025, 1024, "gray")?;
    for arguments in [
        vec![image.arg(), int_arg(0), bool_arg(true), int_arg(1)],
        vec![image.arg(), int_arg(256), bool_arg(true), int_arg(1)],
        vec![image.arg(), int_arg(10), bool_arg(true), int_arg(0)],
        vec![image.arg(), int_arg(10), bool_arg(true), int_arg(4097)],
        vec![color.arg(), int_arg(10), bool_arg(true), int_arg(1)],
        vec![oversized.arg(), int_arg(10), bool_arg(true), int_arg(1)],
    ] {
        expect_error_code(
            helper.call("opencv.imgproc.fast_keypoints", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.fast_keypoints",
            &[
                image.arg_with_owner("foreign-worker"),
                int_arg(10),
                bool_arg(true),
                int_arg(1),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.fast_keypoints",
            &[
                image.arg_with_type("other.Package.Resource"),
                int_arg(10),
                bool_arg(true),
                int_arg(1),
            ],
        )?,
        "handle_type_mismatch",
    )?;

    let stale = matrix(helper, &fixture, 32, 32, "gray")?;
    expect_unit(helper.call("opencv.mat.dispose", &[stale.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.fast_keypoints",
            &[stale.arg(), int_arg(10), bool_arg(true), int_arg(1)],
        )?,
        "stale_handle",
    )?;

    for handle in [&oversized, &color, &empty, &image] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Detects one bounded set of copied FAST coordinates.
fn detect(
    helper: &mut NativeHelper,
    image: &super::NativeHandle,
    threshold: i64,
    nonmax_suppression: bool,
    max_keypoints: i64,
) -> Result<Vec<f64>, String> {
    expect_floats(helper.call(
        "opencv.imgproc.fast_keypoints",
        &[
            image.arg(),
            int_arg(threshold),
            bool_arg(nonmax_suppression),
            int_arg(max_keypoints),
        ],
    )?)
}
