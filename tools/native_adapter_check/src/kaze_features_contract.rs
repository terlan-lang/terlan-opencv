//! Owned KAZE keypoint and floating-point descriptor contract.

use super::{
    checkerboard_fixture, expect_bytes, expect_error_code, expect_floats, expect_int,
    expect_typed_handle, expect_unit, float_arg, int_arg, matrix, validate_feature_points,
    NativeHandle, NativeHelper,
};

const KAZE_TYPE: &str = "opencv.Features.KazeFeatures";

/// Runs bounded extraction, row alignment, ownership, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.features.live_kaze_count", &[])?, 0)?;
    let fixture = checkerboard_fixture();
    let image = matrix(helper, &fixture, 160, 160, "gray")?;
    let first = detect(helper, &image, 0.001, 64)?;
    let second = detect(helper, &image, 0.001, 64)?;
    expect_int(helper.call("opencv.features.live_kaze_count", &[])?, 2)?;
    let count = feature_count(helper, &first)?;
    if !(1..=64).contains(&count) {
        return Err(format!("unexpected KAZE feature count {count}"));
    }
    let points = expect_floats(helper.call("opencv.features.kaze_keypoints", &[first.arg()])?)?;
    validate_feature_points("KAZE", &points, count as usize, 160.0, 160.0)?;
    expect_int(
        helper.call("opencv.features.kaze_descriptor_width", &[first.arg()])?,
        64,
    )?;
    let descriptors =
        expect_floats(helper.call("opencv.features.kaze_descriptors", &[first.arg()])?)?;
    if descriptors.len() != count as usize * 64
        || descriptors.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "invalid KAZE descriptor projection of length {}",
            descriptors.len()
        ));
    }
    let repeated =
        expect_floats(helper.call("opencv.features.kaze_descriptors", &[second.arg()])?)?;
    if feature_count(helper, &second)? != count || repeated != descriptors {
        return Err("repeated KAZE extraction was not deterministic".to_owned());
    }
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, &fixture)?;

    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)?;
    expect_int(
        helper.call("opencv.features.kaze_count", &[first.arg()])?,
        count,
    )?;

    let empty_image = matrix(helper, &[0; 160 * 160], 160, 160, "gray")?;
    let empty = detect(helper, &empty_image, 0.001, 8)?;
    expect_int(
        helper.call("opencv.features.kaze_count", &[empty.arg()])?,
        0,
    )?;
    expect_int(
        helper.call("opencv.features.kaze_descriptor_width", &[empty.arg()])?,
        64,
    )?;
    if !expect_floats(helper.call("opencv.features.kaze_keypoints", &[empty.arg()])?)?.is_empty()
        || !expect_floats(helper.call("opencv.features.kaze_descriptors", &[empty.arg()])?)?
            .is_empty()
    {
        return Err("zero image produced KAZE features".to_owned());
    }

    for (threshold, maximum) in [
        (0.0, 8),
        (-0.1, 8),
        (1.1, 8),
        (f64::NAN, 8),
        (f64::INFINITY, 8),
        (0.001, 0),
        (0.001, 4097),
    ] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_kaze",
                &[empty_image.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.features.detect_kaze",
            &[
                empty_image.arg_with_owner("foreign-worker"),
                float_arg(0.001),
                int_arg(8),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.features.detect_kaze",
            &[
                empty_image.arg_with_type("other.Package.Resource"),
                float_arg(0.001),
                int_arg(8),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.features.kaze_count", &[empty_image.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.mat.rows", &[empty.arg()])?,
        "handle_type_mismatch",
    )?;

    let color = matrix(helper, &[0; 3], 1, 1, "bgr")?;
    let oversized = matrix(helper, &[0; 1025 * 1024], 1025, 1024, "gray")?;
    for source in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_kaze",
                &[source.arg(), float_arg(0.001), int_arg(8)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for features in [&first, &second, &empty] {
        expect_unit(helper.call("opencv.features.dispose_kaze", &[features.arg()])?)?;
        expect_error_code(
            helper.call("opencv.features.kaze_count", &[features.arg()])?,
            "stale_handle",
        )?;
    }
    for source in [&oversized, &color, &empty_image] {
        expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    }
    expect_int(helper.call("opencv.features.live_kaze_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Extracts one generated KAZE analysis handle.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.detect_kaze",
            &[image.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        KAZE_TYPE,
    )
}

/// Reads the row-aligned KAZE feature count.
fn feature_count(helper: &mut NativeHelper, features: &NativeHandle) -> Result<i64, String> {
    let reply = helper.call("opencv.features.kaze_count", &[features.arg()])?;
    match reply {
        crate::protocol::NativeReply::Int(value) => Ok(value),
        other => Err(format!("expected KAZE count, received {other:?}")),
    }
}
