//! Owned SIFT keypoint and floating-point descriptor contract.

use super::{
    checkerboard_fixture, expect_bytes, expect_error_code, expect_floats, expect_int,
    expect_typed_handle, expect_unit, float_arg, int_arg, matrix, validate_feature_points,
    NativeHandle, NativeHelper,
};

const SIFT_TYPE: &str = "opencv.Features.SiftFeatures";

/// Runs bounded extraction, row alignment, ownership, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 0)?;
    let fixture = checkerboard_fixture();
    let image = matrix(helper, &fixture, 160, 160, "gray")?;
    let first = detect(helper, &image, 0.04, 64)?;
    let second = detect(helper, &image, 0.04, 64)?;
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 2)?;
    let count = feature_count(helper, &first)?;
    if !(1..=64).contains(&count) {
        return Err(format!("unexpected SIFT feature count {count}"));
    }
    let points = expect_floats(helper.call("opencv.features.sift_keypoints", &[first.arg()])?)?;
    validate_feature_points("SIFT", &points, count as usize, 160.0, 160.0)?;
    expect_int(
        helper.call("opencv.features.sift_descriptor_width", &[first.arg()])?,
        128,
    )?;
    let descriptors =
        expect_floats(helper.call("opencv.features.sift_descriptors", &[first.arg()])?)?;
    if descriptors.len() != count as usize * 128
        || descriptors.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "invalid SIFT descriptor projection of length {}",
            descriptors.len()
        ));
    }
    let repeated =
        expect_floats(helper.call("opencv.features.sift_descriptors", &[second.arg()])?)?;
    if feature_count(helper, &second)? != count || repeated != descriptors {
        return Err("repeated SIFT extraction was not deterministic".to_owned());
    }
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, &fixture)?;

    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)?;
    expect_int(
        helper.call("opencv.features.sift_count", &[first.arg()])?,
        count,
    )?;

    let empty_image = matrix(helper, &[0; 160 * 160], 160, 160, "gray")?;
    let empty = detect(helper, &empty_image, 0.04, 8)?;
    expect_int(
        helper.call("opencv.features.sift_count", &[empty.arg()])?,
        0,
    )?;
    expect_int(
        helper.call("opencv.features.sift_descriptor_width", &[empty.arg()])?,
        128,
    )?;
    if !expect_floats(helper.call("opencv.features.sift_keypoints", &[empty.arg()])?)?.is_empty()
        || !expect_floats(helper.call("opencv.features.sift_descriptors", &[empty.arg()])?)?
            .is_empty()
    {
        return Err("zero image produced SIFT features".to_owned());
    }

    for (threshold, maximum) in [
        (0.0, 8),
        (-0.1, 8),
        (1.1, 8),
        (f64::NAN, 8),
        (f64::INFINITY, 8),
        (0.04, 0),
        (0.04, 4097),
    ] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_sift",
                &[empty_image.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.features.detect_sift",
            &[
                empty_image.arg_with_owner("foreign-worker"),
                float_arg(0.04),
                int_arg(8),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.features.detect_sift",
            &[
                empty_image.arg_with_type("other.Package.Resource"),
                float_arg(0.04),
                int_arg(8),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.features.sift_count", &[empty_image.arg()])?,
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
                "opencv.features.detect_sift",
                &[source.arg(), float_arg(0.04), int_arg(8)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for features in [&first, &second, &empty] {
        expect_unit(helper.call("opencv.features.dispose_sift", &[features.arg()])?)?;
        expect_error_code(
            helper.call("opencv.features.sift_count", &[features.arg()])?,
            "stale_handle",
        )?;
    }
    for source in [&oversized, &color, &empty_image] {
        expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    }
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Extracts one generated SIFT analysis handle.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.detect_sift",
            &[image.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        SIFT_TYPE,
    )
}

/// Reads the row-aligned SIFT feature count.
fn feature_count(helper: &mut NativeHelper, features: &NativeHandle) -> Result<i64, String> {
    let reply = helper.call("opencv.features.sift_count", &[features.arg()])?;
    match reply {
        crate::protocol::NativeReply::Int(value) => Ok(value),
        other => Err(format!("expected SIFT count, received {other:?}")),
    }
}
