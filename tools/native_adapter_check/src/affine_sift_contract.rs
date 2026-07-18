//! Owned affine-invariant SIFT extraction contract.

use super::{
    checkerboard_fixture, expect_bytes, expect_error_code, expect_floats, expect_int,
    expect_typed_handle, expect_unit, float_arg, int_arg, matrix, validate_feature_points,
    NativeHandle, NativeHelper,
};

const SIFT_TYPE: &str = "opencv.Features.SiftFeatures";
const SIFT_MATCH_TYPE: &str = "opencv.Features.SiftMatches";

/// Runs bounded extraction, shared-owner matching, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 0)?;
    let fixture = checkerboard_fixture();
    let image = matrix(helper, &fixture, 160, 160, "gray")?;
    let first = detect(helper, &image, 0.04, 64)?;
    let second = detect(helper, &image, 0.04, 64)?;
    let truncated = detect(helper, &image, 0.04, 1)?;
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 3)?;

    let count = feature_count(helper, &first)?;
    if !(1..=64).contains(&count) || feature_count(helper, &second)? != count {
        return Err(format!("unexpected affine SIFT feature count {count}"));
    }
    expect_int(
        helper.call("opencv.features.sift_descriptor_width", &[first.arg()])?,
        128,
    )?;
    let points = expect_floats(helper.call("opencv.features.sift_keypoints", &[first.arg()])?)?;
    validate_feature_points("affine SIFT", &points, count as usize, 160.0, 160.0)?;
    let descriptors =
        expect_floats(helper.call("opencv.features.sift_descriptors", &[first.arg()])?)?;
    let repeated =
        expect_floats(helper.call("opencv.features.sift_descriptors", &[second.arg()])?)?;
    if descriptors.len() != count as usize * 128
        || descriptors.iter().any(|value| !value.is_finite())
        || repeated != descriptors
    {
        return Err("invalid or nondeterministic affine SIFT descriptors".to_owned());
    }
    expect_int(
        helper.call("opencv.features.sift_count", &[truncated.arg()])?,
        1,
    )?;
    let matches = expect_typed_handle(
        helper.call(
            "opencv.features.match_sift",
            &[first.arg(), first.arg(), int_arg(64)],
        )?,
        SIFT_MATCH_TYPE,
    )?;
    let match_count =
        expect_int_value(helper.call("opencv.features.sift_match_count", &[matches.arg()])?)?;
    if !(1..=count).contains(&match_count) {
        return Err(format!("unexpected affine SIFT match count {match_count}"));
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
    if !expect_floats(helper.call("opencv.features.sift_keypoints", &[empty.arg()])?)?.is_empty()
        || !expect_floats(helper.call("opencv.features.sift_descriptors", &[empty.arg()])?)?
            .is_empty()
    {
        return Err("zero image produced affine SIFT features".to_owned());
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
                "opencv.features.detect_affine_sift",
                &[empty_image.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (argument, code) in [
        (
            empty_image.arg_with_owner("foreign-worker"),
            "cross_owner_handle",
        ),
        (
            empty_image.arg_with_type("other.Package.Resource"),
            "handle_type_mismatch",
        ),
    ] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_affine_sift",
                &[argument, float_arg(0.04), int_arg(8)],
            )?,
            code,
        )?;
    }
    let color = matrix(helper, &[0; 3], 1, 1, "bgr")?;
    let oversized = matrix(helper, &[0; 1025 * 1024], 1025, 1024, "gray")?;
    for source in [&color, &oversized] {
        expect_error_code(
            helper.call(
                "opencv.features.detect_affine_sift",
                &[source.arg(), float_arg(0.04), int_arg(8)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    expect_unit(helper.call("opencv.features.dispose_sift_matches", &[matches.arg()])?)?;
    for features in [&first, &second, &truncated, &empty] {
        expect_unit(helper.call("opencv.features.dispose_sift", &[features.arg()])?)?;
        expect_error_code(
            helper.call("opencv.features.sift_count", &[features.arg()])?,
            "stale_handle",
        )?;
    }
    for source in [&oversized, &color, &empty_image] {
        expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    }
    expect_int(
        helper.call("opencv.features.live_sift_match_count", &[])?,
        0,
    )?;
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Extracts one affine SIFT analysis through the generated operation.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.detect_affine_sift",
            &[image.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        SIFT_TYPE,
    )
}

/// Reads one shared SIFT feature count.
fn feature_count(helper: &mut NativeHelper, features: &NativeHandle) -> Result<i64, String> {
    expect_int_value(helper.call("opencv.features.sift_count", &[features.arg()])?)
}

/// Extracts an integer from a native reply.
fn expect_int_value(reply: crate::protocol::NativeReply) -> Result<i64, String> {
    match reply {
        crate::protocol::NativeReply::Int(value) => Ok(value),
        other => Err(format!("expected integer reply, received {other:?}")),
    }
}
