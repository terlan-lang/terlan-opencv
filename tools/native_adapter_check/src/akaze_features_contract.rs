//! Owned AKAZE keypoint and MLDB descriptor contract.

use super::{
    checkerboard_fixture, expect_bytes, expect_bytes_value, expect_error_code, expect_floats,
    expect_int, expect_typed_handle, expect_unit, float_arg, int_arg, matrix,
    validate_feature_points, NativeHandle, NativeHelper,
};

const AKAZE_TYPE: &str = "opencv.Features.AkazeFeatures";

/// Runs bounded extraction, row alignment, ownership, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.features.live_akaze_count", &[])?, 0)?;
    let fixture = checkerboard_fixture();
    let image = matrix(helper, &fixture, 160, 160, "gray")?;
    let first = detect(helper, &image, 0.001, 64)?;
    let second = detect(helper, &image, 0.001, 64)?;
    expect_int(helper.call("opencv.features.live_akaze_count", &[])?, 2)?;
    let count = feature_count(helper, &first)?;
    if !(1..=64).contains(&count) {
        return Err(format!("unexpected AKAZE feature count {count}"));
    }
    let points = expect_floats(helper.call("opencv.features.akaze_keypoints", &[first.arg()])?)?;
    validate_feature_points("AKAZE", &points, count as usize, 160.0, 160.0)?;
    expect_int(
        helper.call("opencv.features.akaze_descriptor_width", &[first.arg()])?,
        61,
    )?;
    let descriptors =
        expect_bytes_value(helper.call("opencv.features.akaze_descriptors", &[first.arg()])?)?;
    if descriptors.len() != count as usize * 61 {
        return Err(format!(
            "AKAZE descriptor length {} does not match {count} rows",
            descriptors.len()
        ));
    }
    expect_bytes(
        helper.call("opencv.features.akaze_descriptors", &[first.arg()])?,
        &descriptors,
    )?;
    let second_count = feature_count(helper, &second)?;
    if second_count != count {
        return Err(format!(
            "repeated AKAZE extraction changed count from {count} to {second_count}"
        ));
    }
    expect_bytes(
        helper.call("opencv.features.akaze_descriptors", &[second.arg()])?,
        &descriptors,
    )?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[image.arg()])?, &fixture)?;

    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)?;
    expect_int(
        helper.call("opencv.features.akaze_count", &[first.arg()])?,
        count,
    )?;
    expect_bytes(
        helper.call("opencv.features.akaze_descriptors", &[first.arg()])?,
        &descriptors,
    )?;

    let empty_image = matrix(helper, &[0; 160 * 160], 160, 160, "gray")?;
    let empty = detect(helper, &empty_image, 0.001, 8)?;
    expect_int(
        helper.call("opencv.features.akaze_count", &[empty.arg()])?,
        0,
    )?;
    expect_int(
        helper.call("opencv.features.akaze_descriptor_width", &[empty.arg()])?,
        61,
    )?;
    if !expect_floats(helper.call("opencv.features.akaze_keypoints", &[empty.arg()])?)?.is_empty() {
        return Err("zero image produced AKAZE keypoints".to_owned());
    }
    expect_bytes(
        helper.call("opencv.features.akaze_descriptors", &[empty.arg()])?,
        &[],
    )?;

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
                "opencv.features.detect_akaze",
                &[empty_image.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.features.detect_akaze",
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
            "opencv.features.detect_akaze",
            &[
                empty_image.arg_with_type("other.Package.Resource"),
                float_arg(0.001),
                int_arg(8),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.features.akaze_count", &[empty_image.arg()])?,
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
                "opencv.features.detect_akaze",
                &[source.arg(), float_arg(0.001), int_arg(8)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for features in [&first, &second, &empty] {
        expect_unit(helper.call("opencv.features.dispose_akaze", &[features.arg()])?)?;
        expect_error_code(
            helper.call("opencv.features.akaze_count", &[features.arg()])?,
            "stale_handle",
        )?;
    }
    for source in [&oversized, &color, &empty_image] {
        expect_unit(helper.call("opencv.mat.dispose", &[source.arg()])?)?;
    }
    expect_int(helper.call("opencv.features.live_akaze_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Extracts one generated AKAZE analysis handle.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.detect_akaze",
            &[image.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        AKAZE_TYPE,
    )
}

/// Reads the row-aligned AKAZE feature count.
fn feature_count(helper: &mut NativeHelper, features: &NativeHandle) -> Result<i64, String> {
    let reply = helper.call("opencv.features.akaze_count", &[features.arg()])?;
    match reply {
        crate::protocol::NativeReply::Int(value) => Ok(value),
        other => Err(format!("expected AKAZE count, received {other:?}")),
    }
}
