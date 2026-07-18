//! Cross-checked SIFT descriptor matching contract.

use super::{
    checkerboard_fixture, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};
use crate::float_matching_contract_support::{validate_matches, MatchOperations};

const FEATURES_TYPE: &str = "opencv.Features.SiftFeatures";
const MATCHES_TYPE: &str = "opencv.Features.SiftMatches";
const MATCH_OPERATIONS: MatchOperations = MatchOperations {
    family: "SIFT",
    indices: "opencv.features.sift_match_indices",
    distances: "opencv.features.sift_match_distances",
    coordinates: "opencv.features.sift_match_coordinates",
};

/// Runs canonical matching, truncation, ownership, and rejection checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(
        helper.call("opencv.features.live_sift_match_count", &[])?,
        0,
    )?;
    let image = matrix(helper, &checkerboard_fixture(), 160, 160, "gray")?;
    let first = detect(helper, &image, 64)?;
    let second = detect(helper, &image, 64)?;
    let matches = match_sift(helper, &first, &second, 64)?;
    let count = match_count(helper, &matches)?;
    if !(1..=64).contains(&count) {
        return Err(format!("unexpected SIFT match count {count}"));
    }
    validate_matches(helper, &matches, &MATCH_OPERATIONS, count as usize, 64, 64)?;
    let truncated = match_sift(helper, &first, &second, 1)?;
    expect_int(
        helper.call("opencv.features.sift_match_count", &[truncated.arg()])?,
        1,
    )?;
    validate_matches(helper, &truncated, &MATCH_OPERATIONS, 1, 64, 64)?;

    expect_unit(helper.call("opencv.features.dispose_sift", &[first.arg()])?)?;
    expect_unit(helper.call("opencv.features.dispose_sift", &[second.arg()])?)?;
    validate_matches(helper, &matches, &MATCH_OPERATIONS, count as usize, 64, 64)?;

    let empty_image = matrix(helper, &[0; 160 * 160], 160, 160, "gray")?;
    let empty = detect(helper, &empty_image, 8)?;
    let empty_matches = match_sift(helper, &empty, &empty, 8)?;
    validate_matches(helper, &empty_matches, &MATCH_OPERATIONS, 0, 8, 8)?;
    for maximum in [0, 4097] {
        expect_error_code(
            helper.call(
                "opencv.features.match_sift",
                &[empty.arg(), empty.arg(), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.features.match_sift",
            &[empty.arg(), image.arg(), int_arg(8)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.features.match_sift",
            &[
                empty.arg(),
                empty.arg_with_owner("foreign-worker"),
                int_arg(8),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.features.sift_match_count", &[empty.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.features.sift_count", &[matches.arg()])?,
        "handle_type_mismatch",
    )?;

    for value in [&matches, &truncated, &empty_matches] {
        expect_unit(helper.call("opencv.features.dispose_sift_matches", &[value.arg()])?)?;
        expect_error_code(
            helper.call("opencv.features.sift_match_count", &[value.arg()])?,
            "stale_handle",
        )?;
    }
    expect_unit(helper.call("opencv.features.dispose_sift", &[empty.arg()])?)?;
    for value in [&empty_image, &image] {
        expect_unit(helper.call("opencv.mat.dispose", &[value.arg()])?)?;
    }
    expect_int(
        helper.call("opencv.features.live_sift_match_count", &[])?,
        0,
    )?;
    expect_int(helper.call("opencv.features.live_sift_count", &[])?, 0)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// Detects one SIFT feature resource.
fn detect(
    helper: &mut NativeHelper,
    image: &NativeHandle,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.detect_sift",
            &[image.arg(), float_arg(0.04), int_arg(maximum)],
        )?,
        FEATURES_TYPE,
    )
}

/// Creates one bounded cross-checked SIFT match resource.
fn match_sift(
    helper: &mut NativeHelper,
    query: &NativeHandle,
    train: &NativeHandle,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.features.match_sift",
            &[query.arg(), train.arg(), int_arg(maximum)],
        )?,
        MATCHES_TYPE,
    )
}

/// Reads the retained SIFT match count.
fn match_count(helper: &mut NativeHelper, matches: &NativeHandle) -> Result<i64, String> {
    let reply = helper.call("opencv.features.sift_match_count", &[matches.arg()])?;
    match reply {
        crate::protocol::NativeReply::Int(value) => Ok(value),
        other => Err(format!("expected SIFT match count, received {other:?}")),
    }
}
