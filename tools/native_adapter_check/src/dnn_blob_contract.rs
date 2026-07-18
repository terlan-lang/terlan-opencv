//! Fixed CPU DNN image-to-blob ownership contract.

use super::dnn_blob_contract_support::*;
use super::{
    expect_bytes, expect_error_code, expect_floats, expect_int, expect_unit, float_list_arg,
    int_arg, matrix, NativeHelper,
};

const BGR_PIXELS: [u8; 12] = [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255];
const GRAY_PIXELS: [u8; 4] = [0, 64, 128, 255];
const WIDE_BGR_PIXELS: [u8; 24] = [
    255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
];
const WIDE_GRAY_PIXELS: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
const EXPECTED_CENTER_CROP_RGB: [f64; 12] =
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const EXPECTED_CENTER_CROP_GRAY: [f64; 4] = [
    (20.0_f32 / 255.0_f32) as f64,
    (30.0_f32 / 255.0_f32) as f64,
    (60.0_f32 / 255.0_f32) as f64,
    (70.0_f32 / 255.0_f32) as f64,
];
const EXPECTED_LETTERBOX_GRAY: [f64; 16] = [
    0.0,
    0.0,
    0.0,
    0.0,
    (10.0_f32 / 255.0_f32) as f64,
    (20.0_f32 / 255.0_f32) as f64,
    (30.0_f32 / 255.0_f32) as f64,
    (40.0_f32 / 255.0_f32) as f64,
    (50.0_f32 / 255.0_f32) as f64,
    (60.0_f32 / 255.0_f32) as f64,
    (70.0_f32 / 255.0_f32) as f64,
    (80.0_f32 / 255.0_f32) as f64,
    0.0,
    0.0,
    0.0,
    0.0,
];
const EXPECTED_NCHW_RGB: [f64; 12] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const EXPECTED_NCHW_GRAY: [f64; 4] = [
    0.0,
    (64.0_f32 / 255.0_f32) as f64,
    (128.0_f32 / 255.0_f32) as f64,
    1.0,
];
const EXPECTED_NORMALIZED_GRAY: [f64; 4] = [
    -2.0,
    ((64.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((128.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    2.0,
];
const EXPECTED_NORMALIZED_CENTER_CROP_GRAY: [f64; 4] = [
    ((20.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((30.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((60.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((70.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
];
const EXPECTED_NORMALIZED_LETTERBOX_GRAY: [f64; 16] = [
    -2.0,
    -2.0,
    -2.0,
    -2.0,
    ((10.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((20.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((30.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((40.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((50.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((60.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((70.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    ((80.0_f32 - 127.5_f32) * (4.0_f32 / 255.0_f32)) as f64,
    -2.0,
    -2.0,
    -2.0,
    -2.0,
];
const EXPECTED_NORMALIZED_RGB: [f64; 12] = [
    4.0, 2.0, 2.0, 4.0, -2.0, 2.0, -2.0, 2.0, -8.0, -8.0, 0.0, 0.0,
];
const EXPECTED_NORMALIZED_CENTER_CROP_RGB: [f64; 12] = [
    2.0, 4.0, 2.0, 4.0, 2.0, -2.0, 2.0, -2.0, -8.0, -8.0, -8.0, -8.0,
];
const EXPECTED_LETTERBOX_RGB: [f64; 48] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
];
const EXPECTED_NORMALIZED_LETTERBOX_RGB: [f64; 48] = [
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0,
    -2.0, -2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -8.0, -8.0,
    -8.0, -8.0, 0.0, -8.0, -8.0, 0.0, 0.0, -8.0, -8.0, 0.0, -8.0, -8.0, -8.0, -8.0,
];

/// Runs exact-value, lifecycle, malformed-input, owner, and bound checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 0)?;
    let image = matrix(helper, &BGR_PIXELS, 2, 2, "bgr")?;
    let blob = create(helper, &image, 2, 2)?;
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 1)?;
    for (operation, expected) in [
        ("opencv.dnn.batch_count", 1),
        ("opencv.dnn.channel_count", 3),
        ("opencv.dnn.height", 2),
        ("opencv.dnn.width", 2),
    ] {
        expect_int(helper.call(operation, &[blob.arg()])?, expected)?;
    }
    let values = expect_floats(helper.call("opencv.dnn.values", &[blob.arg()])?)?;
    if !values_are_close(&values, &EXPECTED_NCHW_RGB) {
        return Err(format!("unexpected NCHW RGB values: {values:?}"));
    }
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[image.arg()])?,
        &BGR_PIXELS,
    )?;
    let normalized_blob = create_normalized(helper, &image, 2, 2)?;
    let normalized_values =
        expect_floats(helper.call("opencv.dnn.values", &[normalized_blob.arg()])?)?;
    if !values_are_close(&normalized_values, &EXPECTED_NORMALIZED_RGB) {
        return Err(format!(
            "unexpected normalized NCHW RGB values: {normalized_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose", &[normalized_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[normalized_blob.arg()])?,
        "stale_handle",
    )?;
    let grayscale = matrix(helper, &GRAY_PIXELS, 2, 2, "gray")?;
    let grayscale_blob = create_grayscale(helper, &grayscale, 2, 2)?;
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 2)?;
    for (operation, expected) in [
        ("opencv.dnn.batch_count", 1),
        ("opencv.dnn.channel_count", 1),
        ("opencv.dnn.height", 2),
        ("opencv.dnn.width", 2),
    ] {
        expect_int(helper.call(operation, &[grayscale_blob.arg()])?, expected)?;
    }
    let grayscale_values =
        expect_floats(helper.call("opencv.dnn.values", &[grayscale_blob.arg()])?)?;
    if !values_are_close(&grayscale_values, &EXPECTED_NCHW_GRAY) {
        return Err(format!(
            "unexpected NCHW grayscale values: {grayscale_values:?}"
        ));
    }
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[grayscale.arg()])?,
        &GRAY_PIXELS,
    )?;
    let normalized_grayscale_blob = create_normalized_grayscale(helper, &grayscale, 2, 2)?;
    let normalized_grayscale_values =
        expect_floats(helper.call("opencv.dnn.values", &[normalized_grayscale_blob.arg()])?)?;
    if !values_are_close(&normalized_grayscale_values, &EXPECTED_NORMALIZED_GRAY) {
        return Err(format!(
            "unexpected normalized grayscale values: {normalized_grayscale_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose", &[normalized_grayscale_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[normalized_grayscale_blob.arg()])?,
        "stale_handle",
    )?;
    let wide_image = matrix(helper, &WIDE_BGR_PIXELS, 2, 4, "bgr")?;
    let cropped_blob = create_center_crop(helper, &wide_image, 2, 2)?;
    let cropped_values = expect_floats(helper.call("opencv.dnn.values", &[cropped_blob.arg()])?)?;
    if !values_are_close(&cropped_values, &EXPECTED_CENTER_CROP_RGB) {
        return Err(format!(
            "unexpected center-cropped RGB values: {cropped_values:?}"
        ));
    }
    let letterbox_blob = create_letterbox(helper, &wide_image, 4, 4)?;
    for (operation, expected) in [
        ("opencv.dnn.batch_count", 1),
        ("opencv.dnn.channel_count", 3),
        ("opencv.dnn.height", 4),
        ("opencv.dnn.width", 4),
    ] {
        expect_int(helper.call(operation, &[letterbox_blob.arg()])?, expected)?;
    }
    let letterbox_values =
        expect_floats(helper.call("opencv.dnn.values", &[letterbox_blob.arg()])?)?;
    if !values_are_close(&letterbox_values, &EXPECTED_LETTERBOX_RGB) {
        return Err(format!(
            "unexpected letterboxed RGB values: {letterbox_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose", &[letterbox_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[letterbox_blob.arg()])?,
        "stale_handle",
    )?;
    let normalized_letterbox_blob = create_normalized_letterbox(helper, &wide_image, 4, 4)?;
    let normalized_letterbox_values =
        expect_floats(helper.call("opencv.dnn.values", &[normalized_letterbox_blob.arg()])?)?;
    if !values_are_close(
        &normalized_letterbox_values,
        &EXPECTED_NORMALIZED_LETTERBOX_RGB,
    ) {
        return Err(format!(
            "unexpected normalized letterboxed RGB values: {normalized_letterbox_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose", &[normalized_letterbox_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[normalized_letterbox_blob.arg()])?,
        "stale_handle",
    )?;
    let normalized_cropped_blob = create_normalized_center_crop(helper, &wide_image, 2, 2)?;
    let normalized_cropped_values =
        expect_floats(helper.call("opencv.dnn.values", &[normalized_cropped_blob.arg()])?)?;
    if !values_are_close(
        &normalized_cropped_values,
        &EXPECTED_NORMALIZED_CENTER_CROP_RGB,
    ) {
        return Err(format!(
            "unexpected normalized center-cropped RGB values: {normalized_cropped_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose", &[normalized_cropped_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[normalized_cropped_blob.arg()])?,
        "stale_handle",
    )?;
    let wide_grayscale = matrix(helper, &WIDE_GRAY_PIXELS, 2, 4, "gray")?;
    let cropped_grayscale_blob = create_grayscale_center_crop(helper, &wide_grayscale, 2, 2)?;
    let cropped_grayscale_values =
        expect_floats(helper.call("opencv.dnn.values", &[cropped_grayscale_blob.arg()])?)?;
    if !values_are_close(&cropped_grayscale_values, &EXPECTED_CENTER_CROP_GRAY) {
        return Err(format!(
            "unexpected center-cropped grayscale values: {cropped_grayscale_values:?}"
        ));
    }
    let normalized_cropped_grayscale_blob =
        create_normalized_grayscale_center_crop(helper, &wide_grayscale, 2, 2)?;
    let normalized_cropped_grayscale_values = expect_floats(helper.call(
        "opencv.dnn.values",
        &[normalized_cropped_grayscale_blob.arg()],
    )?)?;
    if !values_are_close(
        &normalized_cropped_grayscale_values,
        &EXPECTED_NORMALIZED_CENTER_CROP_GRAY,
    ) {
        return Err(format!(
            "unexpected normalized center-cropped grayscale values: {normalized_cropped_grayscale_values:?}"
        ));
    }
    expect_unit(helper.call(
        "opencv.dnn.dispose",
        &[normalized_cropped_grayscale_blob.arg()],
    )?)?;
    expect_error_code(
        helper.call(
            "opencv.dnn.values",
            &[normalized_cropped_grayscale_blob.arg()],
        )?,
        "stale_handle",
    )?;
    let letterbox_grayscale_blob = create_grayscale_letterbox(helper, &wide_grayscale, 4, 4)?;
    for (operation, expected) in [
        ("opencv.dnn.batch_count", 1),
        ("opencv.dnn.channel_count", 1),
        ("opencv.dnn.height", 4),
        ("opencv.dnn.width", 4),
    ] {
        expect_int(
            helper.call(operation, &[letterbox_grayscale_blob.arg()])?,
            expected,
        )?;
    }
    let letterbox_grayscale_values =
        expect_floats(helper.call("opencv.dnn.values", &[letterbox_grayscale_blob.arg()])?)?;
    if !values_are_close(&letterbox_grayscale_values, &EXPECTED_LETTERBOX_GRAY) {
        return Err(format!(
            "unexpected letterboxed grayscale values: {letterbox_grayscale_values:?}"
        ));
    }
    let normalized_letterbox_grayscale_blob =
        create_normalized_grayscale_letterbox(helper, &wide_grayscale, 4, 4)?;
    let normalized_letterbox_grayscale_values = expect_floats(helper.call(
        "opencv.dnn.values",
        &[normalized_letterbox_grayscale_blob.arg()],
    )?)?;
    if !values_are_close(
        &normalized_letterbox_grayscale_values,
        &EXPECTED_NORMALIZED_LETTERBOX_GRAY,
    ) {
        return Err(format!(
            "unexpected normalized letterboxed grayscale values: {normalized_letterbox_grayscale_values:?}"
        ));
    }
    expect_unit(helper.call(
        "opencv.dnn.dispose",
        &[normalized_letterbox_grayscale_blob.arg()],
    )?)?;
    expect_error_code(
        helper.call(
            "opencv.dnn.values",
            &[normalized_letterbox_grayscale_blob.arg()],
        )?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose", &[letterbox_grayscale_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[letterbox_grayscale_blob.arg()])?,
        "stale_handle",
    )?;
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 4)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[wide_image.arg()])?,
        &WIDE_BGR_PIXELS,
    )?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[wide_grayscale.arg()])?,
        &WIDE_GRAY_PIXELS,
    )?;

    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[
                image.arg_with_owner("foreign-worker"),
                int_arg(2),
                int_arg(2),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_center_crop",
            &[
                wide_image.arg_with_owner("foreign-worker"),
                int_arg(2),
                int_arg(2),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_letterbox",
            &[
                wide_image.arg_with_owner("foreign-worker"),
                int_arg(4),
                int_arg(4),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_normalized",
            &[
                image.arg_with_owner("foreign-worker"),
                int_arg(2),
                int_arg(2),
                float_list_arg(&RGB_MEANS),
                float_list_arg(&RGB_SCALES),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
            &[
                wide_image.arg_with_owner("foreign-worker"),
                int_arg(4),
                int_arg(4),
                float_list_arg(&RGB_MEANS),
                float_list_arg(&RGB_SCALES),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_center_crop",
            &[
                wide_grayscale.arg_with_owner("foreign-worker"),
                int_arg(2),
                int_arg(2),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_letterbox",
            &[
                wide_grayscale.arg_with_owner("foreign-worker"),
                int_arg(4),
                int_arg(4),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale",
            &[
                grayscale.arg_with_owner("foreign-worker"),
                int_arg(2),
                int_arg(2),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_normalized",
            &[
                grayscale.arg_with_owner("foreign-worker"),
                int_arg(2),
                int_arg(2),
                float_list_arg(&GRAYSCALE_MEAN),
                float_list_arg(&GRAYSCALE_SCALE),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_normalized_grayscale_mode_errors(
        helper,
        wide_grayscale.arg_with_owner("foreign-worker"),
        2,
        2,
        &GRAYSCALE_MEAN,
        &GRAYSCALE_SCALE,
        "cross_owner_handle",
    )?;
    for (width, height) in [(0, 2), (2, 0), (-1, 2), (2, -1), (i64::MAX, 2)] {
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb",
                &[image.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_center_crop",
                &[wide_image.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_letterbox",
                &[wide_image.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_grayscale_center_crop",
                &[wide_grayscale.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_grayscale_letterbox",
                &[wide_grayscale.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_grayscale",
                &[grayscale.arg(), int_arg(width), int_arg(height)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_grayscale_normalized",
                &[
                    grayscale.arg(),
                    int_arg(width),
                    int_arg(height),
                    float_list_arg(&GRAYSCALE_MEAN),
                    float_list_arg(&GRAYSCALE_SCALE),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_normalized_grayscale_mode_errors(
            helper,
            wide_grayscale.arg(),
            width,
            height,
            &GRAYSCALE_MEAN,
            &GRAYSCALE_SCALE,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_normalized",
                &[
                    image.arg(),
                    int_arg(width),
                    int_arg(height),
                    float_list_arg(&RGB_MEANS),
                    float_list_arg(&RGB_SCALES),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
                &[
                    wide_image.arg(),
                    int_arg(width),
                    int_arg(height),
                    float_list_arg(&RGB_MEANS),
                    float_list_arg(&RGB_SCALES),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (means, scales) in [
        (&[][..], &RGB_SCALES[..]),
        (&RGB_MEANS[..2], &RGB_SCALES[..]),
        (&RGB_MEANS[..], &RGB_SCALES[..2]),
        (&[f64::NAN, 0.0, 0.0][..], &RGB_SCALES[..]),
        (&RGB_MEANS[..], &[1.0, 0.0, 1.0][..]),
    ] {
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_normalized",
                &[
                    image.arg(),
                    int_arg(2),
                    int_arg(2),
                    float_list_arg(means),
                    float_list_arg(scales),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_normalized_center_crop",
                &[
                    wide_image.arg(),
                    int_arg(2),
                    int_arg(2),
                    float_list_arg(means),
                    float_list_arg(scales),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
                &[
                    wide_image.arg(),
                    int_arg(2),
                    int_arg(2),
                    float_list_arg(means),
                    float_list_arg(scales),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (means, scales) in [
        (&[][..], &GRAYSCALE_SCALE[..]),
        (&[0.5, 0.0][..], &GRAYSCALE_SCALE[..]),
        (&GRAYSCALE_MEAN[..], &[][..]),
        (&GRAYSCALE_MEAN[..], &[4.0, 1.0][..]),
        (&[f64::NAN][..], &GRAYSCALE_SCALE[..]),
        (&GRAYSCALE_MEAN[..], &[0.0][..]),
    ] {
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_grayscale_normalized",
                &[
                    grayscale.arg(),
                    int_arg(2),
                    int_arg(2),
                    float_list_arg(means),
                    float_list_arg(scales),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_normalized_grayscale_mode_errors(
            helper,
            wide_grayscale.arg(),
            2,
            2,
            means,
            scales,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_center_crop",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_letterbox",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_normalized",
            &[
                image.arg(),
                int_arg(2),
                int_arg(2),
                float_list_arg(&GRAYSCALE_MEAN),
                float_list_arg(&GRAYSCALE_SCALE),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_normalized_grayscale_mode_errors(
        helper,
        image.arg(),
        2,
        2,
        &GRAYSCALE_MEAN,
        &GRAYSCALE_SCALE,
        "opencv.invalid_argument",
    )?;
    for layout in ["gray", "bgra"] {
        let channels = if layout == "gray" { 1 } else { 4 };
        let other = matrix(helper, &vec![0; 4 * channels], 2, 2, layout)?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb",
                &[other.arg(), int_arg(2), int_arg(2)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_center_crop",
                &[other.arg(), int_arg(2), int_arg(2)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_letterbox",
                &[other.arg(), int_arg(2), int_arg(2)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
                &[
                    other.arg(),
                    int_arg(2),
                    int_arg(2),
                    float_list_arg(&RGB_MEANS),
                    float_list_arg(&RGB_SCALES),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
        if layout == "bgra" {
            expect_error_code(
                helper.call(
                    "opencv.dnn.blob_from_grayscale",
                    &[other.arg(), int_arg(2), int_arg(2)],
                )?,
                "opencv.invalid_argument",
            )?;
            expect_error_code(
                helper.call(
                    "opencv.dnn.blob_from_grayscale_center_crop",
                    &[other.arg(), int_arg(2), int_arg(2)],
                )?,
                "opencv.invalid_argument",
            )?;
            expect_error_code(
                helper.call(
                    "opencv.dnn.blob_from_grayscale_letterbox",
                    &[other.arg(), int_arg(2), int_arg(2)],
                )?,
                "opencv.invalid_argument",
            )?;
            expect_error_code(
                helper.call(
                    "opencv.dnn.blob_from_grayscale_normalized",
                    &[
                        other.arg(),
                        int_arg(2),
                        int_arg(2),
                        float_list_arg(&GRAYSCALE_MEAN),
                        float_list_arg(&GRAYSCALE_SCALE),
                    ],
                )?,
                "opencv.invalid_argument",
            )?;
            expect_normalized_grayscale_mode_errors(
                helper,
                other.arg(),
                2,
                2,
                &GRAYSCALE_MEAN,
                &GRAYSCALE_SCALE,
                "opencv.invalid_argument",
            )?;
        }
        expect_unit(helper.call("opencv.mat.dispose", &[other.arg()])?)?;
    }
    expect_error_code(
        helper.call("opencv.dnn.values", &[image.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_center_crop",
            &[blob.arg(), int_arg(2), int_arg(2)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_letterbox",
            &[blob.arg(), int_arg(2), int_arg(2)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_letterbox",
            &[blob.arg(), int_arg(2), int_arg(2)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_normalized",
            &[
                blob.arg(),
                int_arg(2),
                int_arg(2),
                float_list_arg(&GRAYSCALE_MEAN),
                float_list_arg(&GRAYSCALE_SCALE),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_normalized_grayscale_mode_errors(
        helper,
        blob.arg(),
        2,
        2,
        &GRAYSCALE_MEAN,
        &GRAYSCALE_SCALE,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_normalized",
            &[
                blob.arg(),
                int_arg(2),
                int_arg(2),
                float_list_arg(&RGB_MEANS),
                float_list_arg(&RGB_SCALES),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
            &[
                blob.arg(),
                int_arg(2),
                int_arg(2),
                float_list_arg(&RGB_MEANS),
                float_list_arg(&RGB_SCALES),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), "a:Mg==".to_owned(), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale",
            &[grayscale.arg(), "a:Mg==".to_owned(), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_center_crop",
            &[wide_image.arg(), "a:Mg==".to_owned(), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_letterbox",
            &[wide_image.arg(), "a:Mg==".to_owned(), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_center_crop",
            &[wide_grayscale.arg(), "a:Mg==".to_owned(), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_letterbox",
            &[wide_grayscale.arg(), "a:Mg==".to_owned(), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_normalized",
            &[
                image.arg(),
                int_arg(2),
                int_arg(2),
                "lf:1,,2".to_owned(),
                float_list_arg(&RGB_SCALES),
            ],
        )?,
        "invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_normalized",
            &[
                grayscale.arg(),
                int_arg(2),
                int_arg(2),
                "lf:0.5,,".to_owned(),
                float_list_arg(&GRAYSCALE_SCALE),
            ],
        )?,
        "invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_normalized_center_crop",
            &[
                wide_grayscale.arg(),
                int_arg(2),
                int_arg(2),
                "lf:0.5,,".to_owned(),
                float_list_arg(&GRAYSCALE_SCALE),
            ],
        )?,
        "invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_grayscale_normalized_letterbox",
            &[
                wide_grayscale.arg(),
                int_arg(4),
                int_arg(4),
                "lf:0.5,,".to_owned(),
                float_list_arg(&GRAYSCALE_SCALE),
            ],
        )?,
        "invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb_normalized_letterbox",
            &[
                wide_image.arg(),
                int_arg(4),
                int_arg(4),
                "lf:1,,2".to_owned(),
                float_list_arg(&RGB_SCALES),
            ],
        )?,
        "invalid_argument",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 3)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[blob.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose", &[grayscale_blob.arg()])?)?;
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 2)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[grayscale_blob.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose", &[cropped_grayscale_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[cropped_grayscale_blob.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose", &[cropped_blob.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.values", &[cropped_blob.arg()])?,
        "stale_handle",
    )?;
    expect_int(helper.call("opencv.dnn.live_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.mat.dispose", &[wide_grayscale.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[wide_image.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[grayscale.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Compares copied DNN values within one conservative `f32` rounding envelope.
fn values_are_close(actual: &[f64], expected: &[f64]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-6)
}
