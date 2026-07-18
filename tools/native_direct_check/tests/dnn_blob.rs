//! Direct CPU DNN preprocessing tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

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
const EXPECTED_NCHW_GRAY: [f64; 4] = [
    0.0,
    (64.0_f32 / 255.0_f32) as f64,
    (128.0_f32 / 255.0_f32) as f64,
    1.0,
];
const RGB_MEANS: [f64; 3] = [-1.0, 0.5, 1.0];
const RGB_SCALES: [f64; 3] = [2.0, 4.0, 8.0];
const GRAYSCALE_MEAN: [f64; 1] = [0.5];
const GRAYSCALE_SCALE: [f64; 1] = [4.0];
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

/// Proves fixed, normalized, cropped, and letterboxed preprocessing contracts.
#[test]
fn creates_checked_copied_image_blobs() {
    assert_eq!(ffi::live_dnn_blob_count(), 0);
    let image = ffi::mat_from_u8(&BGR_PIXELS, 2, 2, 3);
    let image_ref = image.as_ref().expect("BGR image");
    let blob = image_ref.blob_from_bgr_rgb_owned(2, 2);
    let blob_ref = blob.as_ref().expect("RGB blob");
    assert_eq!(ffi::live_dnn_blob_count(), 1);
    assert_eq!(blob_ref.batch_count(), 1);
    assert_eq!(blob_ref.channel_count(), 3);
    assert_eq!(blob_ref.height(), 2);
    assert_eq!(blob_ref.width(), 2);
    assert_values_close(
        &copied_values(blob_ref),
        &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    );
    assert_eq!(
        image_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source")
            .as_slice(),
        BGR_PIXELS
    );
    let normalized_blob =
        image_ref.blob_from_bgr_rgb_normalized_owned(2, 2, &RGB_MEANS, &RGB_SCALES);
    assert_values_close(
        &copied_values(normalized_blob.as_ref().expect("normalized RGB blob")),
        &EXPECTED_NORMALIZED_RGB,
    );

    let grayscale = ffi::mat_from_u8(&GRAY_PIXELS, 2, 2, 1);
    let grayscale_ref = grayscale.as_ref().expect("grayscale image");
    let grayscale_blob = grayscale_ref.blob_from_grayscale_owned(2, 2);
    let grayscale_blob_ref = grayscale_blob.as_ref().expect("grayscale blob");
    assert_eq!(ffi::live_dnn_blob_count(), 3);
    assert_eq!(grayscale_blob_ref.batch_count(), 1);
    assert_eq!(grayscale_blob_ref.channel_count(), 1);
    assert_eq!(grayscale_blob_ref.height(), 2);
    assert_eq!(grayscale_blob_ref.width(), 2);
    assert_values_close(&copied_values(grayscale_blob_ref), &EXPECTED_NCHW_GRAY);
    assert_eq!(
        grayscale_ref
            .to_u8()
            .as_ref()
            .expect("unchanged grayscale source")
            .as_slice(),
        GRAY_PIXELS
    );
    let normalized_grayscale_blob =
        grayscale_ref.blob_from_grayscale_normalized_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE);
    assert_values_close(
        &copied_values(
            normalized_grayscale_blob
                .as_ref()
                .expect("normalized grayscale blob"),
        ),
        &EXPECTED_NORMALIZED_GRAY,
    );
    assert_eq!(ffi::live_dnn_blob_count(), 4);
    drop(normalized_grayscale_blob);
    assert_eq!(ffi::live_dnn_blob_count(), 3);

    let wide_image = ffi::mat_from_u8(&WIDE_BGR_PIXELS, 2, 4, 3);
    let wide_image_ref = wide_image.as_ref().expect("wide BGR image");
    let cropped_blob = wide_image_ref.blob_from_bgr_rgb_center_crop_owned(2, 2);
    assert_values_close(
        &copied_values(cropped_blob.as_ref().expect("center-cropped RGB blob")),
        &EXPECTED_CENTER_CROP_RGB,
    );
    let letterbox_blob = wide_image_ref.blob_from_bgr_rgb_letterbox_owned(4, 4);
    let letterbox_ref = letterbox_blob.as_ref().expect("letterboxed RGB blob");
    assert_eq!(letterbox_ref.batch_count(), 1);
    assert_eq!(letterbox_ref.channel_count(), 3);
    assert_eq!(letterbox_ref.height(), 4);
    assert_eq!(letterbox_ref.width(), 4);
    assert_values_close(&copied_values(letterbox_ref), &EXPECTED_LETTERBOX_RGB);
    assert_eq!(ffi::live_dnn_blob_count(), 5);
    drop(letterbox_blob);
    assert_eq!(ffi::live_dnn_blob_count(), 4);
    let normalized_letterbox_blob =
        wide_image_ref.blob_from_bgr_rgb_normalized_letterbox_owned(4, 4, &RGB_MEANS, &RGB_SCALES);
    assert_values_close(
        &copied_values(
            normalized_letterbox_blob
                .as_ref()
                .expect("normalized letterboxed RGB blob"),
        ),
        &EXPECTED_NORMALIZED_LETTERBOX_RGB,
    );
    assert_eq!(ffi::live_dnn_blob_count(), 5);
    drop(normalized_letterbox_blob);
    assert_eq!(ffi::live_dnn_blob_count(), 4);
    let normalized_cropped_blob = wide_image_ref.blob_from_bgr_rgb_normalized_center_crop_owned(
        2,
        2,
        &RGB_MEANS,
        &RGB_SCALES,
    );
    assert_values_close(
        &copied_values(
            normalized_cropped_blob
                .as_ref()
                .expect("normalized center-cropped RGB blob"),
        ),
        &EXPECTED_NORMALIZED_CENTER_CROP_RGB,
    );
    let wide_grayscale = ffi::mat_from_u8(&WIDE_GRAY_PIXELS, 2, 4, 1);
    let wide_grayscale_ref = wide_grayscale.as_ref().expect("wide grayscale image");
    let cropped_grayscale_blob = wide_grayscale_ref.blob_from_grayscale_center_crop_owned(2, 2);
    assert_values_close(
        &copied_values(
            cropped_grayscale_blob
                .as_ref()
                .expect("center-cropped grayscale blob"),
        ),
        &EXPECTED_CENTER_CROP_GRAY,
    );
    let normalized_cropped_grayscale_blob = wide_grayscale_ref
        .blob_from_grayscale_normalized_center_crop_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE);
    assert_values_close(
        &copied_values(
            normalized_cropped_grayscale_blob
                .as_ref()
                .expect("normalized center-cropped grayscale blob"),
        ),
        &EXPECTED_NORMALIZED_CENTER_CROP_GRAY,
    );
    drop(normalized_cropped_grayscale_blob);
    let letterbox_grayscale_blob = wide_grayscale_ref.blob_from_grayscale_letterbox_owned(4, 4);
    let letterbox_grayscale_ref = letterbox_grayscale_blob
        .as_ref()
        .expect("letterboxed grayscale blob");
    assert_eq!(letterbox_grayscale_ref.batch_count(), 1);
    assert_eq!(letterbox_grayscale_ref.channel_count(), 1);
    assert_eq!(letterbox_grayscale_ref.height(), 4);
    assert_eq!(letterbox_grayscale_ref.width(), 4);
    assert_values_close(
        &copied_values(letterbox_grayscale_ref),
        &EXPECTED_LETTERBOX_GRAY,
    );
    let normalized_letterbox_grayscale_blob = wide_grayscale_ref
        .blob_from_grayscale_normalized_letterbox_owned(4, 4, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE);
    assert_values_close(
        &copied_values(
            normalized_letterbox_grayscale_blob
                .as_ref()
                .expect("normalized letterboxed grayscale blob"),
        ),
        &EXPECTED_NORMALIZED_LETTERBOX_GRAY,
    );
    drop(normalized_letterbox_grayscale_blob);
    assert_eq!(ffi::live_dnn_blob_count(), 7);
    drop(letterbox_grayscale_blob);
    assert_eq!(ffi::live_dnn_blob_count(), 6);
    assert_eq!(
        wide_image_ref
            .to_u8()
            .as_ref()
            .expect("unchanged wide BGR source")
            .as_slice(),
        WIDE_BGR_PIXELS
    );
    assert_eq!(
        wide_grayscale_ref
            .to_u8()
            .as_ref()
            .expect("unchanged wide grayscale source")
            .as_slice(),
        WIDE_GRAY_PIXELS
    );

    for (width, height) in [(0, 2), (2, 0), (-1, 2), (2, -1), (i64::MAX, 2)] {
        assert!(image_ref.blob_from_bgr_rgb_owned(width, height).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(image_ref
            .blob_from_bgr_rgb_normalized_owned(width, height, &RGB_MEANS, &RGB_SCALES,)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_image_ref
            .blob_from_bgr_rgb_normalized_center_crop_owned(width, height, &RGB_MEANS, &RGB_SCALES,)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_image_ref
            .blob_from_bgr_rgb_normalized_letterbox_owned(width, height, &RGB_MEANS, &RGB_SCALES,)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_image_ref
            .blob_from_bgr_rgb_center_crop_owned(width, height)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_image_ref
            .blob_from_bgr_rgb_letterbox_owned(width, height)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_grayscale_ref
            .blob_from_grayscale_center_crop_owned(width, height)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_grayscale_ref
            .blob_from_grayscale_letterbox_owned(width, height)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(grayscale_ref
            .blob_from_grayscale_owned(width, height)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(grayscale_ref
            .blob_from_grayscale_normalized_owned(width, height, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE,)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_grayscale_ref
            .blob_from_grayscale_normalized_center_crop_owned(
                width,
                height,
                &GRAYSCALE_MEAN,
                &GRAYSCALE_SCALE,
            )
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_grayscale_ref
            .blob_from_grayscale_normalized_letterbox_owned(
                width,
                height,
                &GRAYSCALE_MEAN,
                &GRAYSCALE_SCALE,
            )
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (means, scales) in [
        (&[][..], &RGB_SCALES[..]),
        (&RGB_MEANS[..2], &RGB_SCALES[..]),
        (&RGB_MEANS[..], &RGB_SCALES[..2]),
        (&[f64::NAN, 0.0, 0.0][..], &RGB_SCALES[..]),
        (&RGB_MEANS[..], &[1.0, 0.0, 1.0][..]),
        (&RGB_MEANS[..], &[1.0, -1.0, 1.0][..]),
    ] {
        assert!(image_ref
            .blob_from_bgr_rgb_normalized_owned(2, 2, means, scales)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_image_ref
            .blob_from_bgr_rgb_normalized_center_crop_owned(2, 2, means, scales)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_image_ref
            .blob_from_bgr_rgb_normalized_letterbox_owned(2, 2, means, scales)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (means, scales) in [
        (&[][..], &GRAYSCALE_SCALE[..]),
        (&[0.5, 0.0][..], &GRAYSCALE_SCALE[..]),
        (&GRAYSCALE_MEAN[..], &[][..]),
        (&GRAYSCALE_MEAN[..], &[4.0, 1.0][..]),
        (&[f64::NAN][..], &GRAYSCALE_SCALE[..]),
        (&GRAYSCALE_MEAN[..], &[0.0][..]),
        (&GRAYSCALE_MEAN[..], &[-1.0][..]),
    ] {
        assert!(grayscale_ref
            .blob_from_grayscale_normalized_owned(2, 2, means, scales)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_grayscale_ref
            .blob_from_grayscale_normalized_center_crop_owned(2, 2, means, scales)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(wide_grayscale_ref
            .blob_from_grayscale_normalized_letterbox_owned(2, 2, means, scales)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for layout in [1, 4] {
        let other = ffi::mat_zeros(2, 2, layout);
        assert!(other
            .as_ref()
            .expect("unsupported image layout")
            .blob_from_bgr_rgb_owned(2, 2)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(other
            .as_ref()
            .expect("unsupported normalized layout")
            .blob_from_bgr_rgb_normalized_owned(2, 2, &RGB_MEANS, &RGB_SCALES)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(other
            .as_ref()
            .expect("unsupported normalized letterbox layout")
            .blob_from_bgr_rgb_normalized_letterbox_owned(2, 2, &RGB_MEANS, &RGB_SCALES)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(other
            .as_ref()
            .expect("unsupported center-crop layout")
            .blob_from_bgr_rgb_center_crop_owned(2, 2)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(other
            .as_ref()
            .expect("unsupported letterbox layout")
            .blob_from_bgr_rgb_letterbox_owned(2, 2)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert!(image_ref.blob_from_grayscale_owned(2, 2).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(image_ref
        .blob_from_grayscale_center_crop_owned(2, 2)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(image_ref
        .blob_from_grayscale_letterbox_owned(2, 2)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(image_ref
        .blob_from_grayscale_normalized_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(image_ref
        .blob_from_grayscale_normalized_center_crop_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE,)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(image_ref
        .blob_from_grayscale_normalized_letterbox_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE,)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let bgra = ffi::mat_zeros(2, 2, 4);
    assert!(bgra
        .as_ref()
        .expect("BGRA image")
        .blob_from_grayscale_owned(2, 2)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgra
        .as_ref()
        .expect("BGRA image")
        .blob_from_grayscale_center_crop_owned(2, 2)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgra
        .as_ref()
        .expect("BGRA image")
        .blob_from_grayscale_letterbox_owned(2, 2)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgra
        .as_ref()
        .expect("BGRA image")
        .blob_from_grayscale_normalized_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgra
        .as_ref()
        .expect("BGRA image")
        .blob_from_grayscale_normalized_center_crop_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE,)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgra
        .as_ref()
        .expect("BGRA image")
        .blob_from_grayscale_normalized_letterbox_owned(2, 2, &GRAYSCALE_MEAN, &GRAYSCALE_SCALE,)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(normalized_cropped_blob);
    drop(normalized_blob);
    drop(cropped_grayscale_blob);
    drop(cropped_blob);
    drop(grayscale_blob);
    drop(blob);
    drop(wide_grayscale);
    drop(wide_image);
    assert_eq!(ffi::live_dnn_blob_count(), 0);
}

/// Copies one generated CXX vector into ordinary Rust storage.
fn copied_values(blob: &ffi::OpenCvDnnBlob) -> Vec<f64> {
    blob.values_owned()
        .as_ref()
        .expect("copied blob values")
        .iter()
        .copied()
        .collect()
}

/// Compares copied DNN values within one conservative `f32` rounding envelope.
fn assert_values_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "DNN value length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "DNN value {index} differs: actual={actual}, expected={expected}"
        );
    }
}
