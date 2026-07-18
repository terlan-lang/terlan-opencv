//! Direct CPU ONNX inference tests for the generated CXX bridge.

mod support;

use support::{GRAYSCALE_IDENTITY_ONNX, IDENTITY_ONNX};
use terlan_opencv_native::ffi;

const BGR_PIXELS: [u8; 12] = [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255];
const EXPECTED_VALUES: [f64; 12] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const GRAYSCALE_PIXELS: [u8; 4] = [0, 64, 128, 255];
const EXPECTED_GRAYSCALE_VALUES: [f64; 4] = [
    0.0,
    (64.0_f32 / 255.0_f32) as f64,
    (128.0_f32 / 255.0_f32) as f64,
    1.0,
];

/// Proves copied in-memory loading, fixed CPU inference, and independent owners.
#[test]
fn runs_identity_onnx_from_owned_bytes() {
    assert_eq!(ffi::live_onnx_model_count(), 0);
    assert_eq!(ffi::live_dnn_output_count(), 0);
    assert!(ffi::load_onnx_model_owned(&[]).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(ffi::load_onnx_model_owned(&[1, 2, 3]).is_null());
    assert_eq!(ffi::take_last_failure(), 3);
    let image = ffi::mat_from_u8(&BGR_PIXELS, 2, 2, 3);
    let blob = image
        .as_ref()
        .expect("BGR image")
        .blob_from_bgr_rgb_owned(2, 2);
    let model = ffi::load_onnx_model_owned(IDENTITY_ONNX);
    let output = model
        .as_ref()
        .expect("identity model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    let output_ref = output.as_ref().expect("identity output");

    assert_eq!(ffi::live_onnx_model_count(), 1);
    assert_eq!(ffi::live_dnn_output_count(), 1);
    assert_eq!(output_ref.rank(), 4);
    assert_eq!(copied_shape(output_ref), [1, 3, 2, 2]);
    assert_eq!(copied_values(output_ref), EXPECTED_VALUES);
    assert_eq!(output_ref.value_count(), 12);
    assert_eq!(copied_values_slice(output_ref, 0, 3), [1.0, 0.0, 0.0]);
    assert_eq!(copied_values_slice(output_ref, 3, 4), [1.0, 0.0, 1.0, 0.0]);
    assert_eq!(copied_values_slice(output_ref, 10, 99), [1.0, 1.0]);
    assert!(copied_values_slice(output_ref, 12, 1).is_empty());
    assert_eq!(copied_values_slice(output_ref, 3, 4), [1.0, 0.0, 1.0, 0.0]);
    assert!(output_ref.values_slice_owned(-1, 1).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref.values_slice_owned(0, 0).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref.values_slice_owned(0, 1_048_577).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_ref.values_slice_owned(13, 1).is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let repeated = model
        .as_ref()
        .expect("identity model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    assert_eq!(
        copied_values(repeated.as_ref().expect("repeated output")),
        EXPECTED_VALUES
    );
    drop(repeated);

    let grayscale_image = ffi::mat_from_u8(&GRAYSCALE_PIXELS, 2, 2, 1);
    let grayscale_blob = grayscale_image
        .as_ref()
        .expect("grayscale image")
        .blob_from_grayscale_owned(2, 2);
    let grayscale_model = ffi::load_onnx_model_owned(GRAYSCALE_IDENTITY_ONNX);
    let grayscale_output = grayscale_model
        .as_ref()
        .expect("grayscale identity model")
        .forward_owned(grayscale_blob.as_ref().expect("grayscale blob"));
    let grayscale_output_ref = grayscale_output
        .as_ref()
        .expect("grayscale identity output");
    assert_eq!(ffi::live_onnx_model_count(), 2);
    assert_eq!(ffi::live_dnn_output_count(), 2);
    assert_eq!(grayscale_output_ref.rank(), 4);
    assert_eq!(grayscale_output_ref.value_count(), 4);
    assert_eq!(copied_shape(grayscale_output_ref), [1, 1, 2, 2]);
    assert_eq!(
        copied_values(grayscale_output_ref),
        EXPECTED_GRAYSCALE_VALUES
    );
    assert_eq!(
        copied_values_slice(grayscale_output_ref, 1, 2),
        EXPECTED_GRAYSCALE_VALUES[1..3]
    );
    let repeated_grayscale = grayscale_model
        .as_ref()
        .expect("grayscale identity model")
        .forward_owned(grayscale_blob.as_ref().expect("grayscale blob"));
    assert_eq!(
        copied_values(
            repeated_grayscale
                .as_ref()
                .expect("repeated grayscale output")
        ),
        EXPECTED_GRAYSCALE_VALUES
    );
    drop(repeated_grayscale);
    drop(grayscale_model);
    assert_eq!(ffi::live_onnx_model_count(), 1);
    assert_eq!(
        copied_values(grayscale_output_ref),
        EXPECTED_GRAYSCALE_VALUES
    );
    drop(grayscale_output);
    drop(grayscale_blob);
    drop(grayscale_image);

    drop(model);
    assert_eq!(ffi::live_onnx_model_count(), 0);
    assert_eq!(copied_values(output_ref), EXPECTED_VALUES);
    drop(output);
    assert_eq!(ffi::live_dnn_output_count(), 0);
    drop(blob);
    drop(image);
    assert_eq!(ffi::live_dnn_blob_count(), 0);
    assert_eq!(ffi::live_onnx_model_count(), 0);
}

/// Copies one generated CXX shape vector into ordinary Rust storage.
fn copied_shape(output: &ffi::OpenCvDnnOutput) -> Vec<i64> {
    output
        .shape_owned()
        .as_ref()
        .expect("copied output shape")
        .iter()
        .copied()
        .collect()
}

/// Copies one generated CXX value vector into ordinary Rust storage.
fn copied_values(output: &ffi::OpenCvDnnOutput) -> Vec<f64> {
    output
        .values_owned()
        .as_ref()
        .expect("copied output values")
        .iter()
        .copied()
        .collect()
}

/// Copies one generated CXX output slice into ordinary Rust storage.
fn copied_values_slice(output: &ffi::OpenCvDnnOutput, offset: i64, maximum: i64) -> Vec<f64> {
    output
        .values_slice_owned(offset, maximum)
        .as_ref()
        .expect("copied output value slice")
        .iter()
        .copied()
        .collect()
}
