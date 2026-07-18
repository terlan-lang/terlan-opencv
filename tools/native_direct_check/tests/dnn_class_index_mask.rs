//! Direct deterministic DNN class-index mask projection tests.

mod support;

use support::{FLATTEN_ONNX, IDENTITY_ONNX};
use terlan_opencv_native::ffi;

const BGR_PIXELS: [u8; 12] = [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255];
const ZERO_PIXELS: [u8; 12] = [0; 12];

/// Proves deterministic per-pixel argmax, tie selection, rank rejection, and ownership.
#[test]
fn projects_output_classes_to_index_bytes() {
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
    let mask = ffi::class_index_mask_owned(output.as_ref().expect("identity output"));
    assert_mask(&mask, &[0, 1, 2, 0]);
    let repeated = ffi::class_index_mask_owned(output.as_ref().expect("identity output"));
    assert_mask(&repeated, &[0, 1, 2, 0]);

    let zero_image = ffi::mat_from_u8(&ZERO_PIXELS, 2, 2, 3);
    let zero_blob = zero_image
        .as_ref()
        .expect("zero image")
        .blob_from_bgr_rgb_owned(2, 2);
    let zero_output = model
        .as_ref()
        .expect("identity model")
        .forward_owned(zero_blob.as_ref().expect("zero blob"));
    let tied = ffi::class_index_mask_owned(zero_output.as_ref().expect("zero output"));
    assert_mask(&tied, &[0, 0, 0, 0]);

    let flatten_model = ffi::load_onnx_model_owned(FLATTEN_ONNX);
    let flatten_output = flatten_model
        .as_ref()
        .expect("flatten model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    assert!(
        ffi::class_index_mask_owned(flatten_output.as_ref().expect("rank-two output")).is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);

    drop(flatten_output);
    drop(flatten_model);
    drop(tied);
    drop(zero_output);
    drop(zero_blob);
    drop(zero_image);
    drop(repeated);
    drop(mask);
    drop(output);
    drop(model);
    drop(blob);
    drop(image);
}

/// Requires one owned two-by-two grayscale mask with exact copied bytes.
fn assert_mask(mask: &cxx::UniquePtr<ffi::OpenCvMat>, expected: &[u8]) {
    let mask = mask.as_ref().expect("class-index mask");
    assert_eq!(mask.rows(), 2);
    assert_eq!(mask.cols(), 2);
    assert_eq!(mask.channels(), 1);
    assert_eq!(
        mask.to_u8().as_ref().expect("copied mask bytes").as_slice(),
        expected
    );
}
