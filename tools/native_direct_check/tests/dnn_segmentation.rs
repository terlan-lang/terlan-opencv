//! Direct deterministic DNN channel-mask projection tests.

mod support;

use support::{FLATTEN_ONNX, IDENTITY_ONNX};
use terlan_opencv_native::ffi;

const BGR_PIXELS: [u8; 12] = [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255];

/// Proves channel selection, strict thresholding, shape rejection, and ownership.
#[test]
fn projects_one_output_channel_to_a_mask() {
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

    let green_mask = ffi::channel_mask_owned(output_ref, 1, 0.5);
    assert_mask(&green_mask, &[0, 255, 0, 255]);
    let red_mask = ffi::channel_mask_owned(output_ref, 0, 0.5);
    assert_mask(&red_mask, &[255, 0, 0, 255]);
    let strict_mask = ffi::channel_mask_owned(output_ref, 1, 1.0);
    assert_mask(&strict_mask, &[0, 0, 0, 0]);

    assert!(ffi::channel_mask_owned(output_ref, -1, 0.5).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(ffi::channel_mask_owned(output_ref, 3, 0.5).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(ffi::channel_mask_owned(output_ref, 1, f64::INFINITY).is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let flatten_model = ffi::load_onnx_model_owned(FLATTEN_ONNX);
    let flatten_output = flatten_model
        .as_ref()
        .expect("flatten model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    assert!(
        ffi::channel_mask_owned(flatten_output.as_ref().expect("rank-two output"), 0, 0.5)
            .is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);

    drop(flatten_output);
    drop(flatten_model);
    drop(strict_mask);
    drop(red_mask);
    drop(green_mask);
    drop(output);
    drop(model);
    drop(blob);
    drop(image);
}

/// Requires one owned two-by-two grayscale mask with exact copied bytes.
fn assert_mask(mask: &cxx::UniquePtr<ffi::OpenCvMat>, expected: &[u8]) {
    let mask = mask.as_ref().expect("channel mask");
    assert_eq!(mask.rows(), 2);
    assert_eq!(mask.cols(), 2);
    assert_eq!(mask.channels(), 1);
    assert_eq!(
        mask.to_u8().as_ref().expect("copied mask bytes").as_slice(),
        expected
    );
}
