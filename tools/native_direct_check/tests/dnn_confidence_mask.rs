//! Direct deterministic DNN class-confidence mask projection tests.

mod support;

use support::{FLATTEN_ONNX, GRAYSCALE_IDENTITY_ONNX, IDENTITY_ONNX};
use terlan_opencv_native::ffi;

const BGR_PROBABILITIES: [u8; 12] = [0, 0, 255, 0, 128, 0, 64, 0, 0, 32, 32, 32];
const ZERO_PIXELS: [u8; 12] = [0; 12];
const GRAYSCALE_PIXELS: [u8; 4] = [0, 64, 128, 255];

/// Proves deterministic probability and stable-softmax confidence projection.
#[test]
fn projects_highest_probabilities_to_bytes() {
    let image = ffi::mat_from_u8(&BGR_PROBABILITIES, 2, 2, 3);
    let blob = image
        .as_ref()
        .expect("BGR image")
        .blob_from_bgr_rgb_owned(2, 2);
    let model = ffi::load_onnx_model_owned(IDENTITY_ONNX);
    let output = model
        .as_ref()
        .expect("identity model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    let mask = ffi::class_confidence_mask_owned(output.as_ref().expect("identity output"));
    assert_mask(&mask, &[255, 128, 64, 32]);
    let repeated = ffi::class_confidence_mask_owned(output.as_ref().expect("identity output"));
    assert_mask(&repeated, &[255, 128, 64, 32]);
    let softmax_mask = output
        .as_ref()
        .expect("identity output")
        .softmax_class_confidence_mask_owned();
    assert_mask(&softmax_mask, &[147, 115, 100, 85]);
    let repeated_softmax = output
        .as_ref()
        .expect("identity output")
        .softmax_class_confidence_mask_owned();
    assert_mask(&repeated_softmax, &[147, 115, 100, 85]);
    let class_zero = output
        .as_ref()
        .expect("identity output")
        .softmax_class_probability_mask_owned(0);
    assert_mask(&class_zero, &[147, 70, 78, 85]);
    let class_two = output
        .as_ref()
        .expect("identity output")
        .softmax_class_probability_mask_owned(2);
    assert_mask(&class_two, &[54, 70, 100, 85]);
    let softmax_mask_zero = output
        .as_ref()
        .expect("identity output")
        .softmax_class_mask_owned(0, 0.5);
    assert_mask(&softmax_mask_zero, &[255, 0, 0, 0]);
    let softmax_mask_two = output
        .as_ref()
        .expect("identity output")
        .softmax_class_mask_owned(2, 0.3);
    assert_mask(&softmax_mask_two, &[0, 0, 255, 255]);
    let sigmoid_class_zero = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_probability_mask_owned(0);
    assert_mask(&sigmoid_class_zero, &[186, 128, 128, 135]);
    let sigmoid_class_two = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_probability_mask_owned(2);
    assert_mask(&sigmoid_class_two, &[128, 128, 143, 135]);
    let sigmoid_mask_zero = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_mask_owned(0, 0.55);
    assert_mask(&sigmoid_mask_zero, &[255, 0, 0, 0]);
    let sigmoid_mask_two = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_mask_owned(2, 0.55);
    assert_mask(&sigmoid_mask_two, &[0, 0, 255, 0]);
    let sigmoid_boundary = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_mask_owned(2, 0.5);
    assert_mask(&sigmoid_boundary, &[255, 255, 255, 255]);
    let sigmoid_any = output
        .as_ref()
        .expect("identity output")
        .sigmoid_any_class_mask_owned(0.55);
    assert_mask(&sigmoid_any, &[255, 255, 255, 0]);
    let sigmoid_any_boundary = output
        .as_ref()
        .expect("identity output")
        .sigmoid_any_class_mask_owned(0.5);
    assert_mask(&sigmoid_any_boundary, &[255, 255, 255, 255]);
    let sigmoid_confidence = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_confidence_mask_owned();
    assert_mask(&sigmoid_confidence, &[186, 159, 143, 135]);
    let repeated_sigmoid_confidence = output
        .as_ref()
        .expect("identity output")
        .sigmoid_class_confidence_mask_owned();
    assert_mask(&repeated_sigmoid_confidence, &[186, 159, 143, 135]);
    for class_index in [-1, 3] {
        assert!(output
            .as_ref()
            .expect("identity output")
            .softmax_class_probability_mask_owned(class_index)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(output
            .as_ref()
            .expect("identity output")
            .softmax_class_mask_owned(class_index, 0.5)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(output
            .as_ref()
            .expect("identity output")
            .sigmoid_class_probability_mask_owned(class_index)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(output
            .as_ref()
            .expect("identity output")
            .sigmoid_class_mask_owned(class_index, 0.5)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for threshold in [f64::NEG_INFINITY, -0.01, 1.01, f64::INFINITY, f64::NAN] {
        assert!(output
            .as_ref()
            .expect("identity output")
            .softmax_class_mask_owned(0, threshold)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(output
            .as_ref()
            .expect("identity output")
            .sigmoid_class_mask_owned(0, threshold)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(output
            .as_ref()
            .expect("identity output")
            .sigmoid_any_class_mask_owned(threshold)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    let zero_image = ffi::mat_from_u8(&ZERO_PIXELS, 2, 2, 3);
    let zero_blob = zero_image
        .as_ref()
        .expect("zero image")
        .blob_from_bgr_rgb_owned(2, 2);
    let zero_output = model
        .as_ref()
        .expect("identity model")
        .forward_owned(zero_blob.as_ref().expect("zero blob"));
    let zero_mask = ffi::class_confidence_mask_owned(zero_output.as_ref().expect("zero output"));
    assert_mask(&zero_mask, &[0, 0, 0, 0]);
    let zero_softmax = zero_output
        .as_ref()
        .expect("zero output")
        .softmax_class_confidence_mask_owned();
    assert_mask(&zero_softmax, &[85, 85, 85, 85]);
    let zero_class = zero_output
        .as_ref()
        .expect("zero output")
        .softmax_class_probability_mask_owned(1);
    assert_mask(&zero_class, &[85, 85, 85, 85]);
    let zero_softmax_mask = zero_output
        .as_ref()
        .expect("zero output")
        .softmax_class_mask_owned(1, 1.0 / 3.0);
    assert_mask(&zero_softmax_mask, &[255, 255, 255, 255]);
    let zero_sigmoid_class = zero_output
        .as_ref()
        .expect("zero output")
        .sigmoid_class_probability_mask_owned(1);
    assert_mask(&zero_sigmoid_class, &[128, 128, 128, 128]);
    let zero_sigmoid_any = zero_output
        .as_ref()
        .expect("zero output")
        .sigmoid_any_class_mask_owned(0.5);
    assert_mask(&zero_sigmoid_any, &[255, 255, 255, 255]);
    let zero_sigmoid_confidence = zero_output
        .as_ref()
        .expect("zero output")
        .sigmoid_class_confidence_mask_owned();
    assert_mask(&zero_sigmoid_confidence, &[128, 128, 128, 128]);

    let grayscale_image = ffi::mat_from_u8(&GRAYSCALE_PIXELS, 2, 2, 1);
    let grayscale_blob = grayscale_image
        .as_ref()
        .expect("grayscale image")
        .blob_from_grayscale_owned(2, 2);
    let grayscale_model = ffi::load_onnx_model_owned(GRAYSCALE_IDENTITY_ONNX);
    let grayscale_output = grayscale_model
        .as_ref()
        .expect("grayscale model")
        .forward_owned(grayscale_blob.as_ref().expect("grayscale blob"));
    let binary_sigmoid = grayscale_output
        .as_ref()
        .expect("single-channel output")
        .sigmoid_class_probability_mask_owned(0);
    assert_mask(&binary_sigmoid, &[128, 143, 159, 186]);
    let binary_mask = grayscale_output
        .as_ref()
        .expect("single-channel output")
        .sigmoid_class_mask_owned(0, 0.6);
    assert_mask(&binary_mask, &[0, 0, 255, 255]);
    let binary_any = grayscale_output
        .as_ref()
        .expect("single-channel output")
        .sigmoid_any_class_mask_owned(0.6);
    assert_mask(&binary_any, &[0, 0, 255, 255]);
    let binary_confidence = grayscale_output
        .as_ref()
        .expect("single-channel output")
        .sigmoid_class_confidence_mask_owned();
    assert_mask(&binary_confidence, &[128, 143, 159, 186]);
    assert!(grayscale_output
        .as_ref()
        .expect("single-channel output")
        .softmax_class_mask_owned(0, 0.5)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let flatten_model = ffi::load_onnx_model_owned(FLATTEN_ONNX);
    let flatten_output = flatten_model
        .as_ref()
        .expect("flatten model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    assert!(
        ffi::class_confidence_mask_owned(flatten_output.as_ref().expect("rank-two output"))
            .is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .softmax_class_mask_owned(0, 0.5)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .sigmoid_class_confidence_mask_owned()
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .sigmoid_any_class_mask_owned(0.5)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .softmax_class_confidence_mask_owned()
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .softmax_class_probability_mask_owned(0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .sigmoid_class_probability_mask_owned(0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(flatten_output
        .as_ref()
        .expect("rank-two output")
        .sigmoid_class_mask_owned(0, 0.5)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(flatten_output);
    drop(flatten_model);
    drop(binary_mask);
    drop(binary_any);
    drop(binary_confidence);
    drop(binary_sigmoid);
    drop(grayscale_output);
    drop(grayscale_model);
    drop(grayscale_blob);
    drop(grayscale_image);
    drop(zero_sigmoid_class);
    drop(zero_sigmoid_any);
    drop(zero_sigmoid_confidence);
    drop(zero_softmax_mask);
    drop(zero_class);
    drop(zero_softmax);
    drop(zero_mask);
    drop(zero_output);
    drop(zero_blob);
    drop(zero_image);
    drop(class_two);
    drop(class_zero);
    drop(softmax_mask_two);
    drop(softmax_mask_zero);
    drop(sigmoid_class_two);
    drop(sigmoid_class_zero);
    drop(sigmoid_boundary);
    drop(sigmoid_any_boundary);
    drop(sigmoid_any);
    drop(repeated_sigmoid_confidence);
    drop(sigmoid_confidence);
    drop(sigmoid_mask_two);
    drop(sigmoid_mask_zero);
    drop(repeated_softmax);
    drop(softmax_mask);
    drop(repeated);
    drop(mask);
    drop(output);
    drop(model);
    drop(blob);
    drop(image);
}

/// Requires one owned two-by-two grayscale mask with exact copied bytes.
fn assert_mask(mask: &cxx::UniquePtr<ffi::OpenCvMat>, expected: &[u8]) {
    let mask = mask.as_ref().expect("class-confidence mask");
    assert_eq!(mask.rows(), 2);
    assert_eq!(mask.cols(), 2);
    assert_eq!(mask.channels(), 1);
    assert_eq!(
        mask.to_u8().as_ref().expect("copied mask bytes").as_slice(),
        expected
    );
}
