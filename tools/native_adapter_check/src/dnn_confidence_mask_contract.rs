//! Deterministic DNN class-confidence mask helper contract.

use super::dnn_classification_contract::FLATTEN_ONNX;
use super::dnn_inference_contract::{GRAYSCALE_IDENTITY_ONNX, IDENTITY_ONNX};
use super::{
    bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

const BGR_PROBABILITIES: [u8; 12] = [0, 0, 255, 0, 128, 0, 64, 0, 0, 32, 32, 32];
const GRAYSCALE_PIXELS: [u8; 4] = [0, 64, 128, 255];

/// Runs probability and stable-softmax confidence-mask boundary checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let image = matrix(helper, &BGR_PROBABILITIES, 2, 2, "bgr")?;
    let blob = create_blob(helper, &image)?;
    let model = load_model(helper, IDENTITY_ONNX)?;
    let output = forward(helper, &model, &blob)?;
    let mask = confidence_mask(helper, &output)?;
    assert_mask(helper, &mask, &[255, 128, 64, 32])?;
    let repeated = confidence_mask(helper, &output)?;
    assert_mask(helper, &repeated, &[255, 128, 64, 32])?;
    let softmax = softmax_confidence_mask(helper, &output)?;
    assert_mask(helper, &softmax, &[147, 115, 100, 85])?;
    let repeated_softmax = softmax_confidence_mask(helper, &output)?;
    assert_mask(helper, &repeated_softmax, &[147, 115, 100, 85])?;
    let class_zero = softmax_probability_mask(helper, &output, 0)?;
    assert_mask(helper, &class_zero, &[147, 70, 78, 85])?;
    let class_two = softmax_probability_mask(helper, &output, 2)?;
    assert_mask(helper, &class_two, &[54, 70, 100, 85])?;
    let softmax_mask_zero = softmax_mask(helper, &output, 0, 0.5)?;
    assert_mask(helper, &softmax_mask_zero, &[255, 0, 0, 0])?;
    let softmax_mask_two = softmax_mask(helper, &output, 2, 0.3)?;
    assert_mask(helper, &softmax_mask_two, &[0, 0, 255, 255])?;
    let sigmoid_class_zero = sigmoid_probability_mask(helper, &output, 0)?;
    assert_mask(helper, &sigmoid_class_zero, &[186, 128, 128, 135])?;
    let sigmoid_class_two = sigmoid_probability_mask(helper, &output, 2)?;
    assert_mask(helper, &sigmoid_class_two, &[128, 128, 143, 135])?;
    let sigmoid_mask_zero = sigmoid_mask(helper, &output, 0, 0.55)?;
    assert_mask(helper, &sigmoid_mask_zero, &[255, 0, 0, 0])?;
    let sigmoid_mask_two = sigmoid_mask(helper, &output, 2, 0.55)?;
    assert_mask(helper, &sigmoid_mask_two, &[0, 0, 255, 0])?;
    let sigmoid_boundary = sigmoid_mask(helper, &output, 2, 0.5)?;
    assert_mask(helper, &sigmoid_boundary, &[255, 255, 255, 255])?;
    let sigmoid_any = sigmoid_any_mask(helper, &output, 0.55)?;
    assert_mask(helper, &sigmoid_any, &[255, 255, 255, 0])?;
    let sigmoid_any_boundary = sigmoid_any_mask(helper, &output, 0.5)?;
    assert_mask(helper, &sigmoid_any_boundary, &[255, 255, 255, 255])?;
    let sigmoid_confidence = sigmoid_confidence_mask(helper, &output)?;
    assert_mask(helper, &sigmoid_confidence, &[186, 159, 143, 135])?;
    let repeated_sigmoid_confidence = sigmoid_confidence_mask(helper, &output)?;
    assert_mask(helper, &repeated_sigmoid_confidence, &[186, 159, 143, 135])?;
    for class_index in [-1, 3] {
        expect_error_code(
            helper.call(
                "opencv.dnn.softmax_class_probability_mask",
                &[output.arg(), int_arg(class_index)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.softmax_class_mask",
                &[output.arg(), int_arg(class_index), float_arg(0.5)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_class_probability_mask",
                &[output.arg(), int_arg(class_index)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_class_mask",
                &[output.arg(), int_arg(class_index), float_arg(0.5)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for threshold in [f64::NEG_INFINITY, -0.01, 1.01, f64::INFINITY, f64::NAN] {
        expect_error_code(
            helper.call(
                "opencv.dnn.softmax_class_mask",
                &[output.arg(), int_arg(0), float_arg(threshold)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_class_mask",
                &[output.arg(), int_arg(0), float_arg(threshold)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_any_class_mask",
                &[output.arg(), float_arg(threshold)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let zero_image = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    let zero_blob = create_blob(helper, &zero_image)?;
    let zero_output = forward(helper, &model, &zero_blob)?;
    let zero_mask = confidence_mask(helper, &zero_output)?;
    assert_mask(helper, &zero_mask, &[0, 0, 0, 0])?;
    let zero_softmax = softmax_confidence_mask(helper, &zero_output)?;
    assert_mask(helper, &zero_softmax, &[85, 85, 85, 85])?;
    let zero_class = softmax_probability_mask(helper, &zero_output, 1)?;
    assert_mask(helper, &zero_class, &[85, 85, 85, 85])?;
    let zero_softmax_mask = softmax_mask(helper, &zero_output, 1, 1.0 / 3.0)?;
    assert_mask(helper, &zero_softmax_mask, &[255, 255, 255, 255])?;
    let zero_sigmoid_class = sigmoid_probability_mask(helper, &zero_output, 1)?;
    assert_mask(helper, &zero_sigmoid_class, &[128, 128, 128, 128])?;
    let zero_sigmoid_any = sigmoid_any_mask(helper, &zero_output, 0.5)?;
    assert_mask(helper, &zero_sigmoid_any, &[255, 255, 255, 255])?;
    let zero_sigmoid_confidence = sigmoid_confidence_mask(helper, &zero_output)?;
    assert_mask(helper, &zero_sigmoid_confidence, &[128, 128, 128, 128])?;

    let grayscale_image = matrix(helper, &GRAYSCALE_PIXELS, 2, 2, "gray")?;
    let grayscale_blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_grayscale",
            &[grayscale_image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let grayscale_model = load_model(helper, GRAYSCALE_IDENTITY_ONNX)?;
    let grayscale_output = forward(helper, &grayscale_model, &grayscale_blob)?;
    let binary_sigmoid = sigmoid_probability_mask(helper, &grayscale_output, 0)?;
    assert_mask(helper, &binary_sigmoid, &[128, 143, 159, 186])?;
    let binary_mask = sigmoid_mask(helper, &grayscale_output, 0, 0.6)?;
    assert_mask(helper, &binary_mask, &[0, 0, 255, 255])?;
    let binary_any = sigmoid_any_mask(helper, &grayscale_output, 0.6)?;
    assert_mask(helper, &binary_any, &[0, 0, 255, 255])?;
    let binary_confidence = sigmoid_confidence_mask(helper, &grayscale_output)?;
    assert_mask(helper, &binary_confidence, &[128, 143, 159, 186])?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_mask",
            &[grayscale_output.arg(), int_arg(0), float_arg(0.5)],
        )?,
        "opencv.invalid_argument",
    )?;
    let flatten_model = load_model(helper, &FLATTEN_ONNX)?;
    let flatten_output = forward(helper, &flatten_model, &blob)?;
    expect_error_code(
        helper.call("opencv.dnn.class_confidence_mask", &[flatten_output.arg()])?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_confidence_mask",
            &[flatten_output.arg()],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_any_class_mask",
            &[flatten_output.arg(), float_arg(0.5)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_mask",
            &[flatten_output.arg(), int_arg(0), float_arg(0.5)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_mask",
            &[flatten_output.arg(), int_arg(0), float_arg(0.5)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_confidence_mask",
            &[flatten_output.arg()],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_probability_mask",
            &[flatten_output.arg(), int_arg(0)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_probability_mask",
            &[flatten_output.arg(), int_arg(0)],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.class_confidence_mask",
            &[output.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.class_confidence_mask", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.class_confidence_mask", &["i:1".to_owned()])?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_confidence_mask",
            &[output.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.sigmoid_class_confidence_mask", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_confidence_mask",
            &["i:1".to_owned()],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_any_class_mask",
            &[output.arg_with_owner("foreign-worker"), float_arg(0.5)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_any_class_mask",
            &[model.arg(), float_arg(0.5)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_any_class_mask",
            &[output.arg(), int_arg(1)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_mask",
            &[
                output.arg_with_owner("foreign-worker"),
                int_arg(0),
                float_arg(0.5),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_mask",
            &[model.arg(), int_arg(0), float_arg(0.5)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_mask",
            &[output.arg(), "f:0.0".to_owned(), float_arg(0.5)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_mask",
            &[
                output.arg_with_owner("foreign-worker"),
                int_arg(0),
                float_arg(0.5),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_mask",
            &[model.arg(), int_arg(0), float_arg(0.5)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_mask",
            &[output.arg(), "f:0.0".to_owned(), float_arg(0.5)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_confidence_mask",
            &[output.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.softmax_class_confidence_mask", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_confidence_mask",
            &["i:1".to_owned()],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_probability_mask",
            &[output.arg_with_owner("foreign-worker"), int_arg(0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_probability_mask",
            &[model.arg(), int_arg(0)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_class_probability_mask",
            &[output.arg(), "f:0.0".to_owned()],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_probability_mask",
            &[output.arg_with_owner("foreign-worker"), int_arg(0)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_probability_mask",
            &[model.arg(), int_arg(0)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_class_probability_mask",
            &[output.arg(), "f:0.0".to_owned()],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.mat.dispose", &[mask.arg()])?)?;
    expect_error_code(
        helper.call("opencv.mat.to_u8", &[mask.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_mask(helper, &softmax, &[147, 115, 100, 85])?;
    assert_mask(helper, &class_zero, &[147, 70, 78, 85])?;
    assert_mask(helper, &sigmoid_class_zero, &[186, 128, 128, 135])?;
    for owner in [
        &repeated,
        &softmax,
        &repeated_softmax,
        &class_zero,
        &class_two,
        &softmax_mask_zero,
        &softmax_mask_two,
        &sigmoid_class_zero,
        &sigmoid_class_two,
        &sigmoid_mask_zero,
        &sigmoid_mask_two,
        &sigmoid_boundary,
        &sigmoid_any,
        &sigmoid_any_boundary,
        &sigmoid_confidence,
        &repeated_sigmoid_confidence,
        &zero_mask,
        &zero_softmax,
        &zero_class,
        &zero_softmax_mask,
        &zero_sigmoid_class,
        &zero_sigmoid_any,
        &zero_sigmoid_confidence,
        &binary_sigmoid,
        &binary_mask,
        &binary_any,
        &binary_confidence,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[owner.arg()])?)?;
    }
    for owner in [&flatten_output, &zero_output, &grayscale_output] {
        expect_unit(helper.call("opencv.dnn.dispose_output", &[owner.arg()])?)?;
    }
    for owner in [&flatten_model, &model, &grayscale_model] {
        expect_unit(helper.call("opencv.dnn.dispose_model", &[owner.arg()])?)?;
    }
    for owner in [&zero_blob, &blob, &grayscale_blob] {
        expect_unit(helper.call("opencv.dnn.dispose", &[owner.arg()])?)?;
    }
    for owner in [&zero_image, &image, &grayscale_image] {
        expect_unit(helper.call("opencv.mat.dispose", &[owner.arg()])?)?;
    }
    Ok(())
}

/// Creates one fixed two-by-two RGB inference blob.
fn create_blob(helper: &mut NativeHelper, image: &NativeHandle) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.Dnn.Blob",
    )
}

/// Loads one copied in-memory ONNX model.
fn load_model(helper: &mut NativeHelper, bytes: &[u8]) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(bytes)])?,
        "opencv.Dnn.OnnxModel",
    )
}

/// Runs one model against one immutable blob.
fn forward(
    helper: &mut NativeHelper,
    model: &NativeHandle,
    blob: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )
}

/// Projects one output into a typed confidence-mask owner.
fn confidence_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.class_confidence_mask", &[output.arg()])?,
        "opencv.Mat.Mat",
    )
}

/// Projects one output into a typed stable-softmax confidence-mask owner.
fn softmax_confidence_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.softmax_class_confidence_mask", &[output.arg()])?,
        "opencv.Mat.Mat",
    )
}

/// Projects one selected class into a typed stable-softmax probability mask.
fn softmax_probability_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    class_index: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.softmax_class_probability_mask",
            &[output.arg(), int_arg(class_index)],
        )?,
        "opencv.Mat.Mat",
    )
}

/// Thresholds one selected class into a typed stable-softmax binary mask.
fn softmax_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    class_index: i64,
    minimum_probability: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.softmax_class_mask",
            &[
                output.arg(),
                int_arg(class_index),
                float_arg(minimum_probability),
            ],
        )?,
        "opencv.Mat.Mat",
    )
}

/// Projects one selected class into a typed stable-sigmoid probability mask.
fn sigmoid_probability_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    class_index: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.sigmoid_class_probability_mask",
            &[output.arg(), int_arg(class_index)],
        )?,
        "opencv.Mat.Mat",
    )
}

/// Thresholds one selected class into a typed stable-sigmoid binary mask.
fn sigmoid_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    class_index: i64,
    minimum_probability: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.sigmoid_class_mask",
            &[
                output.arg(),
                int_arg(class_index),
                float_arg(minimum_probability),
            ],
        )?,
        "opencv.Mat.Mat",
    )
}

/// Projects whether any class meets a stable-sigmoid probability threshold.
fn sigmoid_any_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    minimum_probability: f64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.sigmoid_any_class_mask",
            &[output.arg(), float_arg(minimum_probability)],
        )?,
        "opencv.Mat.Mat",
    )
}

/// Projects the maximum independently normalized sigmoid probability.
fn sigmoid_confidence_mask(
    helper: &mut NativeHelper,
    output: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.sigmoid_class_confidence_mask", &[output.arg()])?,
        "opencv.Mat.Mat",
    )
}

/// Requires exact two-by-two grayscale dimensions and copied confidence bytes.
fn assert_mask(
    helper: &mut NativeHelper,
    mask: &NativeHandle,
    expected: &[u8],
) -> Result<(), String> {
    expect_int(helper.call("opencv.mat.rows", &[mask.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[mask.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.channels", &[mask.arg()])?, 1)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mask.arg()])?, expected)
}
