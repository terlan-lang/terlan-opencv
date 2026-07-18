//! Deterministic DNN class-index mask helper contract.

use super::dnn_classification_contract::FLATTEN_ONNX;
use super::dnn_inference_contract::{BGR_PIXELS, IDENTITY_ONNX};
use super::{
    bytes_arg, expect_bytes, expect_error_code, expect_int, expect_typed_handle, expect_unit,
    int_arg, matrix, NativeHandle, NativeHelper,
};

/// Runs exact bytes, tie selection, owner, type, malformed-input, and stale-handle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    let image = matrix(helper, &BGR_PIXELS, 2, 2, "bgr")?;
    let blob = create_blob(helper, &image)?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(IDENTITY_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let mask = class_mask(helper, &output)?;
    assert_mask(helper, &mask, &[0, 1, 2, 0])?;
    let repeated = class_mask(helper, &output)?;
    assert_mask(helper, &repeated, &[0, 1, 2, 0])?;

    let zero_image = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    let zero_blob = create_blob(helper, &zero_image)?;
    let zero_output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), zero_blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let tied = class_mask(helper, &zero_output)?;
    assert_mask(helper, &tied, &[0, 0, 0, 0])?;

    let flatten_model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(&FLATTEN_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let flatten_output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[flatten_model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.class_index_mask", &[flatten_output.arg()])?,
        "opencv.invalid_argument",
    )?;

    expect_error_code(
        helper.call(
            "opencv.dnn.class_index_mask",
            &[output.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.class_index_mask", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.class_index_mask", &["i:1".to_owned()])?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.mat.dispose", &[mask.arg()])?)?;
    expect_error_code(
        helper.call("opencv.mat.to_u8", &[mask.arg()])?,
        "stale_handle",
    )?;
    for owner in [&repeated, &tied] {
        expect_unit(helper.call("opencv.mat.dispose", &[owner.arg()])?)?;
    }
    for owner in [&zero_output, &output] {
        expect_unit(helper.call("opencv.dnn.dispose_output", &[owner.arg()])?)?;
    }
    expect_unit(helper.call("opencv.dnn.dispose_output", &[flatten_output.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[flatten_model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    for owner in [&zero_blob, &blob] {
        expect_unit(helper.call("opencv.dnn.dispose", &[owner.arg()])?)?;
    }
    for owner in [&zero_image, &image] {
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

/// Projects one output into a typed class-index matrix owner.
fn class_mask(helper: &mut NativeHelper, output: &NativeHandle) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.class_index_mask", &[output.arg()])?,
        "opencv.Mat.Mat",
    )
}

/// Requires exact two-by-two grayscale dimensions and copied class indexes.
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
