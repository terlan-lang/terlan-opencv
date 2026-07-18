//! Shared generated-helper contract for field-major detection projections.

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

/// Describes one inference-backed field-major detection operation.
pub(super) struct FieldMajorDetectionContract<'a> {
    /// Generated native operation name.
    pub(super) operation: &'a str,
    /// Serialized ONNX reshape fixture.
    pub(super) model: &'a [u8],
    /// BGR image bytes whose NCHW storage becomes field-major output.
    pub(super) pixels: &'a [u8],
    /// Input and output field count.
    pub(super) fields: i64,
    /// Inclusive threshold retaining every fixture row.
    pub(super) threshold: f64,
    /// Inclusive threshold retaining only the highest-confidence row.
    pub(super) filtered_threshold: f64,
    /// Diagnostic label for this contract.
    pub(super) label: &'a str,
}

/// Runs decoding, protocol rejection, ownership, and cleanup checks.
pub(super) fn run(
    helper: &mut NativeHelper,
    contract: FieldMajorDetectionContract<'_>,
) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    let image = matrix(helper, contract.pixels, 1, contract.fields, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(contract.fields), int_arg(1)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(contract.model)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let shape = expect_ints(helper.call("opencv.dnn.output_shape", &[output.arg()])?)?;
    if shape != [1, contract.fields, 3] {
        return Err(format!(
            "unexpected {} output shape: {shape:?}",
            contract.label
        ));
    }

    let detections = decode(helper, contract.operation, &output, contract.threshold, 3)?;
    assert_detections(helper, contract.label, &detections, &[1, 0, 0])?;
    let repeated = decode(helper, contract.operation, &output, contract.threshold, 3)?;
    assert_detections(helper, contract.label, &repeated, &[1, 0, 0])?;
    let filtered = decode(
        helper,
        contract.operation,
        &output,
        contract.filtered_threshold,
        3,
    )?;
    assert_detections(helper, contract.label, &filtered, &[1])?;
    let bounded = decode(helper, contract.operation, &output, 0.0, 1)?;
    assert_detections(helper, contract.label, &bounded, &[1])?;

    expect_error_code(
        helper.call(
            contract.operation,
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(contract.threshold),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            contract.operation,
            &[model.arg(), float_arg(contract.threshold), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.5, 0), (0.5, 10001)] {
        expect_error_code(
            helper.call(
                contract.operation,
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            contract.operation,
            &[output.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, contract.label, &detections, &[1, 0, 0])?;
    for owner in [&bounded, &filtered, &repeated, &detections] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Decodes one field-major output through the generated protocol.
fn decode(
    helper: &mut NativeHelper,
    operation: &str,
    output: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            operation,
            &[output.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        "opencv.Dnn.Detections",
    )
}

/// Requires canonical classes, descending confidence, and normalized boxes.
fn assert_detections(
    helper: &mut NativeHelper,
    label: &str,
    detections: &NativeHandle,
    classes: &[i64],
) -> Result<(), String> {
    expect_int(
        helper.call("opencv.dnn.detection_count", &[detections.arg()])?,
        classes.len() as i64,
    )?;
    let actual =
        expect_ints(helper.call("opencv.dnn.detection_class_indices", &[detections.arg()])?)?;
    let confidence =
        expect_floats(helper.call("opencv.dnn.detection_confidences", &[detections.arg()])?)?;
    let boxes = expect_floats(helper.call("opencv.dnn.detection_boxes", &[detections.arg()])?)?;
    if actual != classes
        || confidence.len() != classes.len()
        || boxes.len() != classes.len() * 4
        || !confidence.windows(2).all(|pair| pair[0] >= pair[1])
        || !boxes.iter().all(|value| (0.0..=1.0).contains(value))
    {
        return Err(format!(
            "malformed {label} detections: classes={actual:?}, confidence={confidence:?}, boxes={boxes:?}"
        ));
    }
    Ok(())
}
