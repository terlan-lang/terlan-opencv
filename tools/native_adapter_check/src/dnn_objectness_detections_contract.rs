//! Inference-backed variable-class objectness detection contract.

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

const OBJECTNESS_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../tests/fixtures/objectness_reshape.onnx");
const BGR_OBJECTNESS_ROWS: [u8; 21] = [
    204, 102, 64, 204, 102, 64, 51, 102, 77, 51, 102, 77, 128, 229, 204, 128, 26, 204, 128, 229, 51,
];
const OPERATION: &str = "opencv.dnn.normalized_objectness_detections";

/// Runs objectness decoding, protocol rejection, ownership, and cleanup checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    let image = matrix(helper, &BGR_OBJECTNESS_ROWS, 1, 7, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(7), int_arg(1)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call(
            "opencv.dnn.load_onnx",
            &[bytes_arg(OBJECTNESS_RESHAPE_ONNX)],
        )?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "opencv.Dnn.Output",
    )?;
    let shape = expect_ints(helper.call("opencv.dnn.output_shape", &[output.arg()])?)?;
    if shape != [1, 3, 7] {
        return Err(format!("unexpected objectness output shape: {shape:?}"));
    }

    let detections = decode(helper, &output, 0.2, 3)?;
    assert_detections(helper, &detections, &[1, 0, 0])?;
    let repeated = decode(helper, &output, 0.2, 3)?;
    assert_detections(helper, &repeated, &[1, 0, 0])?;
    let filtered = decode(helper, &output, 0.7, 3)?;
    assert_detections(helper, &filtered, &[1])?;
    let bounded = decode(helper, &output, 0.0, 1)?;
    assert_detections(helper, &bounded, &[1])?;

    expect_error_code(
        helper.call(
            OPERATION,
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(0.2),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(OPERATION, &[model.arg(), float_arg(0.2), int_arg(3)])?,
        "handle_type_mismatch",
    )?;
    for (threshold, maximum) in [(-0.1, 3), (1.1, 3), (0.2, 0), (0.2, 10001)] {
        expect_error_code(
            helper.call(
                OPERATION,
                &[output.arg(), float_arg(threshold), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(OPERATION, &[output.arg(), "i:1".to_owned(), int_arg(3)])?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    assert_detections(helper, &detections, &[1, 0, 0])?;
    for owner in [&bounded, &filtered, &repeated, &detections] {
        expect_unit(helper.call("opencv.dnn.dispose_detections", &[owner.arg()])?)?;
    }
    expect_int(helper.call("opencv.dnn.live_detection_count", &[])?, 0)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Decodes one objectness output through the generated protocol.
fn decode(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    threshold: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            OPERATION,
            &[output.arg(), float_arg(threshold), int_arg(maximum)],
        )?,
        "opencv.Dnn.Detections",
    )
}

/// Requires canonical classes, confidence products, and normalized boxes.
fn assert_detections(
    helper: &mut NativeHelper,
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
            "malformed objectness detections: classes={actual:?}, confidence={confidence:?}, boxes={boxes:?}"
        ));
    }
    Ok(())
}
