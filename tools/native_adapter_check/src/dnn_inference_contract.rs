//! Fixed CPU in-memory ONNX inference ownership contract.

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, int_arg, matrix, NativeHandle, NativeHelper,
};

const MODEL_TYPE: &str = "opencv.Dnn.OnnxModel";
const OUTPUT_TYPE: &str = "opencv.Dnn.Output";
pub(super) const BGR_PIXELS: [u8; 12] = [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255];
const EXPECTED_VALUES: [f64; 12] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const GRAYSCALE_PIXELS: [u8; 4] = [0, 64, 128, 255];
const EXPECTED_GRAYSCALE_VALUES: [f64; 4] = [
    0.0,
    (64.0_f32 / 255.0_f32) as f64,
    (128.0_f32 / 255.0_f32) as f64,
    1.0,
];
pub(super) const IDENTITY_ONNX: &[u8] = include_bytes!("../../../tests/fixtures/identity_rgb.onnx");
pub(super) const GRAYSCALE_IDENTITY_ONNX: &[u8] =
    include_bytes!("../../../tests/fixtures/identity_grayscale.onnx");

/// Runs exact-value, repeatability, lifecycle, owner, type, and malformed-input checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.dnn.live_model_count", &[])?, 0)?;
    expect_int(helper.call("opencv.dnn.live_output_count", &[])?, 0)?;
    let image = matrix(helper, &BGR_PIXELS, 2, 2, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = load_model(helper, IDENTITY_ONNX)?;
    let output = forward(helper, &model, &blob)?;
    expect_int(helper.call("opencv.dnn.live_model_count", &[])?, 1)?;
    expect_int(helper.call("opencv.dnn.live_output_count", &[])?, 1)?;
    expect_int(helper.call("opencv.dnn.output_rank", &[output.arg()])?, 4)?;
    let shape = expect_ints(helper.call("opencv.dnn.output_shape", &[output.arg()])?)?;
    if shape != [1, 3, 2, 2] {
        return Err(format!("unexpected identity output shape: {shape:?}"));
    }
    let values = expect_floats(helper.call("opencv.dnn.output_values", &[output.arg()])?)?;
    if values != EXPECTED_VALUES {
        return Err(format!("unexpected identity output values: {values:?}"));
    }
    expect_int(
        helper.call("opencv.dnn.output_value_count", &[output.arg()])?,
        12,
    )?;
    expect_output_slice(helper, &output, 0, 3, &[1.0, 0.0, 0.0])?;
    expect_output_slice(helper, &output, 3, 4, &[1.0, 0.0, 1.0, 0.0])?;
    expect_output_slice(helper, &output, 10, 99, &[1.0, 1.0])?;
    expect_output_slice(helper, &output, 12, 1, &[])?;
    expect_output_slice(helper, &output, 3, 4, &[1.0, 0.0, 1.0, 0.0])?;
    for (offset, maximum) in [(-1, 1), (0, 0), (0, 1_048_577), (13, 1)] {
        expect_error_code(
            helper.call(
                "opencv.dnn.output_values_slice",
                &[output.arg(), int_arg(offset), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    let repeated = forward(helper, &model, &blob)?;
    let repeated_values =
        expect_floats(helper.call("opencv.dnn.output_values", &[repeated.arg()])?)?;
    if repeated_values != EXPECTED_VALUES {
        return Err(format!(
            "non-repeatable identity output: {repeated_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose_output", &[repeated.arg()])?)?;

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
    expect_int(helper.call("opencv.dnn.live_model_count", &[])?, 2)?;
    expect_int(helper.call("opencv.dnn.live_output_count", &[])?, 2)?;
    let grayscale_shape =
        expect_ints(helper.call("opencv.dnn.output_shape", &[grayscale_output.arg()])?)?;
    if grayscale_shape != [1, 1, 2, 2] {
        return Err(format!(
            "unexpected grayscale identity shape: {grayscale_shape:?}"
        ));
    }
    let grayscale_values =
        expect_floats(helper.call("opencv.dnn.output_values", &[grayscale_output.arg()])?)?;
    if grayscale_values != EXPECTED_GRAYSCALE_VALUES {
        return Err(format!(
            "unexpected grayscale identity values: {grayscale_values:?}"
        ));
    }
    expect_int(
        helper.call("opencv.dnn.output_value_count", &[grayscale_output.arg()])?,
        4,
    )?;
    expect_output_slice(
        helper,
        &grayscale_output,
        1,
        2,
        &EXPECTED_GRAYSCALE_VALUES[1..3],
    )?;
    let repeated_grayscale = forward(helper, &grayscale_model, &grayscale_blob)?;
    let repeated_grayscale_values =
        expect_floats(helper.call("opencv.dnn.output_values", &[repeated_grayscale.arg()])?)?;
    if repeated_grayscale_values != EXPECTED_GRAYSCALE_VALUES {
        return Err(format!(
            "non-repeatable grayscale identity output: {repeated_grayscale_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose_output", &[repeated_grayscale.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[grayscale_model.arg()])?)?;
    let retained_grayscale_values =
        expect_floats(helper.call("opencv.dnn.output_values", &[grayscale_output.arg()])?)?;
    if retained_grayscale_values != EXPECTED_GRAYSCALE_VALUES {
        return Err("grayscale model disposal changed retained output".to_owned());
    }
    expect_unit(helper.call("opencv.dnn.dispose_output", &[grayscale_output.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[grayscale_blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[grayscale_image.arg()])?)?;
    expect_int(helper.call("opencv.dnn.live_model_count", &[])?, 1)?;
    expect_int(helper.call("opencv.dnn.live_output_count", &[])?, 1)?;

    expect_error_code(
        helper.call(
            "opencv.dnn.forward",
            &[model.arg_with_owner("foreign-worker"), blob.arg()],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.forward",
            &[model.arg(), blob.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.output_rank", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.output_values_slice",
            &[model.arg(), int_arg(0), int_arg(1)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.output_value_count",
            &[output.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.load_onnx", &["i:1".to_owned()])?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(&[])])?,
        "opencv.invalid_argument",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_int(helper.call("opencv.dnn.live_model_count", &[])?, 0)?;
    let retained_values = expect_floats(helper.call("opencv.dnn.output_values", &[output.arg()])?)?;
    if retained_values != EXPECTED_VALUES {
        return Err(format!(
            "model disposal changed retained output: {retained_values:?}"
        ));
    }
    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    expect_int(helper.call("opencv.dnn.live_output_count", &[])?, 0)?;
    expect_error_code(
        helper.call("opencv.dnn.output_rank", &[output.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.output_values_slice",
            &[output.arg(), int_arg(0), int_arg(1)],
        )?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Loads one typed identity-model handle through copied bytes.
fn load_model(helper: &mut NativeHelper, bytes: &[u8]) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(bytes)])?,
        MODEL_TYPE,
    )
}

/// Runs one typed model/blob inference operation.
fn forward(
    helper: &mut NativeHelper,
    model: &NativeHandle,
    blob: &NativeHandle,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call("opencv.dnn.forward", &[model.arg(), blob.arg()])?,
        OUTPUT_TYPE,
    )
}

/// Requires one output slice operation to equal the expected copied values.
fn expect_output_slice(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    offset: i64,
    maximum: i64,
    expected: &[f64],
) -> Result<(), String> {
    let values = expect_floats(helper.call(
        "opencv.dnn.output_values_slice",
        &[output.arg(), int_arg(offset), int_arg(maximum)],
    )?)?;
    if values == expected {
        Ok(())
    } else {
        Err(format!(
            "unexpected output slice at {offset} with limit {maximum}: {values:?}"
        ))
    }
}
