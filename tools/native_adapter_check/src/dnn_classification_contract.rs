//! Deterministic single-batch DNN classification projection contract.

use crate::protocol::{NativeRecordValue, NativeReply};

use super::{
    bytes_arg, expect_error_code, expect_floats, expect_int, expect_ints, expect_typed_handle,
    expect_unit, float_arg, int_arg, matrix, NativeHandle, NativeHelper,
};

const UNIQUE_GREEN_PIXEL: [u8; 12] = [0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0];
pub(super) const FLATTEN_ONNX: [u8; 137] = [
    8, 8, 18, 18, 116, 101, 114, 108, 97, 110, 45, 111, 112, 101, 110, 99, 118, 45, 116, 101, 115,
    116, 58, 109, 10, 33, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111, 117, 116, 112, 117, 116, 26,
    7, 102, 108, 97, 116, 116, 101, 110, 34, 7, 70, 108, 97, 116, 116, 101, 110, 18, 13, 102, 108,
    97, 116, 116, 101, 110, 95, 103, 114, 97, 112, 104, 90, 31, 10, 5, 105, 110, 112, 117, 116, 18,
    22, 10, 20, 8, 1, 18, 16, 10, 2, 8, 1, 10, 2, 8, 3, 10, 2, 8, 2, 10, 2, 8, 2, 98, 24, 10, 6,
    111, 117, 116, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 1, 10, 2, 8, 12, 66, 2,
    16, 13,
];

/// Runs deterministic value, malformed-argument, owner, type, and stale-handle checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(
        helper.call("opencv.dnn.live_classifications_count", &[])?,
        0,
    )?;
    let image = matrix(helper, &UNIQUE_GREEN_PIXEL, 2, 2, "bgr")?;
    let blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let model = expect_typed_handle(
        helper.call("opencv.dnn.load_onnx", &[bytes_arg(&FLATTEN_ONNX)])?,
        "opencv.Dnn.OnnxModel",
    )?;
    let output = forward(helper, &model, &blob)?;
    expect_classification(
        helper.call("opencv.dnn.top_classification", &[output.arg()])?,
        5,
        1.0,
    )?;
    let top_three = top_rankings(helper, &output, 3)?;
    expect_rankings(helper, &top_three, &[5, 0, 1], &[1.0, 0.0, 0.0])?;
    let repeated_rankings = top_rankings(helper, &output, 3)?;
    expect_rankings(helper, &repeated_rankings, &[5, 0, 1], &[1.0, 0.0, 0.0])?;
    let top_one = top_rankings(helper, &output, 1)?;
    expect_rankings(helper, &top_one, &[5], &[1.0])?;
    let all = top_rankings(helper, &output, 100)?;
    expect_int(
        helper.call("opencv.dnn.top_classification_count", &[all.arg()])?,
        12,
    )?;
    let positive_candidates = candidate_rankings(helper, &output, 0.5, 3)?;
    expect_rankings(helper, &positive_candidates, &[5], &[1.0])?;
    let inclusive_candidates = candidate_rankings(helper, &output, 1.0, 3)?;
    expect_rankings(helper, &inclusive_candidates, &[5], &[1.0])?;
    let empty_candidates = candidate_rankings(helper, &output, 1.1, 3)?;
    expect_rankings(helper, &empty_candidates, &[], &[])?;
    let limited_candidates = candidate_rankings(helper, &output, 0.0, 3)?;
    expect_rankings(helper, &limited_candidates, &[5, 0, 1], &[1.0, 0.0, 0.0])?;
    let softmax = softmax_rankings(helper, &output, 3)?;
    let denominator = std::f64::consts::E + 11.0;
    expect_rankings_approx(
        helper,
        &softmax,
        &[5, 0, 1],
        &[
            std::f64::consts::E / denominator,
            1.0 / denominator,
            1.0 / denominator,
        ],
    )?;
    let sigmoid = sigmoid_rankings(helper, &output, 3)?;
    expect_rankings_approx(
        helper,
        &sigmoid,
        &[5, 0, 1],
        &[1.0 / (1.0 + (-1.0_f64).exp()), 0.5, 0.5],
    )?;
    let sigmoid_positive = sigmoid_candidate_rankings(helper, &output, 0.6, 3)?;
    expect_rankings_approx(
        helper,
        &sigmoid_positive,
        &[5],
        &[1.0 / (1.0 + (-1.0_f64).exp())],
    )?;
    let sigmoid_inclusive = sigmoid_candidate_rankings(helper, &output, 0.5, 3)?;
    expect_rankings_approx(
        helper,
        &sigmoid_inclusive,
        &[5, 0, 1],
        &[1.0 / (1.0 + (-1.0_f64).exp()), 0.5, 0.5],
    )?;
    let sigmoid_empty = sigmoid_candidate_rankings(helper, &output, 0.8, 3)?;
    expect_rankings(helper, &sigmoid_empty, &[], &[])?;
    expect_int(
        helper.call("opencv.dnn.live_classifications_count", &[])?,
        13,
    )?;
    for maximum in [-1, 0, 10001] {
        expect_error_code(
            helper.call(
                "opencv.dnn.top_classifications",
                &[output.arg(), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.classification_candidates",
                &[output.arg(), float_arg(0.0), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.softmax_classifications",
                &[output.arg(), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_classifications",
                &[output.arg(), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_classification_candidates",
                &[output.arg(), float_arg(0.5), int_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for minimum_score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        expect_error_code(
            helper.call(
                "opencv.dnn.classification_candidates",
                &[output.arg(), float_arg(minimum_score), int_arg(3)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for minimum_probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        expect_error_code(
            helper.call(
                "opencv.dnn.sigmoid_classification_candidates",
                &[output.arg(), float_arg(minimum_probability), int_arg(3)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_classification(
        helper.call("opencv.dnn.top_classification", &[output.arg()])?,
        5,
        1.0,
    )?;

    let zero_image = matrix(helper, &[0; 12], 2, 2, "bgr")?;
    let zero_blob = expect_typed_handle(
        helper.call(
            "opencv.dnn.blob_from_bgr_rgb",
            &[zero_image.arg(), int_arg(2), int_arg(2)],
        )?,
        "opencv.Dnn.Blob",
    )?;
    let tie_output = forward(helper, &model, &zero_blob)?;
    expect_classification(
        helper.call("opencv.dnn.top_classification", &[tie_output.arg()])?,
        0,
        0.0,
    )?;
    let tie_rankings = top_rankings(helper, &tie_output, 3)?;
    expect_rankings(helper, &tie_rankings, &[0, 1, 2], &[0.0, 0.0, 0.0])?;
    let tie_candidates = candidate_rankings(helper, &tie_output, 0.0, 2)?;
    expect_rankings(helper, &tie_candidates, &[0, 1], &[0.0, 0.0])?;
    let tie_softmax = softmax_rankings(helper, &tie_output, 3)?;
    expect_rankings_approx(
        helper,
        &tie_softmax,
        &[0, 1, 2],
        &[1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0],
    )?;
    let tie_sigmoid = sigmoid_rankings(helper, &tie_output, 3)?;
    expect_rankings_approx(helper, &tie_sigmoid, &[0, 1, 2], &[0.5, 0.5, 0.5])?;
    let tie_sigmoid_candidates = sigmoid_candidate_rankings(helper, &tie_output, 0.5, 2)?;
    expect_rankings_approx(helper, &tie_sigmoid_candidates, &[0, 1], &[0.5, 0.5])?;
    expect_unit(helper.call(
        "opencv.dnn.dispose_classifications",
        &[tie_sigmoid_candidates.arg()],
    )?)?;
    expect_unit(helper.call("opencv.dnn.dispose_classifications", &[tie_sigmoid.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_classifications", &[tie_softmax.arg()])?)?;
    expect_unit(helper.call(
        "opencv.dnn.dispose_classifications",
        &[tie_candidates.arg()],
    )?)?;
    expect_unit(helper.call("opencv.dnn.dispose_classifications", &[tie_rankings.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose_output", &[tie_output.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[zero_blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[zero_image.arg()])?)?;

    expect_error_code(
        helper.call(
            "opencv.dnn.top_classification",
            &[output.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.top_classification", &[model.arg()])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.top_classification", &["i:1".to_owned()])?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.top_classifications",
            &[output.arg_with_owner("foreign-worker"), int_arg(3)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.top_classifications", &[model.arg(), int_arg(3)])?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.top_classifications",
            &[output.arg(), "f:3.0".to_owned()],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.classification_candidates",
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(0.0),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.classification_candidates",
            &[model.arg(), float_arg(0.0), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.classification_candidates",
            &[output.arg(), "i:0".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_classifications",
            &[output.arg_with_owner("foreign-worker"), int_arg(3)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_classifications",
            &[model.arg(), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.softmax_classifications",
            &[output.arg(), "f:3.0".to_owned()],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_classifications",
            &[output.arg_with_owner("foreign-worker"), int_arg(3)],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_classifications",
            &[model.arg(), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_classifications",
            &[output.arg(), "f:3.0".to_owned()],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_classification_candidates",
            &[
                output.arg_with_owner("foreign-worker"),
                float_arg(0.5),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_classification_candidates",
            &[model.arg(), float_arg(0.5), int_arg(3)],
        )?,
        "handle_type_mismatch",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.sigmoid_classification_candidates",
            &[output.arg(), "i:1".to_owned(), int_arg(3)],
        )?,
        "invalid_arguments",
    )?;
    expect_error_code(
        helper.call(
            "opencv.dnn.top_classification_indices",
            &[top_three.arg_with_owner("foreign-worker")],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call("opencv.dnn.top_classification_scores", &[model.arg()])?,
        "handle_type_mismatch",
    )?;

    expect_unit(helper.call("opencv.dnn.dispose_output", &[output.arg()])?)?;
    expect_rankings(helper, &top_three, &[5, 0, 1], &[1.0, 0.0, 0.0])?;
    expect_rankings(helper, &positive_candidates, &[5], &[1.0])?;
    expect_rankings_approx(
        helper,
        &softmax,
        &[5, 0, 1],
        &[
            std::f64::consts::E / denominator,
            1.0 / denominator,
            1.0 / denominator,
        ],
    )?;
    expect_rankings_approx(
        helper,
        &sigmoid,
        &[5, 0, 1],
        &[1.0 / (1.0 + (-1.0_f64).exp()), 0.5, 0.5],
    )?;
    expect_rankings_approx(
        helper,
        &sigmoid_positive,
        &[5],
        &[1.0 / (1.0 + (-1.0_f64).exp())],
    )?;
    expect_error_code(
        helper.call("opencv.dnn.top_classification", &[output.arg()])?,
        "stale_handle",
    )?;
    expect_unit(helper.call("opencv.dnn.dispose_classifications", &[top_three.arg()])?)?;
    expect_error_code(
        helper.call("opencv.dnn.top_classification_count", &[top_three.arg()])?,
        "stale_handle",
    )?;
    for classifications in [
        &limited_candidates,
        &empty_candidates,
        &inclusive_candidates,
        &positive_candidates,
        &all,
        &top_one,
        &repeated_rankings,
        &softmax,
        &sigmoid,
        &sigmoid_positive,
        &sigmoid_inclusive,
        &sigmoid_empty,
    ] {
        expect_unit(helper.call(
            "opencv.dnn.dispose_classifications",
            &[classifications.arg()],
        )?)?;
    }
    expect_int(
        helper.call("opencv.dnn.live_classifications_count", &[])?,
        0,
    )?;
    expect_unit(helper.call("opencv.dnn.dispose_model", &[model.arg()])?)?;
    expect_unit(helper.call("opencv.dnn.dispose", &[blob.arg()])?)?;
    expect_unit(helper.call("opencv.mat.dispose", &[image.arg()])?)
}

/// Creates one typed threshold-filtered stable-sigmoid ranking owner.
fn sigmoid_candidate_rankings(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    minimum_probability: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.sigmoid_classification_candidates",
            &[
                output.arg(),
                float_arg(minimum_probability),
                int_arg(maximum),
            ],
        )?,
        "opencv.Dnn.Classifications",
    )
}

/// Creates one typed bounded stable-sigmoid ranking owner.
fn sigmoid_rankings(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.sigmoid_classifications",
            &[output.arg(), int_arg(maximum)],
        )?,
        "opencv.Dnn.Classifications",
    )
}

/// Creates one typed bounded stable-softmax ranking owner.
fn softmax_rankings(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.softmax_classifications",
            &[output.arg(), int_arg(maximum)],
        )?,
        "opencv.Dnn.Classifications",
    )
}

/// Creates one typed threshold-filtered classification owner.
fn candidate_rankings(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    minimum_score: f64,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.classification_candidates",
            &[output.arg(), float_arg(minimum_score), int_arg(maximum)],
        )?,
        "opencv.Dnn.Classifications",
    )
}

/// Creates one typed bounded ranked-classification owner.
fn top_rankings(
    helper: &mut NativeHelper,
    output: &NativeHandle,
    maximum: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.dnn.top_classifications",
            &[output.arg(), int_arg(maximum)],
        )?,
        "opencv.Dnn.Classifications",
    )
}

/// Requires aligned class indexes and raw scores in canonical ranking order.
fn expect_rankings(
    helper: &mut NativeHelper,
    classifications: &NativeHandle,
    expected_indices: &[i64],
    expected_scores: &[f64],
) -> Result<(), String> {
    expect_int(
        helper.call(
            "opencv.dnn.top_classification_count",
            &[classifications.arg()],
        )?,
        expected_indices.len() as i64,
    )?;
    let indices = expect_ints(helper.call(
        "opencv.dnn.top_classification_indices",
        &[classifications.arg()],
    )?)?;
    let scores = expect_floats(helper.call(
        "opencv.dnn.top_classification_scores",
        &[classifications.arg()],
    )?)?;
    if indices != expected_indices || scores != expected_scores {
        return Err(format!(
            "unexpected top classifications: {indices:?} {scores:?}"
        ));
    }
    Ok(())
}

/// Requires aligned class indexes and approximately equal probability scores.
fn expect_rankings_approx(
    helper: &mut NativeHelper,
    classifications: &NativeHandle,
    expected_indices: &[i64],
    expected_scores: &[f64],
) -> Result<(), String> {
    expect_int(
        helper.call(
            "opencv.dnn.top_classification_count",
            &[classifications.arg()],
        )?,
        expected_indices.len() as i64,
    )?;
    let indices = expect_ints(helper.call(
        "opencv.dnn.top_classification_indices",
        &[classifications.arg()],
    )?)?;
    let scores = expect_floats(helper.call(
        "opencv.dnn.top_classification_scores",
        &[classifications.arg()],
    )?)?;
    if indices != expected_indices
        || scores.len() != expected_scores.len()
        || scores
            .iter()
            .zip(expected_scores)
            .any(|(actual, expected)| (actual - expected).abs() >= 1.0e-12)
    {
        return Err(format!(
            "unexpected softmax classifications: {indices:?} {scores:?}"
        ));
    }
    Ok(())
}

/// Runs one model/blob forward operation through the generated helper.
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

/// Requires the copied class-index and raw-score fields from one projection.
fn expect_classification(
    reply: NativeReply,
    expected_index: i64,
    expected_score: f64,
) -> Result<(), String> {
    let NativeReply::Record { type_name, fields } = reply else {
        return Err("expected copied Classification record".to_owned());
    };
    if type_name != "Classification"
        || fields.get("class_index") != Some(&NativeRecordValue::Int(expected_index))
        || fields.get("score") != Some(&NativeRecordValue::Float(expected_score))
        || fields.len() != 2
    {
        return Err(format!(
            "unexpected classification record: {type_name} {fields:?}"
        ));
    }
    Ok(())
}
