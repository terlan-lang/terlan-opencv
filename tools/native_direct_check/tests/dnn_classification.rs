//! Direct deterministic DNN classification projection tests.

mod support;

use support::{FLATTEN_ONNX, IDENTITY_ONNX};
use terlan_opencv_native::ffi;

const UNIQUE_GREEN_PIXEL: [u8; 12] = [0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0];

/// Proves fixed-shape classification, deterministic projection, and shape rejection.
#[test]
fn projects_one_classification_batch() {
    assert_eq!(ffi::live_dnn_classifications_count(), 0);
    let image = ffi::mat_from_u8(&UNIQUE_GREEN_PIXEL, 2, 2, 3);
    let blob = image
        .as_ref()
        .expect("BGR image")
        .blob_from_bgr_rgb_owned(2, 2);
    let flatten_model = ffi::load_onnx_model_owned(FLATTEN_ONNX);
    let flatten_output = flatten_model
        .as_ref()
        .expect("flatten model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    let classification =
        ffi::top_classification_owned(flatten_output.as_ref().expect("flattened class scores"));
    let classification_ref = classification.as_ref().expect("top classification");
    assert_eq!(classification_ref.class_index_value(), 5);
    assert_eq!(classification_ref.score_value(), 1.0);
    let repeated =
        ffi::top_classification_owned(flatten_output.as_ref().expect("flattened class scores"));
    assert_eq!(
        repeated
            .as_ref()
            .expect("repeated classification")
            .class_index_value(),
        5
    );
    let top_three =
        ffi::top_classifications_owned(flatten_output.as_ref().expect("flattened class scores"), 3);
    assert_rankings(&top_three, &[5, 0, 1], &[1.0, 0.0, 0.0]);
    let repeated_rankings =
        ffi::top_classifications_owned(flatten_output.as_ref().expect("flattened class scores"), 3);
    assert_rankings(&repeated_rankings, &[5, 0, 1], &[1.0, 0.0, 0.0]);
    let top_one =
        ffi::top_classifications_owned(flatten_output.as_ref().expect("flattened class scores"), 1);
    assert_rankings(&top_one, &[5], &[1.0]);
    let all = ffi::top_classifications_owned(
        flatten_output.as_ref().expect("flattened class scores"),
        100,
    );
    assert_eq!(all.as_ref().expect("all classifications").count(), 12);
    let positive_candidates = ffi::classification_candidates_owned(
        flatten_output.as_ref().expect("flattened class scores"),
        0.5,
        3,
    );
    assert_rankings(&positive_candidates, &[5], &[1.0]);
    let inclusive_candidates = ffi::classification_candidates_owned(
        flatten_output.as_ref().expect("flattened class scores"),
        1.0,
        3,
    );
    assert_rankings(&inclusive_candidates, &[5], &[1.0]);
    let empty_candidates = ffi::classification_candidates_owned(
        flatten_output.as_ref().expect("flattened class scores"),
        1.1,
        3,
    );
    assert_rankings(&empty_candidates, &[], &[]);
    let limited_candidates = ffi::classification_candidates_owned(
        flatten_output.as_ref().expect("flattened class scores"),
        0.0,
        3,
    );
    assert_rankings(&limited_candidates, &[5, 0, 1], &[1.0, 0.0, 0.0]);
    let softmax = flatten_output
        .as_ref()
        .expect("flattened class scores")
        .softmax_classifications_owned(3);
    let denominator = std::f64::consts::E + 11.0;
    assert_rankings_approx(
        &softmax,
        &[5, 0, 1],
        &[
            std::f64::consts::E / denominator,
            1.0 / denominator,
            1.0 / denominator,
        ],
    );
    let sigmoid = flatten_output
        .as_ref()
        .expect("flattened class scores")
        .sigmoid_classifications_owned(3);
    assert_rankings_approx(
        &sigmoid,
        &[5, 0, 1],
        &[1.0 / (1.0 + (-1.0_f64).exp()), 0.5, 0.5],
    );
    let sigmoid_positive = flatten_output
        .as_ref()
        .expect("flattened class scores")
        .sigmoid_classification_candidates_owned(0.6, 3);
    assert_rankings_approx(&sigmoid_positive, &[5], &[1.0 / (1.0 + (-1.0_f64).exp())]);
    let sigmoid_inclusive = flatten_output
        .as_ref()
        .expect("flattened class scores")
        .sigmoid_classification_candidates_owned(0.5, 3);
    assert_rankings_approx(
        &sigmoid_inclusive,
        &[5, 0, 1],
        &[1.0 / (1.0 + (-1.0_f64).exp()), 0.5, 0.5],
    );
    let sigmoid_empty = flatten_output
        .as_ref()
        .expect("flattened class scores")
        .sigmoid_classification_candidates_owned(0.8, 3);
    assert_rankings(&sigmoid_empty, &[], &[]);
    assert_eq!(ffi::live_dnn_classifications_count(), 13);
    for maximum in [-1, 0, 10001] {
        assert!(ffi::top_classifications_owned(
            flatten_output.as_ref().expect("flattened class scores"),
            maximum,
        )
        .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(flatten_output
            .as_ref()
            .expect("flattened class scores")
            .sigmoid_classification_candidates_owned(0.5, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(flatten_output
            .as_ref()
            .expect("flattened class scores")
            .softmax_classifications_owned(maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(flatten_output
            .as_ref()
            .expect("flattened class scores")
            .sigmoid_classifications_owned(maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(ffi::classification_candidates_owned(
            flatten_output.as_ref().expect("flattened class scores"),
            0.0,
            maximum,
        )
        .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for minimum_score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(ffi::classification_candidates_owned(
            flatten_output.as_ref().expect("flattened class scores"),
            minimum_score,
            3,
        )
        .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for minimum_probability in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(flatten_output
            .as_ref()
            .expect("flattened class scores")
            .sigmoid_classification_candidates_owned(minimum_probability, 3)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    let zero_image = ffi::mat_zeros(2, 2, 3);
    let zero_blob = zero_image
        .as_ref()
        .expect("zero BGR image")
        .blob_from_bgr_rgb_owned(2, 2);
    let tie_output = flatten_model
        .as_ref()
        .expect("flatten model")
        .forward_owned(zero_blob.as_ref().expect("zero RGB blob"));
    let tie = ffi::top_classification_owned(tie_output.as_ref().expect("equal class scores"));
    let tie_ref = tie.as_ref().expect("tie classification");
    assert_eq!(tie_ref.class_index_value(), 0);
    assert_eq!(tie_ref.score_value(), 0.0);
    let tie_rankings =
        ffi::top_classifications_owned(tie_output.as_ref().expect("equal class scores"), 3);
    assert_rankings(&tie_rankings, &[0, 1, 2], &[0.0, 0.0, 0.0]);
    let tie_candidates = ffi::classification_candidates_owned(
        tie_output.as_ref().expect("equal class scores"),
        0.0,
        2,
    );
    assert_rankings(&tie_candidates, &[0, 1], &[0.0, 0.0]);
    let tie_softmax = tie_output
        .as_ref()
        .expect("equal class scores")
        .softmax_classifications_owned(3);
    assert_rankings_approx(
        &tie_softmax,
        &[0, 1, 2],
        &[1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0],
    );
    let tie_sigmoid = tie_output
        .as_ref()
        .expect("equal class scores")
        .sigmoid_classifications_owned(3);
    assert_rankings_approx(&tie_sigmoid, &[0, 1, 2], &[0.5, 0.5, 0.5]);
    let tie_sigmoid_candidates = tie_output
        .as_ref()
        .expect("equal class scores")
        .sigmoid_classification_candidates_owned(0.5, 2);
    assert_rankings_approx(&tie_sigmoid_candidates, &[0, 1], &[0.5, 0.5]);
    drop(tie_sigmoid_candidates);
    drop(tie_sigmoid);
    drop(tie_softmax);
    drop(tie_candidates);
    drop(tie_rankings);

    let identity_model = ffi::load_onnx_model_owned(IDENTITY_ONNX);
    let identity_output = identity_model
        .as_ref()
        .expect("identity model")
        .forward_owned(blob.as_ref().expect("RGB blob"));
    assert!(
        ffi::top_classification_owned(identity_output.as_ref().expect("rank-four output"))
            .is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(
        ffi::top_classifications_owned(identity_output.as_ref().expect("rank-four output"), 3,)
            .is_null()
    );
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(ffi::classification_candidates_owned(
        identity_output.as_ref().expect("rank-four output"),
        0.0,
        3,
    )
    .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(identity_output
        .as_ref()
        .expect("rank-four output")
        .softmax_classifications_owned(3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(identity_output
        .as_ref()
        .expect("rank-four output")
        .sigmoid_classification_candidates_owned(0.5, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(identity_output
        .as_ref()
        .expect("rank-four output")
        .sigmoid_classifications_owned(3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(flatten_output);
    assert_rankings(&top_three, &[5, 0, 1], &[1.0, 0.0, 0.0]);
    assert_rankings(&positive_candidates, &[5], &[1.0]);
    assert_rankings_approx(
        &softmax,
        &[5, 0, 1],
        &[
            std::f64::consts::E / denominator,
            1.0 / denominator,
            1.0 / denominator,
        ],
    );
    assert_rankings_approx(
        &sigmoid,
        &[5, 0, 1],
        &[1.0 / (1.0 + (-1.0_f64).exp()), 0.5, 0.5],
    );
    assert_rankings_approx(&sigmoid_positive, &[5], &[1.0 / (1.0 + (-1.0_f64).exp())]);

    drop(identity_output);
    drop(identity_model);
    drop(tie);
    drop(tie_output);
    drop(zero_blob);
    drop(zero_image);
    drop(repeated);
    drop(classification);
    drop(sigmoid_empty);
    drop(sigmoid_inclusive);
    drop(sigmoid_positive);
    drop(sigmoid);
    drop(softmax);
    drop(limited_candidates);
    drop(empty_candidates);
    drop(inclusive_candidates);
    drop(positive_candidates);
    drop(all);
    drop(top_one);
    drop(repeated_rankings);
    drop(top_three);
    drop(flatten_model);
    drop(blob);
    drop(image);
    assert_eq!(ffi::live_dnn_output_count(), 0);
    assert_eq!(ffi::live_onnx_model_count(), 0);
    assert_eq!(ffi::live_dnn_blob_count(), 0);
    assert_eq!(ffi::live_dnn_classifications_count(), 0);
}

/// Requires one owned ranking whose copied scores match within float tolerance.
fn assert_rankings_approx(
    classifications: &cxx::UniquePtr<ffi::OpenCvDnnClassifications>,
    indices: &[i64],
    scores: &[f64],
) {
    let classifications = classifications.as_ref().expect("softmax classifications");
    assert_eq!(classifications.count(), indices.len() as i64);
    assert_eq!(copied_indices(classifications), indices);
    let actual = copied_scores(classifications);
    assert_eq!(actual.len(), scores.len());
    for (actual, expected) in actual.iter().zip(scores) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

/// Requires one owned canonical ranking and aligned copied projections.
fn assert_rankings(
    classifications: &cxx::UniquePtr<ffi::OpenCvDnnClassifications>,
    indices: &[i64],
    scores: &[f64],
) {
    let classifications = classifications.as_ref().expect("top classifications");
    assert_eq!(classifications.count(), indices.len() as i64);
    assert_eq!(copied_indices(classifications), indices);
    assert_eq!(copied_scores(classifications), scores);
}

/// Copies ranked class indexes from one generated owner.
fn copied_indices(classifications: &ffi::OpenCvDnnClassifications) -> Vec<i64> {
    classifications
        .class_indices_owned()
        .as_ref()
        .expect("copied class indexes")
        .iter()
        .copied()
        .collect()
}

/// Copies ranked raw scores from one generated owner.
fn copied_scores(classifications: &ffi::OpenCvDnnClassifications) -> Vec<f64> {
    classifications
        .scores_owned()
        .as_ref()
        .expect("copied class scores")
        .iter()
        .copied()
        .collect()
}
