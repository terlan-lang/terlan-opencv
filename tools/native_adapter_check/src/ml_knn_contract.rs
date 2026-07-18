//! Bounded K-nearest-neighbor classifier ownership contract.

use super::{
    expect_error_code, expect_int, expect_ints, expect_typed_handle, expect_unit, float_list_arg,
    int_arg, int_list_arg, NativeHandle, NativeHelper,
};

const MODEL_TYPE: &str = "opencv.Ml.KnnModel";
const TRAINING: [f64; 16] = [
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 10.0, 10.0, 10.0, 11.0, 11.0, 10.0, 11.0, 11.0,
];
const LABELS: [i64; 8] = [0, 0, 0, 0, 1, 1, 1, 1];

/// Runs deterministic prediction, lifecycle, malformed-input, and bound checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.ml.live_count", &[])?, 0)?;
    let model = train(helper, &TRAINING, &LABELS, 2, 3)?;
    expect_int(helper.call("opencv.ml.live_count", &[])?, 1)?;
    expect_int(helper.call("opencv.ml.sample_count", &[model.arg()])?, 8)?;
    expect_int(helper.call("opencv.ml.feature_count", &[model.arg()])?, 2)?;

    let query = [0.2, 0.2, 10.2, 10.2];
    expect_predictions(helper, &model, &query, 3, &[0, 1])?;
    expect_predictions(helper, &model, &query, 3, &[0, 1])?;
    expect_predictions(helper, &model, &[0.0, 0.0], 1, &[0])?;

    expect_error_code(
        helper.call(
            "opencv.ml.predict",
            &[
                model.arg_with_owner("foreign-worker"),
                float_list_arg(&query),
                int_arg(3),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict",
            &[
                model.arg_with_type("opencv.Mat.Mat"),
                float_list_arg(&query),
                int_arg(3),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    for (samples, k) in [
        (vec![], 3),
        (vec![0.0], 3),
        (vec![0.0, 0.0], 0),
        (vec![0.0, 0.0], 9),
        (vec![f64::NAN, 0.0], 1),
        (vec![f64::INFINITY, 0.0], 1),
        (vec![1_000_001.0, 0.0], 1),
    ] {
        expect_error_code(
            helper.call(
                "opencv.ml.predict",
                &[model.arg(), float_list_arg(&samples), int_arg(k)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for (samples, labels, features, k) in [
        (vec![], vec![], 2, 1),
        (TRAINING.to_vec(), LABELS[..7].to_vec(), 2, 3),
        (TRAINING[..15].to_vec(), LABELS.to_vec(), 2, 3),
        (TRAINING.to_vec(), LABELS.to_vec(), 0, 3),
        (TRAINING.to_vec(), LABELS.to_vec(), 4097, 3),
        (TRAINING.to_vec(), LABELS.to_vec(), 2, 0),
        (TRAINING.to_vec(), LABELS.to_vec(), 2, 9),
    ] {
        expect_train_invalid(helper, &samples, &labels, features, k)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut samples = TRAINING;
        samples[0] = invalid;
        expect_train_invalid(helper, &samples, &LABELS, 2, 3)?;
    }
    for invalid in [-1_000_001, 1_000_001, i64::MIN, i64::MAX] {
        let mut labels = LABELS;
        labels[0] = invalid;
        expect_train_invalid(helper, &TRAINING, &labels, 2, 3)?;
    }
    expect_error_code(
        helper.call(
            "opencv.ml.train_knn_classifier",
            &[
                "li:0,0,1,1".to_owned(),
                int_list_arg(&[0, 1]),
                int_arg(2),
                int_arg(1),
            ],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.ml.dispose", &[model.arg()])?)?;
    expect_error_code(
        helper.call("opencv.ml.sample_count", &[model.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict",
            &[model.arg(), float_list_arg(&query), int_arg(3)],
        )?,
        "stale_handle",
    )?;
    expect_int(helper.call("opencv.ml.live_count", &[])?, 0)
}

/// Trains one typed model handle through the generated helper.
fn train(
    helper: &mut NativeHelper,
    samples: &[f64],
    labels: &[i64],
    feature_count: i64,
    default_k: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.ml.train_knn_classifier",
            &[
                float_list_arg(samples),
                int_list_arg(labels),
                int_arg(feature_count),
                int_arg(default_k),
            ],
        )?,
        MODEL_TYPE,
    )
}

/// Requires exact copied predictions for one query matrix.
fn expect_predictions(
    helper: &mut NativeHelper,
    model: &NativeHandle,
    samples: &[f64],
    k: i64,
    expected: &[i64],
) -> Result<(), String> {
    let actual = expect_ints(helper.call(
        "opencv.ml.predict",
        &[model.arg(), float_list_arg(samples), int_arg(k)],
    )?)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "expected predictions {expected:?}, received {actual:?}"
        ))
    }
}

/// Requires one malformed training request to fail without allocating a model.
fn expect_train_invalid(
    helper: &mut NativeHelper,
    samples: &[f64],
    labels: &[i64],
    feature_count: i64,
    default_k: i64,
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.ml.train_knn_classifier",
            &[
                float_list_arg(samples),
                int_list_arg(labels),
                int_arg(feature_count),
                int_arg(default_k),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_int(helper.call("opencv.ml.live_count", &[])?, 1)
}
