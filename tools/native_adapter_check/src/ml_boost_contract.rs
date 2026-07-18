//! Bounded fixed-policy boosted-classifier ownership contract.

use super::{
    expect_error_code, expect_int, expect_ints, expect_typed_handle, expect_unit, float_list_arg,
    int_arg, int_list_arg, NativeHandle, NativeHelper,
};

const MODEL_TYPE: &str = "opencv.Ml.BoostModel";
const TRAINING: [f64; 16] = [
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 10.0, 10.0, 10.0, 11.0, 11.0, 10.0, 11.0, 11.0,
];
const LABELS: [i64; 8] = [-3, -3, -3, -3, 7, 7, 7, 7];

/// Runs deterministic prediction, lifecycle, malformed-input, and bound checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(helper.call("opencv.ml.live_boost_count", &[])?, 0)?;
    let model = train(helper, &TRAINING, &LABELS, 2)?;
    expect_int(helper.call("opencv.ml.live_boost_count", &[])?, 1)?;
    expect_int(
        helper.call("opencv.ml.boost_sample_count", &[model.arg()])?,
        8,
    )?;
    expect_int(
        helper.call("opencv.ml.boost_feature_count", &[model.arg()])?,
        2,
    )?;

    let query = [0.2, 0.2, 10.2, 10.2];
    expect_predictions(helper, &model, &query, &[-3, 7])?;
    expect_predictions(helper, &model, &query, &[-3, 7])?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict_boost",
            &[
                model.arg_with_owner("foreign-worker"),
                float_list_arg(&query),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict_boost",
            &[
                model.arg_with_type("opencv.Ml.DecisionTreeModel"),
                float_list_arg(&query),
            ],
        )?,
        "handle_type_mismatch",
    )?;
    for samples in [
        vec![],
        vec![0.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
    ] {
        expect_error_code(
            helper.call(
                "opencv.ml.predict_boost",
                &[model.arg(), float_list_arg(&samples)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for (samples, labels, features) in [
        (vec![], vec![], 2),
        (TRAINING.to_vec(), LABELS[..7].to_vec(), 2),
        (TRAINING[..15].to_vec(), LABELS.to_vec(), 2),
        (TRAINING.to_vec(), LABELS.to_vec(), 0),
        (TRAINING.to_vec(), LABELS.to_vec(), 4097),
        (TRAINING.to_vec(), vec![1; 8], 2),
        (TRAINING.to_vec(), vec![-3, -3, -3, 0, 7, 7, 7, 7], 2),
    ] {
        expect_train_invalid(helper, &samples, &labels, features)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut samples = TRAINING;
        samples[0] = invalid;
        expect_train_invalid(helper, &samples, &LABELS, 2)?;
    }
    for invalid in [-1_000_001, 1_000_001, i64::MIN, i64::MAX] {
        let mut labels = LABELS;
        labels[0] = invalid;
        expect_train_invalid(helper, &TRAINING, &labels, 2)?;
    }
    expect_error_code(
        helper.call(
            "opencv.ml.train_boost_classifier",
            &["li:0,0,1,1".to_owned(), int_list_arg(&[0, 1]), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.ml.dispose_boost", &[model.arg()])?)?;
    expect_error_code(
        helper.call("opencv.ml.boost_sample_count", &[model.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict_boost",
            &[model.arg(), float_list_arg(&query)],
        )?,
        "stale_handle",
    )?;
    expect_int(helper.call("opencv.ml.live_boost_count", &[])?, 0)
}

/// Trains one typed model handle through the generated helper.
fn train(
    helper: &mut NativeHelper,
    samples: &[f64],
    labels: &[i64],
    feature_count: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.ml.train_boost_classifier",
            &[
                float_list_arg(samples),
                int_list_arg(labels),
                int_arg(feature_count),
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
    expected: &[i64],
) -> Result<(), String> {
    let actual = expect_ints(helper.call(
        "opencv.ml.predict_boost",
        &[model.arg(), float_list_arg(samples)],
    )?)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "expected predictions {expected:?}, received {actual:?}"
        ))
    }
}

/// Requires malformed training to fail without allocating another model.
fn expect_train_invalid(
    helper: &mut NativeHelper,
    samples: &[f64],
    labels: &[i64],
    feature_count: i64,
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.ml.train_boost_classifier",
            &[
                float_list_arg(samples),
                int_list_arg(labels),
                int_arg(feature_count),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_int(helper.call("opencv.ml.live_boost_count", &[])?, 1)
}
