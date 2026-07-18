//! Bounded deterministic Gaussian-mixture ownership contract.

use super::{
    expect_error_code, expect_int, expect_ints, expect_typed_handle, expect_unit, float_list_arg,
    int_arg, NativeHandle, NativeHelper,
};

const MODEL_TYPE: &str = "opencv.Ml.GaussianMixtureModel";
const TRAINING: [f64; 16] = [
    -1.0, -1.0, -1.0, -0.8, -0.8, -1.0, -0.8, -0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0,
];

/// Runs deterministic clustering, lifecycle, malformed-input, and bound checks.
pub(super) fn run(helper: &mut NativeHelper) -> Result<(), String> {
    expect_int(
        helper.call("opencv.ml.live_gaussian_mixture_count", &[])?,
        0,
    )?;
    let model = train(helper, &TRAINING, 2, 2)?;
    expect_int(
        helper.call("opencv.ml.live_gaussian_mixture_count", &[])?,
        1,
    )?;
    expect_int(
        helper.call("opencv.ml.gaussian_mixture_sample_count", &[model.arg()])?,
        8,
    )?;
    expect_int(
        helper.call("opencv.ml.gaussian_mixture_feature_count", &[model.arg()])?,
        2,
    )?;
    expect_int(
        helper.call("opencv.ml.gaussian_mixture_cluster_count", &[model.arg()])?,
        2,
    )?;

    let query = [-0.9, -0.9, 0.9, 0.9];
    let expected = predict_pair(helper, &model, &query)?;
    let repeated_model = train(helper, &TRAINING, 2, 2)?;
    expect_int(
        helper.call("opencv.ml.live_gaussian_mixture_count", &[])?,
        2,
    )?;
    let repeated = predict_pair(helper, &repeated_model, &query)?;
    if repeated != expected {
        return Err(format!(
            "deterministic EM predictions drifted: {expected:?} != {repeated:?}"
        ));
    }
    expect_unit(helper.call(
        "opencv.ml.dispose_gaussian_mixture",
        &[repeated_model.arg()],
    )?)?;
    expect_int(
        helper.call("opencv.ml.live_gaussian_mixture_count", &[])?,
        1,
    )?;

    expect_error_code(
        helper.call(
            "opencv.ml.predict_gaussian_clusters",
            &[
                model.arg_with_owner("foreign-worker"),
                float_list_arg(&query),
            ],
        )?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict_gaussian_clusters",
            &[
                model.arg_with_type("opencv.Ml.MlpClassifierModel"),
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
                "opencv.ml.predict_gaussian_clusters",
                &[model.arg(), float_list_arg(&samples)],
            )?,
            "opencv.invalid_argument",
        )?;
    }

    for (samples, features, clusters) in [
        (vec![], 2, 2),
        (TRAINING[..15].to_vec(), 2, 2),
        (TRAINING.to_vec(), 0, 2),
        (TRAINING.to_vec(), 4097, 2),
        (TRAINING.to_vec(), 2, 1),
        (TRAINING.to_vec(), 2, 65),
        (TRAINING[..8].to_vec(), 2, 3),
    ] {
        expect_train_invalid(helper, &samples, features, clusters)?;
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut samples = TRAINING;
        samples[0] = invalid;
        expect_train_invalid(helper, &samples, 2, 2)?;
    }
    expect_error_code(
        helper.call(
            "opencv.ml.train_gaussian_mixture",
            &["li:0,0,1,1".to_owned(), int_arg(2), int_arg(2)],
        )?,
        "invalid_arguments",
    )?;

    expect_unit(helper.call("opencv.ml.dispose_gaussian_mixture", &[model.arg()])?)?;
    expect_error_code(
        helper.call("opencv.ml.gaussian_mixture_sample_count", &[model.arg()])?,
        "stale_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.ml.predict_gaussian_clusters",
            &[model.arg(), float_list_arg(&query)],
        )?,
        "stale_handle",
    )?;
    expect_int(
        helper.call("opencv.ml.live_gaussian_mixture_count", &[])?,
        0,
    )
}

/// Trains one typed Gaussian-mixture handle through the generated helper.
fn train(
    helper: &mut NativeHelper,
    samples: &[f64],
    feature_count: i64,
    cluster_count: i64,
) -> Result<NativeHandle, String> {
    expect_typed_handle(
        helper.call(
            "opencv.ml.train_gaussian_mixture",
            &[
                float_list_arg(samples),
                int_arg(feature_count),
                int_arg(cluster_count),
            ],
        )?,
        MODEL_TYPE,
    )
}

/// Copies and validates one pair of distinct binary cluster indexes.
fn predict_pair(
    helper: &mut NativeHelper,
    model: &NativeHandle,
    samples: &[f64],
) -> Result<Vec<i64>, String> {
    let actual = expect_ints(helper.call(
        "opencv.ml.predict_gaussian_clusters",
        &[model.arg(), float_list_arg(samples)],
    )?)?;
    if actual == [0, 1] || actual == [1, 0] {
        Ok(actual)
    } else {
        Err(format!(
            "expected two distinct cluster indexes, received {actual:?}"
        ))
    }
}

/// Requires malformed training to fail without allocating another model.
fn expect_train_invalid(
    helper: &mut NativeHelper,
    samples: &[f64],
    feature_count: i64,
    cluster_count: i64,
) -> Result<(), String> {
    expect_error_code(
        helper.call(
            "opencv.ml.train_gaussian_mixture",
            &[
                float_list_arg(samples),
                int_arg(feature_count),
                int_arg(cluster_count),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_int(
        helper.call("opencv.ml.live_gaussian_mixture_count", &[])?,
        1,
    )
}
