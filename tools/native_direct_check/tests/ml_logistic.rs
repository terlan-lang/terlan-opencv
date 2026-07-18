//! Direct logistic-regression model tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

const TRAINING: [f64; 16] = [
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 10.0, 10.0, 10.0, 11.0, 11.0, 10.0, 11.0, 11.0,
];
const LABELS: [i64; 8] = [-3, -3, -3, -3, 7, 7, 7, 7];

/// Proves copied training, categorical prediction, validation, and owner lifecycle behavior.
#[test]
fn trains_predicts_and_rejects_invalid_logistic_inputs() {
    assert_eq!(ffi::live_logistic_model_count(), 0);
    let model = train(&TRAINING, &LABELS, 2);
    assert!(!model.is_null());
    assert_eq!(ffi::live_logistic_model_count(), 1);
    let model_ref = model.as_ref().expect("trained model");
    assert_eq!(model_ref.sample_count(), 8);
    assert_eq!(model_ref.feature_count(), 2);
    assert_eq!(predict(model_ref, &[0.2, 0.2, 10.2, 10.2]), [-3, 7]);
    assert_eq!(predict(model_ref, &[0.2, 0.2, 10.2, 10.2]), [-3, 7]);

    for samples in [
        vec![],
        vec![0.0],
        vec![f64::NAN, 0.0],
        vec![f64::INFINITY, 0.0],
        vec![1_000_001.0, 0.0],
    ] {
        assert!(model_ref.predict_owned(&samples).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (samples, labels, features) in [
        (vec![], vec![], 2),
        (TRAINING.to_vec(), LABELS[..7].to_vec(), 2),
        (TRAINING[..15].to_vec(), LABELS.to_vec(), 2),
        (TRAINING.to_vec(), LABELS.to_vec(), 0),
        (TRAINING.to_vec(), LABELS.to_vec(), 4097),
        (TRAINING.to_vec(), vec![1; 8], 2),
    ] {
        assert!(train(&samples, &labels, features).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut samples = TRAINING;
        samples[0] = invalid;
        assert!(train(&samples, &LABELS, 2).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid in [-1_000_001, 1_000_001, i64::MIN, i64::MAX] {
        let mut labels = LABELS;
        labels[0] = invalid;
        assert!(train(&TRAINING, &labels, 2).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_logistic_model_count(), 1);
    drop(model);
    assert_eq!(ffi::live_logistic_model_count(), 0);
}

/// Trains one model through the direct generated bridge.
fn train(
    samples: &[f64],
    labels: &[i64],
    feature_count: i64,
) -> cxx::UniquePtr<ffi::OpenCvLogisticRegressionModel> {
    ffi::train_logistic_classifier_owned(samples, labels, feature_count)
}

/// Copies one prediction result from a live model.
fn predict(model: &ffi::OpenCvLogisticRegressionModel, samples: &[f64]) -> Vec<i64> {
    model
        .predict_owned(samples)
        .as_ref()
        .expect("predictions")
        .iter()
        .copied()
        .collect()
}
