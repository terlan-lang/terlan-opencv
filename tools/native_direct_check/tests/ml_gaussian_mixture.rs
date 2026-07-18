//! Direct Gaussian-mixture tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

const TRAINING: [f64; 16] = [
    -1.0, -1.0, -1.0, -0.8, -0.8, -1.0, -0.8, -0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0,
];

/// Proves deterministic clustering, validation, and owner lifecycle behavior.
#[test]
fn trains_predicts_and_rejects_invalid_gaussian_mixture_inputs() {
    assert_eq!(ffi::live_gaussian_mixture_model_count(), 0);
    let model = train(&TRAINING, 2, 2);
    assert!(!model.is_null());
    assert_eq!(ffi::live_gaussian_mixture_model_count(), 1);
    let model_ref = model.as_ref().expect("trained model");
    assert_eq!(model_ref.sample_count(), 8);
    assert_eq!(model_ref.feature_count(), 2);
    assert_eq!(model_ref.cluster_count(), 2);
    let expected = predict_pair(model_ref, &[-0.9, -0.9, 0.9, 0.9]);
    let repeated = train(&TRAINING, 2, 2);
    assert_eq!(ffi::live_gaussian_mixture_model_count(), 2);
    assert_eq!(
        predict_pair(
            repeated.as_ref().expect("repeated model"),
            &[-0.9, -0.9, 0.9, 0.9]
        ),
        expected
    );
    drop(repeated);

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
    for (samples, features, clusters) in [
        (vec![], 2, 2),
        (TRAINING[..15].to_vec(), 2, 2),
        (TRAINING.to_vec(), 0, 2),
        (TRAINING.to_vec(), 4097, 2),
        (TRAINING.to_vec(), 2, 1),
        (TRAINING.to_vec(), 2, 65),
        (TRAINING[..8].to_vec(), 2, 3),
    ] {
        assert!(train(&samples, features, clusters).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1_000_001.0] {
        let mut samples = TRAINING;
        samples[0] = invalid;
        assert!(train(&samples, 2, 2).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_gaussian_mixture_model_count(), 1);
    drop(model);
    assert_eq!(ffi::live_gaussian_mixture_model_count(), 0);
}

/// Trains one model through the direct generated bridge.
fn train(
    samples: &[f64],
    feature_count: i64,
    cluster_count: i64,
) -> cxx::UniquePtr<ffi::OpenCvGaussianMixtureModel> {
    ffi::train_gaussian_mixture_owned(samples, feature_count, cluster_count)
}

/// Copies and validates one pair of distinct binary cluster indexes.
fn predict_pair(model: &ffi::OpenCvGaussianMixtureModel, samples: &[f64]) -> Vec<i64> {
    let actual: Vec<i64> = model
        .predict_owned(samples)
        .as_ref()
        .expect("predictions")
        .iter()
        .copied()
        .collect();
    assert!(actual == [0, 1] || actual == [1, 0]);
    actual
}
