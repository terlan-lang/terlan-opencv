//! Direct behavioral tests for typed OpenCV core array operations.

use terlan_opencv_native::ffi;

/// Encodes row-major F64 values as native-endian matrix bytes.
fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// Constructs one F64 matrix from row-major values.
fn f64_matrix(values: &[f64], rows: i64, cols: i64) -> cxx::UniquePtr<ffi::OpenCvMat> {
    ffi::mat_from_bytes(&f64_bytes(values), rows, cols, 1, 6)
}

/// Decodes one owned F64 matrix into row-major values.
fn f64_values(matrix: &ffi::OpenCvMat) -> Vec<f64> {
    matrix
        .to_bytes()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .chunks_exact(8)
        .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("eight-byte scalar")))
        .collect()
}

/// Decodes one owned I32 matrix into row-major values.
fn i32_values(matrix: &ffi::OpenCvMat) -> Vec<i32> {
    matrix
        .to_bytes()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("four-byte scalar")))
        .collect()
}

/// Requires two finite values to differ by no more than one tolerance.
fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "expected {expected} +/- {tolerance}, got {actual}"
    );
}

/// Requires an operation to fail with the stable invalid-argument status.
fn assert_invalid(operation: impl FnOnce() -> cxx::UniquePtr<ffi::OpenCvMat>) {
    assert!(operation().is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Verifies deterministic normal generation and generic channel statistics.
#[test]
fn random_normal_and_statistics_are_owned_and_finite() {
    let first = ffi::mat_random_normal(8, 8, 6, 2.0, 0.5, 27);
    let second = ffi::mat_random_normal(8, 8, 6, 2.0, 0.5, 27);
    let first = first.as_ref().expect("first normal matrix");
    let second = second.as_ref().expect("second normal matrix");
    assert_eq!(f64_values(first), f64_values(second));
    assert!(f64_values(first).iter().all(|value| value.is_finite()));

    let values = f64_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let values = values.as_ref().expect("statistics matrix");
    assert_eq!(
        values.sum_values_owned().as_ref().expect("sum").as_slice(),
        &[10.0]
    );
    assert_eq!(
        values
            .mean_values_owned()
            .as_ref()
            .expect("mean")
            .as_slice(),
        &[2.5]
    );
    let deviations = values.standard_deviation_values_owned();
    assert_close(
        deviations.as_ref().expect("deviation").as_slice()[0],
        1.25_f64.sqrt(),
        1.0e-12,
    );
}

/// Verifies symbolic reductions preserve the expected axis and scalar order.
#[test]
fn reductions_cover_every_operation_and_dimension() {
    let source = f64_matrix(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], 2, 3);
    let source = source.as_ref().expect("reduction source");

    let row_sums = source.reduce_owned(0, 0);
    assert_eq!(
        f64_values(row_sums.as_ref().expect("row sums")),
        vec![9.0, 5.0, 7.0]
    );
    let column_average = source.reduce_owned(1, 1);
    assert_eq!(
        f64_values(column_average.as_ref().expect("column average")),
        vec![2.0, 5.0]
    );
    let row_maximum = source.reduce_owned(0, 2);
    assert_eq!(
        f64_values(row_maximum.as_ref().expect("row maximum")),
        vec![6.0, 4.0, 5.0]
    );
    let column_minimum = source.reduce_owned(1, 3);
    assert_eq!(
        f64_values(column_minimum.as_ref().expect("column minimum")),
        vec![1.0, 4.0]
    );
}

/// Verifies sorting values and source indexes in both supported directions.
#[test]
fn sorting_values_and_indices_is_deterministic() {
    let source = f64_matrix(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], 2, 3);
    let source = source.as_ref().expect("sort source");

    let ascending = source.sort_owned(0, 0);
    assert_eq!(
        f64_values(ascending.as_ref().expect("ascending rows")),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let indexes = source.sort_indices_owned(0, 0);
    assert_eq!(
        i32_values(indexes.as_ref().expect("ascending indexes")),
        vec![1, 2, 0, 1, 2, 0]
    );
    let descending_columns = source.sort_owned(1, 1);
    assert_eq!(
        f64_values(descending_columns.as_ref().expect("descending columns")),
        vec![6.0, 4.0, 5.0, 3.0, 1.0, 2.0]
    );
}

/// Verifies norm normalization and finite element-wise numeric transforms.
#[test]
fn normalization_and_numeric_transforms_match_known_values() {
    let identity = ffi::mat_identity_typed(2, 2, 6);
    let identity = identity.as_ref().expect("identity");
    let normalized = identity.normalize_norm_owned(1.0, 1);
    let normalized = normalized.as_ref().expect("normalized identity");
    assert_close(normalized.norm_value(1), 1.0, 1.0e-12);

    let positive = f64_matrix(&[1.0, 4.0, 9.0, 16.0], 2, 2);
    let positive = positive.as_ref().expect("positive values");
    let roots = positive.square_root_owned();
    assert_eq!(
        f64_values(roots.as_ref().expect("square roots")),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    let squares = roots.as_ref().expect("roots").power_owned(2.0);
    assert_eq!(
        f64_values(squares.as_ref().expect("squares")),
        vec![1.0, 4.0, 9.0, 16.0]
    );

    let logarithms = positive.logarithm_owned();
    let restored = logarithms.as_ref().expect("logarithms").exponential_owned();
    for (actual, expected) in f64_values(restored.as_ref().expect("restored"))
        .iter()
        .zip([1.0, 4.0, 9.0, 16.0])
    {
        assert_close(*actual, expected, 1.0e-12);
    }
}

/// Verifies rejected domains never leak partially constructed matrix owners.
#[test]
fn invalid_array_domains_fail_with_stable_errors() {
    assert_invalid(|| ffi::mat_random_normal(1, 1, 6, 0.0, 0.0, 1));
    assert_invalid(|| ffi::mat_random_normal(1, 1, 6, 0.0, 1.0, -1));
    assert_invalid(|| ffi::mat_random_normal(1, 1, 0, 0.0, 1.0, 1));

    let source = f64_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let source = source.as_ref().expect("valid source");
    assert_invalid(|| source.reduce_owned(i64::MAX, 0));
    assert_invalid(|| source.reduce_owned(0, i64::MAX));
    assert_invalid(|| source.sort_owned(i64::MAX, 0));
    assert_invalid(|| source.sort_indices_owned(0, i64::MAX));
    assert_invalid(|| source.normalize_norm_owned(0.0, 1));
    assert_invalid(|| source.normalize_norm_owned(1.0, i64::MAX));
    assert_invalid(|| source.power_owned(f64::NAN));

    let nonpositive = f64_matrix(&[0.0, -1.0], 1, 2);
    let nonpositive = nonpositive.as_ref().expect("nonpositive source");
    assert_invalid(|| nonpositive.logarithm_owned());
    assert_invalid(|| nonpositive.square_root_owned());

    let multichannel = ffi::mat_ones_typed(1, 2, 3, 6);
    let multichannel = multichannel.as_ref().expect("multichannel source");
    assert_invalid(|| multichannel.sort_owned(0, 0));
}
