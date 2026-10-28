//! Direct behavioral tests for OpenCV core linear algebra.

use terlan_opencv_native::ffi;

/// Encodes row-major F64 values as native-endian matrix bytes.
fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// Decodes one owned F64 matrix into row-major values.
fn f64_values(matrix: &ffi::OpenCvMat) -> Vec<f64> {
    let bytes = matrix.to_bytes();
    let bytes = bytes.as_ref().expect("matrix bytes");
    bytes
        .as_slice()
        .chunks_exact(8)
        .map(|chunk| f64::from_ne_bytes(chunk.try_into().expect("eight-byte scalar")))
        .collect()
}

/// Constructs one F64 matrix from row-major values.
fn f64_matrix(values: &[f64], rows: i64, cols: i64) -> cxx::UniquePtr<ffi::OpenCvMat> {
    ffi::mat_from_bytes(&f64_bytes(values), rows, cols, 1, 6)
}

/// Requires two finite values to differ by no more than one tolerance.
fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "expected {expected} +/- {tolerance}, got {actual}"
    );
}

/// Requires one matrix operation to return the public invalid-argument status.
fn assert_invalid_matrix(operation: impl FnOnce() -> cxx::UniquePtr<ffi::OpenCvMat>) {
    let matrix = operation();
    assert!(matrix.is_null());
    assert_eq!(ffi::take_last_failure(), 1);
}

/// Verifies all-one and identity constructors preserve exact typed values.
#[test]
fn typed_value_constructors_create_exact_owned_matrices() {
    let ones = ffi::mat_ones_typed(2, 2, 3, 0);
    let ones = ones.as_ref().expect("three-channel ones");
    assert_eq!((ones.rows(), ones.cols(), ones.channels()), (2, 2, 3));
    assert_eq!(
        ones.to_bytes().as_ref().expect("ones bytes").as_slice(),
        &[1; 12]
    );

    let identity = ffi::mat_identity_typed(2, 3, 6);
    let identity = identity.as_ref().expect("rectangular identity");
    assert_eq!(f64_values(identity), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    assert!(!identity.is_submatrix());
    assert!(identity.is_contiguous());
}

/// Verifies seeded uniform generation is deterministic and range bounded.
#[test]
fn random_uniform_is_seeded_and_bounded() {
    let first = ffi::mat_random_uniform(4, 5, 6, -2.0, 3.0, 42);
    let second = ffi::mat_random_uniform(4, 5, 6, -2.0, 3.0, 42);
    let different = ffi::mat_random_uniform(4, 5, 6, -2.0, 3.0, 43);
    let first_values = f64_values(first.as_ref().expect("first random matrix"));
    let second_values = f64_values(second.as_ref().expect("second random matrix"));
    let different_values = f64_values(different.as_ref().expect("different random matrix"));

    assert_eq!(first_values, second_values);
    assert_ne!(first_values, different_values);
    assert!(first_values
        .iter()
        .all(|value| value.is_finite() && *value >= -2.0 && *value < 3.0));
}

/// Verifies products, inverses, determinants, norms, traces, and dot products.
#[test]
fn dense_matrix_algebra_matches_known_results() {
    let matrix = f64_matrix(&[4.0, 7.0, 2.0, 6.0], 2, 2);
    let matrix = matrix.as_ref().expect("coefficient matrix");
    let identity = ffi::mat_identity_typed(2, 2, 6);
    let identity = identity.as_ref().expect("identity matrix");

    assert_close(matrix.determinant_value(), 10.0, 1.0e-12);
    assert_close(matrix.dot_value(identity), 10.0, 1.0e-12);
    assert_close(matrix.norm_value(0), 19.0, 1.0e-12);
    assert_close(matrix.norm_value(1), 105.0_f64.sqrt(), 1.0e-12);
    assert_close(matrix.norm_value(2), 7.0, 1.0e-12);
    assert_eq!(
        matrix
            .trace_values_owned()
            .as_ref()
            .expect("trace")
            .as_slice(),
        &[10.0]
    );

    let inverse = matrix.matrix_inverse_owned(0);
    let inverse = inverse.as_ref().expect("LU inverse");
    let expected_inverse = [0.6, -0.7, -0.2, 0.4];
    for (actual, expected) in f64_values(inverse).iter().zip(expected_inverse) {
        assert_close(*actual, expected, 1.0e-12);
    }

    let product = matrix.matrix_multiply_owned(inverse);
    let product = product.as_ref().expect("inverse product");
    for (actual, expected) in f64_values(product).iter().zip([1.0, 0.0, 0.0, 1.0]) {
        assert_close(*actual, expected, 1.0e-12);
    }
}

/// Verifies square and least-squares solution paths return owned values.
#[test]
fn linear_system_solvers_cover_square_and_rectangular_systems() {
    let coefficients = f64_matrix(&[3.0, 1.0, 1.0, 2.0], 2, 2);
    let right_hand_side = f64_matrix(&[9.0, 8.0], 2, 1);
    let solution = coefficients
        .as_ref()
        .expect("coefficients")
        .matrix_solve_owned(right_hand_side.as_ref().expect("right-hand side"), 0);
    let solution = solution.as_ref().expect("LU solution");
    for (actual, expected) in f64_values(solution).iter().zip([2.0, 3.0]) {
        assert_close(*actual, expected, 1.0e-12);
    }

    let rectangular = f64_matrix(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2);
    let observed = f64_matrix(&[2.0, 3.0, 5.0], 3, 1);
    let least_squares = rectangular
        .as_ref()
        .expect("rectangular coefficients")
        .matrix_solve_owned(observed.as_ref().expect("observations"), 1);
    let least_squares = least_squares.as_ref().expect("SVD solution");
    for (actual, expected) in f64_values(least_squares).iter().zip([2.0, 3.0]) {
        assert_close(*actual, expected, 1.0e-12);
    }
}

/// Verifies every portable FileStorage format round-trips exact matrix state.
#[test]
fn in_memory_persistence_round_trips_every_format() {
    let source = f64_matrix(&[1.25, -2.5, 3.75, 4.5, 5.25, -6.0], 2, 3);
    let source = source.as_ref().expect("persistence source");

    for format in 0..=2 {
        let document = source.serialize_owned(format);
        let document = document.as_ref().expect("serialized matrix");
        let document = document.to_str().expect("UTF-8 OpenCV document");
        assert!(document.contains("matrix"));

        let restored = ffi::mat_deserialize(document, format);
        let restored = restored.as_ref().expect("restored matrix");
        assert_eq!((restored.rows(), restored.cols()), (2, 3));
        assert_eq!(restored.type_code(), 6);
        assert_eq!(f64_values(restored), f64_values(source));
    }
}

/// Verifies invalid domains fail deterministically without returning resources.
#[test]
fn invalid_linear_algebra_domains_are_rejected() {
    assert_invalid_matrix(|| ffi::mat_ones_typed(0, 1, 1, 6));
    assert_invalid_matrix(|| ffi::mat_ones_typed(1, 1, 513, 6));
    assert_invalid_matrix(|| ffi::mat_identity_typed(1, 1, 8));
    assert_invalid_matrix(|| ffi::mat_random_uniform(1, 1, 6, 1.0, 1.0, 0));
    assert_invalid_matrix(|| ffi::mat_random_uniform(1, 1, 6, f64::NAN, 1.0, 0));
    assert_invalid_matrix(|| ffi::mat_random_uniform(1, 1, 6, 0.0, 1.0, -1));

    let integer = ffi::mat_ones_typed(2, 2, 1, 0);
    let integer = integer.as_ref().expect("integer matrix");
    assert!(integer.matrix_inverse_owned(0).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert_eq!(integer.determinant_value(), 0.0);
    assert_eq!(ffi::take_last_failure(), 1);

    let singular = f64_matrix(&[1.0, 2.0, 2.0, 4.0], 2, 2);
    let singular = singular.as_ref().expect("singular matrix");
    assert!(singular.matrix_inverse_owned(0).is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let incompatible = f64_matrix(&[1.0, 2.0, 3.0], 1, 3);
    let incompatible = incompatible.as_ref().expect("incompatible matrix");
    assert!(singular.matrix_multiply_owned(incompatible).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert_eq!(singular.norm_value(i64::MAX), 0.0);
    assert_eq!(ffi::take_last_failure(), 1);

    assert!(singular.serialize_owned(i64::MAX).is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert_invalid_matrix(|| ffi::mat_deserialize("", 0));
    assert_invalid_matrix(|| ffi::mat_deserialize("not an OpenCV document", 0));
    assert_invalid_matrix(|| ffi::mat_deserialize("%YAML:1.0\nmatrix: 1", i64::MAX));
}
