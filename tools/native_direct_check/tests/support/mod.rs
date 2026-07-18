//! Shared deterministic fixtures for direct native integration tests.

use std::collections::BTreeSet;

/// Minimal ONNX Identity graph with one fixed `[1, 3, 2, 2]` float tensor.
pub(crate) const IDENTITY_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/identity_rgb.onnx");

/// Minimal ONNX Identity graph with one fixed `[1, 1, 2, 2]` float tensor.
pub(crate) const GRAYSCALE_IDENTITY_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/identity_grayscale.onnx");

/// Minimal ONNX Flatten graph from `[1, 3, 2, 2]` to `[1, 12]` floats.
pub(crate) const FLATTEN_ONNX: &[u8] = &[
    8, 8, 18, 18, 116, 101, 114, 108, 97, 110, 45, 111, 112, 101, 110, 99, 118, 45, 116, 101, 115,
    116, 58, 109, 10, 33, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111, 117, 116, 112, 117, 116, 26,
    7, 102, 108, 97, 116, 116, 101, 110, 34, 7, 70, 108, 97, 116, 116, 101, 110, 18, 13, 102, 108,
    97, 116, 116, 101, 110, 95, 103, 114, 97, 112, 104, 90, 31, 10, 5, 105, 110, 112, 117, 116, 18,
    22, 10, 20, 8, 1, 18, 16, 10, 2, 8, 1, 10, 2, 8, 3, 10, 2, 8, 2, 10, 2, 8, 2, 98, 24, 10, 6,
    111, 117, 116, 112, 117, 116, 18, 14, 10, 12, 8, 1, 18, 8, 10, 2, 8, 1, 10, 2, 8, 12, 66, 2,
    16, 13,
];

/// Protobuf-generated Reshape graph from `[1, 3, 1, 7]` to `[1, 1, 3, 7]`.
pub(crate) const SSD_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/ssd_reshape.onnx");

/// Protobuf-generated Reshape graph from `[1, 3, 1, 6]` to `[1, 3, 6]`.
pub(crate) const XYXY_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/xyxy_reshape.onnx");

/// Protobuf-generated Reshape graph from `[1, 3, 1, 7]` to `[1, 3, 7]`.
pub(crate) const OBJECTNESS_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/objectness_reshape.onnx");

/// Protobuf-generated Reshape graph from `[1, 3, 1, 6]` to `[1, 6, 3]`.
pub(crate) const TRANSPOSED_CLASS_SCORE_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/transposed_class_score_reshape.onnx");

/// Protobuf-generated Reshape graph from `[1, 3, 1, 7]` to `[1, 7, 3]`.
pub(crate) const TRANSPOSED_OBJECTNESS_RESHAPE_ONNX: &[u8] =
    include_bytes!("../../../../tests/fixtures/transposed_objectness_reshape.onnx");

/// Seven-by-seven grayscale image containing four strong corners.
#[rustfmt::skip]
pub(crate) const CORNER_IMAGE: &[u8] = &[
    0, 0,   0,   0,   0, 0, 0,
    0, 0,   0,   0,   0, 0, 0,
    0, 0, 255, 255, 255, 0, 0,
    0, 0, 255, 255, 255, 0, 0,
    0, 0, 255, 255, 255, 0, 0,
    0, 0,   0,   0,   0, 0, 0,
    0, 0,   0,   0,   0, 0, 0,
];

/// Builds a high-contrast checkerboard with interior multi-scale corners.
pub(crate) fn checkerboard_fixture() -> Vec<u8> {
    let mut image = vec![0; 160 * 160];
    for row in 0..160 {
        for col in 0..160 {
            if (row / 16 + col / 16) % 2 == 0 {
                image[row * 160 + col] = 255;
            }
        }
    }
    image
}

/// Builds isolated high-contrast points that satisfy FAST-family arc tests.
pub(crate) fn detector_fixture() -> Vec<u8> {
    let mut image = vec![0; 32 * 32];
    for (row, column) in [(8, 8), (8, 23), (23, 8), (23, 23)] {
        image[row * 32 + column] = 255;
    }
    image
}

/// Validates bounded unique detector coordinates in canonical row-major order.
pub(crate) fn validate_detector_points(
    family: &str,
    points: &[f64],
    minimum: usize,
    maximum: usize,
) {
    assert_eq!(points.len() % 2, 0, "{family} coordinate alignment");
    assert!(points.len() >= minimum * 2, "{family} minimum count");
    assert!(points.len() <= maximum * 2, "{family} maximum count");
    let mut previous = None;
    for point in points.chunks_exact(2) {
        assert!(point[0].is_finite() && point[1].is_finite());
        assert!((0.0..32.0).contains(&point[0]));
        assert!((0.0..32.0).contains(&point[1]));
        let current = (point[1], point[0]);
        assert!(previous.is_none_or(|value| value < current));
        previous = Some(current);
    }
}

/// Validates finite, bounded, row-major feature-coordinate pairs.
pub(crate) fn validate_feature_points(family: &str, points: &[f64], count: usize) {
    assert_eq!(points.len(), count * 2, "{family} coordinate alignment");
    let mut previous = None;
    for point in points.chunks_exact(2) {
        assert!(point[0].is_finite() && point[1].is_finite());
        assert!((0.0..160.0).contains(&point[0]));
        assert!((0.0..160.0).contains(&point[1]));
        let current = (point[1], point[0]);
        assert!(previous.is_none_or(|value| value <= current));
        previous = Some(current);
    }
}

/// Validates copied binary-match projections and canonical ordering.
pub(crate) fn validate_binary_matches(
    family: &str,
    indices: &[i64],
    distances: &[i64],
    coordinates: &[f64],
    count: usize,
    query_count: i64,
    train_count: i64,
    maximum_distance: i64,
) {
    assert_eq!(indices.len(), count * 2, "{family} index alignment");
    assert_eq!(distances.len(), count, "{family} distance alignment");
    assert_eq!(
        coordinates.len(),
        count * 4,
        "{family} coordinate alignment"
    );
    let mut query_seen = BTreeSet::new();
    let mut train_seen = BTreeSet::new();
    let mut previous = None;
    for row in 0..count {
        let current = (distances[row], indices[row * 2], indices[row * 2 + 1]);
        assert!((0..query_count).contains(&current.1));
        assert!((0..train_count).contains(&current.2));
        assert!((0..=maximum_distance).contains(&current.0));
        assert!(query_seen.insert(current.1));
        assert!(train_seen.insert(current.2));
        assert!(previous.is_none_or(|value| value <= current));
        assert!(coordinates[row * 4..row * 4 + 4]
            .iter()
            .all(|value| value.is_finite() && (0.0..160.0).contains(value)));
        previous = Some(current);
    }
}

/// Validates copied floating-point match projections and canonical ordering.
pub(crate) fn validate_float_matches(
    family: &str,
    indices: &[i64],
    distances: &[f64],
    coordinates: &[f64],
    count: usize,
    query_count: i64,
    train_count: i64,
) {
    assert_eq!(indices.len(), count * 2, "{family} index alignment");
    assert_eq!(distances.len(), count, "{family} distance alignment");
    assert_eq!(
        coordinates.len(),
        count * 4,
        "{family} coordinate alignment"
    );
    let mut query_seen = BTreeSet::new();
    let mut train_seen = BTreeSet::new();
    let mut previous: Option<(f64, i64, i64)> = None;
    for row in 0..count {
        let current = (distances[row], indices[row * 2], indices[row * 2 + 1]);
        assert!((0..query_count).contains(&current.1));
        assert!((0..train_count).contains(&current.2));
        assert!(current.0.is_finite() && current.0 >= 0.0);
        assert!(query_seen.insert(current.1));
        assert!(train_seen.insert(current.2));
        assert!(previous.is_none_or(|value| {
            current.0 > value.0
                || (current.0 == value.0
                    && (current.1 > value.1 || (current.1 == value.1 && current.2 >= value.2)))
        }));
        assert!(coordinates[row * 4..row * 4 + 4]
            .iter()
            .all(|value| value.is_finite() && (0.0..160.0).contains(value)));
        previous = Some(current);
    }
}
