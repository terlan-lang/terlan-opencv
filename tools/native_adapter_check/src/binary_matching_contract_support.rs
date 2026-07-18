//! Shared generated-helper validation for binary descriptor matches.

use std::collections::BTreeSet;

use super::{expect_floats, expect_ints, NativeHandle, NativeHelper};

/// Names the generated projection operations for one match resource family.
pub(super) struct MatchOperations {
    /// Human-readable descriptor family used in failures.
    pub(super) family: &'static str,
    /// NativeBoundary operation that copies match indices.
    pub(super) indices: &'static str,
    /// NativeBoundary operation that copies match distances.
    pub(super) distances: &'static str,
    /// NativeBoundary operation that copies match coordinates.
    pub(super) coordinates: &'static str,
    /// Inclusive maximum Hamming distance for one descriptor row.
    pub(super) maximum_distance: i64,
}

/// Validates aligned unique indices, distances, and endpoint coordinates.
pub(super) fn validate_matches(
    helper: &mut NativeHelper,
    matches: &NativeHandle,
    operations: &MatchOperations,
    count: usize,
    query_count: i64,
    train_count: i64,
) -> Result<(), String> {
    let indices = expect_ints(helper.call(operations.indices, &[matches.arg()])?)?;
    let distances = expect_ints(helper.call(operations.distances, &[matches.arg()])?)?;
    let coordinates = expect_floats(helper.call(operations.coordinates, &[matches.arg()])?)?;
    if indices.len() != count * 2 || distances.len() != count || coordinates.len() != count * 4 {
        return Err(format!(
            "{} match projections are not row aligned",
            operations.family
        ));
    }
    let mut query_seen = BTreeSet::new();
    let mut train_seen = BTreeSet::new();
    let mut previous = None;
    for row in 0..count {
        let query = indices[row * 2];
        let train = indices[row * 2 + 1];
        let distance = distances[row];
        if !(0..query_count).contains(&query)
            || !(0..train_count).contains(&train)
            || !(0..=operations.maximum_distance).contains(&distance)
            || !query_seen.insert(query)
            || !train_seen.insert(train)
            || previous.is_some_and(|value| value > (distance, query, train))
            || coordinates[row * 4..row * 4 + 4]
                .iter()
                .any(|value| !value.is_finite() || !(0.0..160.0).contains(value))
        {
            return Err(format!("invalid {} match row {row}", operations.family));
        }
        previous = Some((distance, query, train));
    }
    Ok(())
}
