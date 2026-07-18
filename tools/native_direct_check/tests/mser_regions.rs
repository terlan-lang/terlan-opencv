//! Direct owned MSER region tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves canonical aligned projections, truncation, and source independence.
#[test]
fn mser_regions_preserve_canonical_owned_geometry() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_mser_region_count(), 0);
    let fixture = nested_regions_fixture();
    let image = ffi::mat_from_u8(&fixture, 160, 160, 1);
    let image_ref = image.as_ref().expect("MSER fixture");
    let first = image_ref.mser_regions_owned(5, 16, 4096, 64);
    let second = image_ref.mser_regions_owned(5, 16, 4096, 64);
    let first_ref = first.as_ref().expect("MSER analysis");
    let second_ref = second.as_ref().expect("repeated MSER analysis");
    let first_projection = projection(first_ref);
    validate_projection(&first_projection, 64);
    assert_eq!(projection(second_ref), first_projection);
    assert_eq!(matrix_bytes(image_ref), fixture);

    let truncated = image_ref.mser_regions_owned(5, 16, 4096, 1);
    let truncated_projection = projection(truncated.as_ref().expect("truncated MSER analysis"));
    assert_eq!(truncated_projection.0, 1);
    validate_projection(&truncated_projection, 1);

    drop(image);
    assert_eq!(projection(first_ref), first_projection);
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_mser_region_count(), 3);
    drop(truncated);
    drop(second);
    drop(first);
    assert_eq!(ffi::live_mser_region_count(), 0);
}

/// Proves empty normalization and bounded request/input rejection.
#[test]
fn mser_regions_reject_invalid_sources_and_limits() {
    let empty_image = ffi::mat_zeros(160, 160, 1);
    let empty_ref = empty_image.as_ref().expect("uniform MSER image");
    let empty = empty_ref.mser_regions_owned(5, 16, 4096, 8);
    assert_eq!(
        projection(empty.as_ref().expect("empty MSER analysis")),
        (0, vec![], vec![0], vec![])
    );
    for arguments in [
        [0, 16, 4096, 8],
        [256, 16, 4096, 8],
        [5, 0, 4096, 8],
        [5, 17, 16, 8],
        [5, 16, 25601, 8],
        [5, 16, 4096, 0],
        [5, 16, 4096, 4097],
        [5, 16, 4096, 257],
    ] {
        assert!(empty_ref
            .mser_regions_owned(arguments[0], arguments[1], arguments[2], arguments[3])
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(3, 3, 3);
    assert!(color
        .as_ref()
        .expect("color MSER image")
        .mser_regions_owned(5, 1, 4, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let too_small = ffi::mat_zeros(2, 2, 1);
    assert!(too_small
        .as_ref()
        .expect("small MSER image")
        .mser_regions_owned(5, 1, 4, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized MSER image")
        .mser_regions_owned(5, 1, 4, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(oversized);
    drop(too_small);
    drop(color);
    drop(empty);
    drop(empty_image);
    assert_eq!(ffi::live_mser_region_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Creates nested stable grayscale regions with multiple intensity plateaus.
fn nested_regions_fixture() -> Vec<u8> {
    let mut pixels = vec![0; 160 * 160];
    for (start, end, value) in [(20, 140, 64), (40, 120, 128), (60, 100, 255)] {
        for y in start..end {
            for x in start..end {
                pixels[y * 160 + x] = value;
            }
        }
    }
    pixels
}

/// Copies one complete MSER projection into Rust ownership.
fn projection(regions: &ffi::OpenCvMserRegions) -> (i64, Vec<i64>, Vec<i64>, Vec<i64>) {
    (
        regions.count(),
        integers(regions.points_owned()),
        integers(regions.offsets_owned()),
        integers(regions.bounds_owned()),
    )
}

/// Validates aligned canonical region points, offsets, and exact bounds.
fn validate_projection(projection: &(i64, Vec<i64>, Vec<i64>, Vec<i64>), maximum: i64) {
    let (count, points, offsets, bounds) = projection;
    assert!((1..=maximum).contains(count));
    assert_eq!(offsets.len(), *count as usize + 1);
    assert_eq!(offsets[0], 0);
    assert_eq!(
        points.len(),
        *offsets.last().expect("last offset") as usize * 2
    );
    assert_eq!(bounds.len(), *count as usize * 4);
    let mut previous_region: Option<(i64, i64, i64, i64, i64)> = None;
    for item in 0..*count as usize {
        let start = offsets[item] as usize;
        let end = offsets[item + 1] as usize;
        let bound = &bounds[item * 4..item * 4 + 4];
        let size = (end - start) as i64;
        let order = (bound[1], bound[0], bound[3], bound[2], size);
        assert!((16..=4096).contains(&size));
        assert!(previous_region.is_none_or(|previous| order >= previous));
        let mut minimum_x = 160;
        let mut minimum_y = 160;
        let mut maximum_x = -1;
        let mut maximum_y = -1;
        let mut previous_point: Option<(i64, i64)> = None;
        for point in start..end {
            let x = points[point * 2];
            let y = points[point * 2 + 1];
            assert!((0..160).contains(&x) && (0..160).contains(&y));
            assert!(previous_point.is_none_or(|previous| (y, x) > previous));
            minimum_x = minimum_x.min(x);
            minimum_y = minimum_y.min(y);
            maximum_x = maximum_x.max(x);
            maximum_y = maximum_y.max(y);
            previous_point = Some((y, x));
        }
        assert_eq!(
            [
                minimum_x,
                minimum_y,
                maximum_x - minimum_x + 1,
                maximum_y - minimum_y + 1
            ],
            bound
        );
        previous_region = Some(order);
    }
}

/// Copies one native integer vector.
fn integers(values: cxx::UniquePtr<cxx::CxxVector<i64>>) -> Vec<i64> {
    values.as_ref().expect("MSER integers").as_slice().to_vec()
}

/// Copies a live matrix into Rust-owned bytes.
fn matrix_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
