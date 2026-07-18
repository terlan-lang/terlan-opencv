//! Direct SimpleBlobDetector tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves bright/dark detection, canonical rows, and source independence.
#[test]
fn simple_blob_keypoints_are_canonical_copied_values() {
    assert_eq!(ffi::live_mat_count(), 0);
    let fixture = blob_fixture(false);
    let image = ffi::mat_from_u8(&fixture, 160, 160, 1);
    let image_ref = image.as_ref().expect("bright blob image");
    let bright = floats(image_ref.simple_blob_keypoints_owned(255, 50, 1000, 5.0, 64));
    validate_blobs(&bright, 2);
    assert_eq!(
        floats(image_ref.simple_blob_keypoints_owned(255, 50, 1000, 5.0, 64)),
        bright
    );
    let truncated = floats(image_ref.simple_blob_keypoints_owned(255, 50, 1000, 5.0, 1));
    validate_blobs(&truncated, 1);
    assert_eq!(matrix_bytes(image_ref), fixture);

    let inverted_fixture = blob_fixture(true);
    let inverted = ffi::mat_from_u8(&inverted_fixture, 160, 160, 1);
    validate_blobs(
        &floats(
            inverted
                .as_ref()
                .expect("dark blob image")
                .simple_blob_keypoints_owned(0, 50, 1000, 5.0, 64),
        ),
        2,
    );
    drop(inverted);
    drop(image);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Proves empty normalization and bounded invalid-input rejection.
#[test]
fn simple_blob_keypoints_reject_invalid_sources_and_parameters() {
    let image = ffi::mat_zeros(160, 160, 1);
    let image_ref = image.as_ref().expect("uniform blob image");
    assert!(floats(image_ref.simple_blob_keypoints_owned(255, 50, 1000, 5.0, 8)).is_empty());
    for (value, minimum_area, maximum_area, distance, maximum) in [
        (1, 50, 1000, 5.0, 8),
        (255, 0, 1000, 5.0, 8),
        (255, 50, 50, 5.0, 8),
        (255, 50, 25602, 5.0, 8),
        (255, 50, 1000, 0.0, 8),
        (255, 50, 1000, -1.0, 8),
        (255, 50, 1000, f64::NAN, 8),
        (255, 50, 1000, f64::INFINITY, 8),
        (255, 50, 1000, 227.0, 8),
        (255, 50, 1000, 5.0, 0),
        (255, 50, 1000, 5.0, 4097),
    ] {
        assert!(image_ref
            .simple_blob_keypoints_owned(value, minimum_area, maximum_area, distance, maximum)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(3, 3, 3);
    assert!(color
        .as_ref()
        .expect("color blob image")
        .simple_blob_keypoints_owned(255, 1, 4, 1.0, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let too_small = ffi::mat_zeros(2, 2, 1);
    assert!(too_small
        .as_ref()
        .expect("small blob image")
        .simple_blob_keypoints_owned(255, 1, 4, 1.0, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized blob image")
        .simple_blob_keypoints_owned(255, 1, 4, 1.0, 1)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    drop(oversized);
    drop(too_small);
    drop(color);
    drop(image);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Builds two separated circular grayscale blobs.
fn blob_fixture(inverted: bool) -> Vec<u8> {
    let background = if inverted { 255 } else { 0 };
    let foreground = if inverted { 0 } else { 255 };
    let mut pixels = vec![background; 160 * 160];
    for (center_x, center_y, radius) in [(40_i64, 50_i64, 8_i64), (110, 100, 12)] {
        for y in center_y - radius..=center_y + radius {
            for x in center_x - radius..=center_x + radius {
                if (x - center_x).pow(2) + (y - center_y).pow(2) <= radius.pow(2) {
                    pixels[y as usize * 160 + x as usize] = foreground;
                }
            }
        }
    }
    pixels
}

/// Validates canonical x/y/diameter/response rows.
fn validate_blobs(values: &[f64], count: usize) {
    assert_eq!(values.len(), count * 4);
    let mut previous: Option<(f64, f64)> = None;
    for item in values.chunks_exact(4) {
        let order = (item[1], item[0]);
        assert!(item.iter().all(|value| value.is_finite()));
        assert!((0.0..160.0).contains(&item[0]));
        assert!((0.0..160.0).contains(&item[1]));
        assert!(item[2] > 0.0 && item[2] <= 453.0);
        assert!(item[3] >= 0.0);
        assert!(previous.is_none_or(|prior| order > prior));
        previous = Some(order);
    }
}

/// Copies one native floating-point vector.
fn floats(values: cxx::UniquePtr<cxx::CxxVector<f64>>) -> Vec<f64> {
    values.as_ref().expect("blob values").as_slice().to_vec()
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
