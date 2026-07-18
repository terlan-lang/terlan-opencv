//! Direct Manhattan mask distance-transform tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact distances, nonbinary input, validation, immutability, and cleanup.
#[test]
fn manhattan_distance_transform_is_checked_and_immutable() {
    assert_eq!(ffi::live_mat_count(), 0);
    let mask_bytes = [
        0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 255, 5, 0, 0, 6, 7, 8, 0, 0, 0, 0, 0, 0,
    ];
    let mask = ffi::mat_from_u8(&mask_bytes, 5, 5, 1);
    let mask_ref = mask.as_ref().expect("distance mask");
    let distances = mask_ref.manhattan_distance_transform_owned();
    assert_eq!(
        mat_bytes(distances.as_ref().expect("distance transform")),
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,]
    );
    assert_eq!(mat_bytes(mask_ref), mask_bytes);

    let mut wide_mask = vec![255; 513];
    wide_mask[0] = 0;
    wide_mask[512] = 0;
    let wide = ffi::mat_from_u8(&wide_mask, 1, 513, 1);
    let wide_distances = wide
        .as_ref()
        .expect("wide mask")
        .manhattan_distance_transform_owned();
    let expected_wide: Vec<u8> = (0..513)
        .map(|column| column.min(512 - column).min(255) as u8)
        .collect();
    assert_eq!(
        mat_bytes(wide_distances.as_ref().expect("wide distances")),
        expected_wide
    );

    let zero = ffi::mat_zeros(3, 3, 1);
    let zero_distances = zero
        .as_ref()
        .expect("zero mask")
        .manhattan_distance_transform_owned();
    assert_eq!(
        mat_bytes(zero_distances.as_ref().expect("zero distances")),
        [0; 9]
    );
    let bgr = ffi::mat_zeros(5, 5, 3);
    let bgra = ffi::mat_zeros(5, 5, 4);
    for image in [&bgr, &bgra] {
        assert!(image
            .as_ref()
            .expect("invalid color mask")
            .manhattan_distance_transform_owned()
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(bgra);
    drop(bgr);
    drop(zero_distances);
    drop(zero);
    drop(wide_distances);
    drop(wide);
    drop(distances);
    drop(mask);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native matrix into a Rust byte vector for exact assertions.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
