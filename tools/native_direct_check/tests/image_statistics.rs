//! Direct copied image-statistics tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves channel ordering, statistics, source integrity, and no new resource.
#[test]
fn copied_channel_means_work_for_every_public_layout() {
    assert_eq!(ffi::live_mat_count(), 0);

    let bgr_bytes = [0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0];
    let bgr = ffi::mat_from_u8(&bgr_bytes, 2, 2, 3);
    let bgr_ref = bgr.as_ref().expect("BGR image");
    assert_eq!(
        bgr_ref
            .mean_channels_owned()
            .as_ref()
            .expect("BGR channel means")
            .as_slice(),
        [63.75, 127.5, 127.5]
    );
    assert_eq!(ffi::live_mat_count(), 1);
    assert_close(
        bgr_ref
            .stddev_channels_owned()
            .as_ref()
            .expect("BGR channel standard deviations")
            .as_slice(),
        &[110.418_238_982_515_93, 127.5, 127.5],
    );
    assert_eq!(
        bgr_ref
            .non_zero_counts_owned()
            .as_ref()
            .expect("BGR nonzero counts")
            .as_slice(),
        [1, 2, 2]
    );
    assert_eq!(
        bgr_ref
            .sum_channels_owned()
            .as_ref()
            .expect("BGR channel sums")
            .as_slice(),
        [255, 510, 510]
    );
    assert_eq!(
        bgr_ref
            .to_u8()
            .as_ref()
            .expect("unchanged BGR bytes")
            .as_slice(),
        bgr_bytes
    );

    let gray_bytes = [0, 255, 76, 150];
    let gray = ffi::mat_from_u8(&gray_bytes, 2, 2, 1);
    let gray_ref = gray.as_ref().expect("grayscale image");
    assert_eq!(
        gray_ref
            .mean_channels_owned()
            .as_ref()
            .expect("grayscale mean")
            .as_slice(),
        [120.25]
    );
    assert_close(
        gray_ref
            .stddev_channels_owned()
            .as_ref()
            .expect("grayscale standard deviation")
            .as_slice(),
        &[94.155_124_661_379_96],
    );
    assert_eq!(
        gray_ref
            .min_max_values_owned()
            .as_ref()
            .expect("grayscale minimum and maximum")
            .as_slice(),
        [0.0, 255.0]
    );
    assert_eq!(
        gray_ref
            .non_zero_counts_owned()
            .as_ref()
            .expect("grayscale nonzero count")
            .as_slice(),
        [3]
    );
    assert_eq!(
        gray_ref
            .sum_channels_owned()
            .as_ref()
            .expect("grayscale sum")
            .as_slice(),
        [481]
    );
    assert_eq!(
        gray_ref
            .to_u8()
            .as_ref()
            .expect("unchanged grayscale bytes")
            .as_slice(),
        gray_bytes
    );

    let bgra_bytes = [0, 10, 20, 30, 100, 110, 120, 130];
    let bgra = ffi::mat_from_u8(&bgra_bytes, 1, 2, 4);
    assert_eq!(
        bgra.as_ref()
            .expect("BGRA image")
            .mean_channels_owned()
            .as_ref()
            .expect("BGRA channel means")
            .as_slice(),
        [50.0, 60.0, 70.0, 80.0]
    );
    assert_close(
        bgra.as_ref()
            .expect("BGRA image")
            .stddev_channels_owned()
            .as_ref()
            .expect("BGRA channel standard deviations")
            .as_slice(),
        &[50.0, 50.0, 50.0, 50.0],
    );
    assert_eq!(
        bgra.as_ref()
            .expect("BGRA image")
            .non_zero_counts_owned()
            .as_ref()
            .expect("BGRA nonzero counts")
            .as_slice(),
        [1, 2, 2, 2]
    );
    assert_eq!(
        bgra.as_ref()
            .expect("BGRA image")
            .sum_channels_owned()
            .as_ref()
            .expect("BGRA channel sums")
            .as_slice(),
        [100, 120, 140, 160]
    );

    let zeros = ffi::mat_zeros(2, 3, 3);
    assert_eq!(
        zeros
            .as_ref()
            .expect("zero BGR image")
            .mean_channels_owned()
            .as_ref()
            .expect("zero channel means")
            .as_slice(),
        [0.0, 0.0, 0.0]
    );
    assert_close(
        zeros
            .as_ref()
            .expect("zero BGR image")
            .stddev_channels_owned()
            .as_ref()
            .expect("zero channel standard deviations")
            .as_slice(),
        &[0.0, 0.0, 0.0],
    );
    assert_eq!(
        zeros
            .as_ref()
            .expect("zero BGR image")
            .non_zero_counts_owned()
            .as_ref()
            .expect("zero nonzero counts")
            .as_slice(),
        [0, 0, 0]
    );
    assert_eq!(
        zeros
            .as_ref()
            .expect("zero BGR image")
            .sum_channels_owned()
            .as_ref()
            .expect("zero channel sums")
            .as_slice(),
        [0, 0, 0]
    );
    assert!(bgr_ref.min_max_values_owned().is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let gray_zeros = ffi::mat_zeros(2, 3, 1);
    assert_eq!(
        gray_zeros
            .as_ref()
            .expect("zero grayscale image")
            .min_max_values_owned()
            .as_ref()
            .expect("zero grayscale minimum and maximum")
            .as_slice(),
        [0.0, 0.0]
    );
    let large_zeros = ffi::mat_zeros(1025, 1024, 1);
    assert_eq!(
        large_zeros
            .as_ref()
            .expect("large zero image")
            .non_zero_counts_owned()
            .as_ref()
            .expect("large zero count")
            .as_slice(),
        [0]
    );
    assert_eq!(
        large_zeros
            .as_ref()
            .expect("large zero image")
            .sum_channels_owned()
            .as_ref()
            .expect("large zero sum")
            .as_slice(),
        [0]
    );
    let maximum_bytes = vec![255; 1024 * 1024];
    let maximum = ffi::mat_from_u8(&maximum_bytes, 1024, 1024, 1);
    assert_eq!(
        maximum
            .as_ref()
            .expect("maximum bounded image")
            .sum_channels_owned()
            .as_ref()
            .expect("maximum bounded sum")
            .as_slice(),
        [267_386_880]
    );

    drop(maximum);
    drop(large_zeros);
    drop(gray_zeros);
    drop(zeros);
    drop(bgra);
    drop(gray);
    drop(bgr);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires copied floating-point values to match within native roundoff.
fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
}
