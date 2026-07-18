//! Direct copied integral-image tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves exact row-major values, source integrity, bounds, and no result owner.
#[test]
fn copied_integral_values_are_exact_and_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);

    let source_bytes = [1, 2, 3, 4, 5, 6];
    let source = ffi::mat_from_u8(&source_bytes, 2, 3, 1);
    let source_ref = source.as_ref().expect("grayscale source");
    assert_eq!(
        source_ref
            .integral_values_owned()
            .as_ref()
            .expect("integral values")
            .as_slice(),
        [0, 0, 0, 0, 0, 1, 3, 6, 0, 5, 12, 21]
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source")
            .as_slice(),
        source_bytes
    );
    assert_eq!(ffi::live_mat_count(), 1);

    let zero = ffi::mat_zeros(2, 2, 1);
    assert_eq!(
        zero.as_ref()
            .expect("zero image")
            .integral_values_owned()
            .as_ref()
            .expect("zero integral values")
            .as_slice(),
        [0; 9]
    );

    let color = ffi::mat_from_u8(&[1, 2, 3], 1, 1, 3);
    assert!(color
        .as_ref()
        .expect("color image")
        .integral_values_owned()
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let oversized_bytes = vec![0; 1025 * 1024];
    let oversized = ffi::mat_from_u8(&oversized_bytes, 1025, 1024, 1);
    assert!(oversized
        .as_ref()
        .expect("oversized image")
        .integral_values_owned()
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    let maximum_bytes = vec![255; 1024 * 1024];
    let maximum = ffi::mat_from_u8(&maximum_bytes, 1024, 1024, 1);
    let maximum_values = maximum
        .as_ref()
        .expect("maximum bounded image")
        .integral_values_owned();
    let maximum_values_ref = maximum_values.as_ref().expect("maximum integral values");
    assert_eq!(maximum_values_ref.len(), 1025 * 1025);
    assert_eq!(maximum_values_ref.as_slice()[1025 * 1025 - 1], 267_386_880);

    drop(oversized);
    drop(maximum);
    drop(color);
    drop(zero);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
