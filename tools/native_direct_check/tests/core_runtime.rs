//! Direct behavioral tests for portable OpenCV runtime information.

use terlan_opencv_native::ffi;

/// Verifies copied runtime identity and build information are bounded.
#[test]
fn runtime_strings_are_copied_and_identify_the_pinned_build() {
    let version = ffi::opencv_version_owned();
    let version = version
        .as_ref()
        .expect("version")
        .to_str()
        .expect("UTF-8 version");
    assert_eq!(version, "4.13.0");

    let information = ffi::opencv_build_information_owned();
    let information = information
        .as_ref()
        .expect("build information")
        .to_str()
        .expect("UTF-8 build information");
    assert!(information.contains("OpenCV 4.13.0"));
    assert!(information.contains("core"));
    assert!(information.len() < 1024 * 1024);
}

/// Verifies portable hardware and timing observations are valid.
#[test]
fn runtime_numeric_information_is_positive_and_stable() {
    let processors = ffi::opencv_processor_count();
    let threads = ffi::opencv_thread_count();
    let frequency = ffi::opencv_tick_frequency();
    assert!(processors > 0);
    assert!(threads > 0);
    assert!(frequency.is_finite() && frequency > 0.0);
    assert_eq!(ffi::take_last_failure(), 0);

    let first = ffi::opencv_optimized_enabled();
    let second = ffi::opencv_optimized_enabled();
    assert_eq!(first, second);
    assert_eq!(ffi::take_last_failure(), 0);
}
