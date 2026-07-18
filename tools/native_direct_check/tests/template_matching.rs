//! Direct bounded template-matching tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves exact scores, aliasing, input preservation, bounds, and cleanup.
#[test]
fn squared_difference_template_matching_is_bounded() {
    assert_eq!(ffi::live_mat_count(), 0);

    let source_bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let pattern_bytes = [1, 2, 4, 5];
    let source = ffi::mat_from_u8(&source_bytes, 3, 3, 1);
    let pattern = ffi::mat_from_u8(&pattern_bytes, 2, 2, 1);
    let source_ref = source.as_ref().expect("source image");
    let pattern_ref = pattern.as_ref().expect("template pattern");
    assert_close(
        source_ref
            .match_template_sqdiff_owned(pattern_ref)
            .as_ref()
            .expect("template scores")
            .as_slice(),
        &[0.0, 4.0, 36.0, 64.0],
    );
    assert_close(
        source_ref
            .match_template_sqdiff_owned(source_ref)
            .as_ref()
            .expect("aliased template score")
            .as_slice(),
        &[0.0],
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source")
            .as_slice(),
        source_bytes
    );
    assert_eq!(
        pattern_ref
            .to_u8()
            .as_ref()
            .expect("unchanged pattern")
            .as_slice(),
        pattern_bytes
    );

    let bgr_bytes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let bgr = ffi::mat_from_u8(&bgr_bytes, 2, 2, 3);
    let bgr_ref = bgr.as_ref().expect("BGR image");
    assert_close(
        bgr_ref
            .match_template_sqdiff_owned(bgr_ref)
            .as_ref()
            .expect("BGR template score")
            .as_slice(),
        &[0.0],
    );

    let larger = ffi::mat_zeros(4, 4, 1);
    let color = ffi::mat_zeros(2, 2, 3);
    let unit = ffi::mat_zeros(1, 1, 1);
    let output_heavy = ffi::mat_zeros(1025, 1024, 1);
    let work_heavy = ffi::mat_zeros(512, 512, 1);
    let work_pattern = ffi::mat_zeros(256, 256, 1);
    assert!(source_ref
        .match_template_sqdiff_owned(larger.as_ref().expect("larger pattern"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(source_ref
        .match_template_sqdiff_owned(color.as_ref().expect("color pattern"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(output_heavy
        .as_ref()
        .expect("output-heavy image")
        .match_template_sqdiff_owned(unit.as_ref().expect("unit pattern"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(work_heavy
        .as_ref()
        .expect("work-heavy image")
        .match_template_sqdiff_owned(work_pattern.as_ref().expect("work-heavy pattern"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(work_pattern);
    drop(work_heavy);
    drop(output_heavy);
    drop(unit);
    drop(color);
    drop(larger);
    drop(bgr);
    drop(pattern);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Requires native scores to match within bounded floating-point roundoff.
fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1.0e-5);
    }
}
