//! Direct immutable bounded grayscale line-drawing tests for the CXX bridge.

use terlan_opencv_native::ffi;

/// Proves deterministic rasterization, validation, fresh ownership, and cleanup.
#[test]
fn draw_line_clones_before_rasterization() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [1; 15];
    let source = ffi::mat_from_u8(&source_bytes, 3, 5, 1);
    let source_ref = source.as_ref().expect("line source");
    let horizontal = source_ref.draw_line_owned(0, 1, 4, 1, 7, 1);
    assert_eq!(
        mat_bytes(horizontal.as_ref().expect("horizontal line")),
        [1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1]
    );
    let erased = source_ref.draw_line_owned(0, 1, 4, 1, 0, 1);
    assert_eq!(
        mat_bytes(erased.as_ref().expect("erased line")),
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    );
    assert_eq!(mat_bytes(source_ref), source_bytes);

    let diagonal_source = ffi::mat_zeros(3, 3, 1);
    let diagonal = diagonal_source
        .as_ref()
        .expect("diagonal source")
        .draw_line_owned(0, 0, 2, 2, 9, 1);
    assert_eq!(
        mat_bytes(diagonal.as_ref().expect("diagonal line")),
        [9, 0, 0, 0, 9, 0, 0, 0, 9]
    );
    let point_source = ffi::mat_zeros(1, 1, 1);
    let point = point_source
        .as_ref()
        .expect("point source")
        .draw_line_owned(0, 0, 0, 0, 255, 1);
    assert_eq!(mat_bytes(point.as_ref().expect("point line")), [255]);

    for parameters in [
        (-1, 0, 0, 0, 0, 1),
        (5, 0, 0, 0, 0, 1),
        (0, -1, 0, 0, 0, 1),
        (0, 3, 0, 0, 0, 1),
        (0, 0, -1, 0, 0, 1),
        (0, 0, 5, 0, 0, 1),
        (0, 0, 0, -1, 0, 1),
        (0, 0, 0, 3, 0, 1),
        (0, 0, 0, 0, -1, 1),
        (0, 0, 0, 0, 256, 1),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 256),
    ] {
        assert!(source_ref
            .draw_line_owned(
                parameters.0,
                parameters.1,
                parameters.2,
                parameters.3,
                parameters.4,
                parameters.5,
            )
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(3, 5, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid line source")
            .draw_line_owned(0, 0, 0, 0, 0, 1)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(point);
    drop(point_source);
    drop(diagonal);
    drop(diagonal_source);
    drop(erased);
    drop(horizontal);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}

/// Copies one native matrix into a Rust byte vector.
fn mat_bytes(image: &ffi::OpenCvMat) -> Vec<u8> {
    image
        .to_u8()
        .as_ref()
        .expect("matrix bytes")
        .as_slice()
        .to_vec()
}
