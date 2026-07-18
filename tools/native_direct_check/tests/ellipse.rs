//! Direct immutable fully contained grayscale ellipse tests.

use terlan_opencv_native::ffi;

/// Proves ellipse outline and fill rasterization, validation, ownership, and cleanup.
#[test]
fn ellipse_operations_clone_before_rasterization() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [1; 25];
    let source = ffi::mat_from_u8(&source_bytes, 5, 5, 1);
    let source_ref = source.as_ref().expect("ellipse source");
    let outlined = source_ref.draw_ellipse_owned(2, 2, 1, 1, 9, 1);
    assert_eq!(
        mat_bytes(outlined.as_ref().expect("ellipse outline")),
        [1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 9, 1, 9, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1]
    );
    let filled = source_ref.fill_ellipse_owned(2, 2, 1, 1, 9);
    assert_eq!(
        mat_bytes(filled.as_ref().expect("ellipse fill")),
        [1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 9, 9, 9, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 1]
    );
    assert_eq!(mat_bytes(source_ref), source_bytes);

    let wide_source = ffi::mat_zeros(5, 7, 1);
    let wide_ref = wide_source.as_ref().expect("wide ellipse source");
    let wide_outline = wide_ref.draw_ellipse_owned(3, 2, 2, 1, 255, 1);
    let wide_fill = wide_ref.fill_ellipse_owned(3, 2, 2, 1, 255);
    assert_eq!(wide_outline.as_ref().expect("wide outline").rows(), 5);
    assert_eq!(wide_outline.as_ref().expect("wide outline").cols(), 7);
    assert_eq!(wide_fill.as_ref().expect("wide fill").rows(), 5);
    assert_eq!(wide_fill.as_ref().expect("wide fill").cols(), 7);
    assert_eq!(mat_bytes(wide_ref), [0; 35]);

    for parameters in [
        (-1, 2, 1, 1, 0, 1),
        (5, 2, 1, 1, 0, 1),
        (2, -1, 1, 1, 0, 1),
        (2, 5, 1, 1, 0, 1),
        (2, 2, 0, 1, 0, 1),
        (2, 2, 1, 0, 0, 1),
        (2, 2, 5, 1, 0, 1),
        (2, 2, 1, 5, 0, 1),
        (0, 2, 1, 1, 0, 1),
        (2, 0, 1, 1, 0, 1),
        (2, 2, 1, 1, -1, 1),
        (2, 2, 1, 1, 256, 1),
        (2, 2, 1, 1, 0, 0),
        (2, 2, 1, 1, 0, 256),
        (2, 2, 1, 1, 0, 5),
    ] {
        assert!(source_ref
            .draw_ellipse_owned(
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
    for parameters in [
        (-1, 2, 1, 1, 0),
        (5, 2, 1, 1, 0),
        (2, -1, 1, 1, 0),
        (2, 5, 1, 1, 0),
        (2, 2, 0, 1, 0),
        (2, 2, 1, 0, 0),
        (2, 2, 5, 1, 0),
        (2, 2, 1, 5, 0),
        (0, 2, 1, 1, 0),
        (2, 0, 1, 1, 0),
        (2, 2, 1, 1, -1),
        (2, 2, 1, 1, 256),
    ] {
        assert!(source_ref
            .fill_ellipse_owned(
                parameters.0,
                parameters.1,
                parameters.2,
                parameters.3,
                parameters.4,
            )
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    let color = ffi::mat_zeros(5, 5, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        let invalid_ref = invalid.as_ref().expect("invalid ellipse source");
        assert!(invalid_ref.draw_ellipse_owned(2, 2, 1, 1, 0, 1).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
        assert!(invalid_ref.fill_ellipse_owned(2, 2, 1, 1, 0).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(wide_fill);
    drop(wide_outline);
    drop(wide_source);
    drop(filled);
    drop(outlined);
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
