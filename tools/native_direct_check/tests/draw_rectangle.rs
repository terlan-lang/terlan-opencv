//! Direct immutable bounded grayscale rectangle-drawing tests.

use terlan_opencv_native::ffi;

/// Proves exact rectangle rasterization, validation, ownership, and cleanup.
#[test]
fn draw_rectangle_clones_before_rasterization() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [1; 30];
    let source = ffi::mat_from_u8(&source_bytes, 5, 6, 1);
    let source_ref = source.as_ref().expect("rectangle source");
    let rectangle = source_ref.draw_rectangle_owned(1, 1, 4, 3, 7, 1);
    assert_eq!(
        mat_bytes(rectangle.as_ref().expect("rectangle result")),
        [
            1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 1, 1, 7, 1, 1, 7, 1, 1, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1,
            1,
        ]
    );
    let erased = source_ref.draw_rectangle_owned(1, 1, 4, 3, 0, 1);
    assert_eq!(
        mat_bytes(erased.as_ref().expect("erased rectangle")),
        [
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            1,
        ]
    );
    assert_eq!(mat_bytes(source_ref), source_bytes);

    let point_source = ffi::mat_zeros(3, 3, 1);
    let point = point_source
        .as_ref()
        .expect("point rectangle source")
        .draw_rectangle_owned(2, 1, 1, 1, 9, 1);
    assert_eq!(
        mat_bytes(point.as_ref().expect("point rectangle")),
        [0, 0, 0, 0, 0, 9, 0, 0, 0]
    );

    for parameters in [
        (-1, 0, 1, 1, 0, 1),
        (6, 0, 1, 1, 0, 1),
        (0, -1, 1, 1, 0, 1),
        (0, 5, 1, 1, 0, 1),
        (0, 0, 0, 1, 0, 1),
        (0, 0, 7, 1, 0, 1),
        (0, 0, 1, 0, 0, 1),
        (0, 0, 1, 6, 0, 1),
        (5, 0, 2, 1, 0, 1),
        (0, 4, 1, 2, 0, 1),
        (0, 0, 1, 1, -1, 1),
        (0, 0, 1, 1, 256, 1),
        (0, 0, 1, 1, 0, 0),
        (0, 0, 1, 1, 0, 256),
    ] {
        assert!(source_ref
            .draw_rectangle_owned(
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
    let color = ffi::mat_zeros(5, 6, 3);
    let oversized = ffi::mat_zeros(1025, 1024, 1);
    for invalid in [&color, &oversized] {
        assert!(invalid
            .as_ref()
            .expect("invalid rectangle source")
            .draw_rectangle_owned(0, 0, 1, 1, 0, 1)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }

    drop(oversized);
    drop(color);
    drop(point);
    drop(point_source);
    drop(erased);
    drop(rectangle);
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
