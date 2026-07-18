//! Direct first-order Sobel and Scharr tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves exact derivatives, source preservation, rejection, and ownership.
#[test]
fn sobel_edges_are_bounded_absolute_first_derivatives() {
    assert_eq!(ffi::live_mat_count(), 0);
    let ramp_bytes = [0, 10, 20, 0, 10, 20, 0, 10, 20];
    let ramp = ffi::mat_from_u8(&ramp_bytes, 3, 3, 1);
    let ramp_ref = ramp.as_ref().expect("horizontal grayscale ramp");
    let x_edges = ramp_ref.sobel_edges_owned(1, 0, 3);
    let x_edges_ref = x_edges.as_ref().expect("x derivative");
    assert_eq!((x_edges_ref.rows(), x_edges_ref.cols()), (3, 3));
    assert_eq!(x_edges_ref.channels(), 1);
    assert_eq!(
        x_edges_ref
            .to_u8()
            .as_ref()
            .expect("x derivative bytes")
            .as_slice(),
        [40, 80, 40, 40, 80, 40, 40, 80, 40]
    );
    let y_edges = ramp_ref.sobel_edges_owned(0, 1, 3);
    assert_eq!(
        y_edges
            .as_ref()
            .expect("y derivative")
            .to_u8()
            .as_ref()
            .expect("y derivative bytes")
            .as_slice(),
        [0; 9]
    );
    let scharr_x = ramp_ref.scharr_edges_owned(1, 0);
    assert_eq!(
        scharr_x
            .as_ref()
            .expect("Scharr x derivative")
            .to_u8()
            .as_ref()
            .expect("Scharr x derivative bytes")
            .as_slice(),
        [160, 255, 160, 160, 255, 160, 160, 255, 160]
    );
    let scharr_y = ramp_ref.scharr_edges_owned(0, 1);
    assert_eq!(
        scharr_y
            .as_ref()
            .expect("Scharr y derivative")
            .to_u8()
            .as_ref()
            .expect("Scharr y derivative bytes")
            .as_slice(),
        [0; 9]
    );
    assert_eq!(
        ramp_ref
            .to_u8()
            .as_ref()
            .expect("unchanged ramp bytes")
            .as_slice(),
        ramp_bytes
    );

    let bgr = ffi::mat_zeros(3, 3, 3);
    assert!(bgr
        .as_ref()
        .expect("BGR image")
        .sobel_edges_owned(1, 0, 3)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    assert!(bgr
        .as_ref()
        .expect("BGR image")
        .scharr_edges_owned(1, 0)
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    for (x_order, y_order, kernel_size) in [
        (0, 0, 3),
        (1, 1, 3),
        (2, 0, 3),
        (0, 2, 3),
        (-1, 0, 3),
        (1, 0, 0),
        (1, 0, 2),
        (1, 0, 9),
    ] {
        assert!(ramp_ref
            .sobel_edges_owned(x_order, y_order, kernel_size)
            .is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    for (x_order, y_order) in [(0, 0), (1, 1), (2, 0), (0, 2), (-1, 0)] {
        assert!(ramp_ref.scharr_edges_owned(x_order, y_order).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 6);

    drop(bgr);
    drop(scharr_y);
    drop(scharr_x);
    drop(y_edges);
    drop(x_edges);
    drop(ramp);
    assert_eq!(ffi::live_mat_count(), 0);
}
