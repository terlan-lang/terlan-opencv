//! Direct copied lookup-table tests for the generated `cxx` bridge.

use terlan_opencv_native::ffi;

/// Proves exact per-channel transformation, rejection, and fresh ownership.
#[test]
fn lookup_table_is_copied_complete_and_layout_preserving() {
    assert_eq!(ffi::live_mat_count(), 0);
    let source_bytes = [0, 1, 2, 127, 128, 129, 253, 254, 255];
    let source = ffi::mat_from_u8(&source_bytes, 3, 3, 1);
    let source_ref = source.as_ref().expect("grayscale source");
    let mut table = (0_u8..=u8::MAX).rev().collect::<Vec<_>>();
    let transformed = source_ref.apply_lut_owned(&table);
    table.fill(0);
    let transformed_ref = transformed.as_ref().expect("lookup-table result");
    assert_eq!(
        transformed_ref
            .to_u8()
            .as_ref()
            .expect("transformed bytes")
            .as_slice(),
        [255, 254, 253, 128, 127, 126, 2, 1, 0]
    );
    assert_eq!(
        source_ref
            .to_u8()
            .as_ref()
            .expect("unchanged source bytes")
            .as_slice(),
        source_bytes
    );

    for (layout, channels) in [(3, 3_usize), (4, 4_usize)] {
        let color = ffi::mat_from_u8(&vec![0; channels], 1, 1, layout);
        let inverted = color
            .as_ref()
            .expect("color source")
            .apply_lut_owned(&(0_u8..=u8::MAX).rev().collect::<Vec<_>>());
        assert_eq!(
            inverted
                .as_ref()
                .expect("color lookup-table result")
                .to_u8()
                .as_ref()
                .expect("color transformed bytes")
                .as_slice(),
            vec![255; channels]
        );
        drop(inverted);
        drop(color);
    }

    for table in [vec![], vec![0; 255], vec![0; 257]] {
        assert!(source_ref.apply_lut_owned(&table).is_null());
        assert_eq!(ffi::take_last_failure(), 1);
    }
    assert_eq!(ffi::live_mat_count(), 2);
    drop(transformed);
    drop(source);
    assert_eq!(ffi::live_mat_count(), 0);
}
