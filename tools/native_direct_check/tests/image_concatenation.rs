//! Direct checked image-concatenation tests for the generated CXX bridge.

use terlan_opencv_native::ffi;

/// Proves exact composition, aliases, compatibility, preservation, and cleanup.
#[test]
fn horizontal_and_vertical_concatenation_are_checked() {
    assert_eq!(ffi::live_mat_count(), 0);
    let horizontal_first = ffi::mat_from_u8(&[1, 2, 3, 4], 2, 2, 1);
    let horizontal_second = ffi::mat_from_u8(&[5, 6], 2, 1, 1);
    let horizontal_first_ref = horizontal_first.as_ref().expect("horizontal first");
    let horizontal = horizontal_first_ref
        .concat_horizontal_owned(horizontal_second.as_ref().expect("horizontal second"));
    let horizontal_ref = horizontal.as_ref().expect("horizontal concatenation");
    assert_eq!(
        (
            horizontal_ref.rows(),
            horizontal_ref.cols(),
            horizontal_ref.channels()
        ),
        (2, 3, 1)
    );
    assert_eq!(
        horizontal_ref
            .to_u8()
            .as_ref()
            .expect("horizontal bytes")
            .as_slice(),
        [1, 2, 5, 3, 4, 6]
    );

    let vertical_first = ffi::mat_from_u8(&[1, 2], 1, 2, 1);
    let vertical_second = ffi::mat_from_u8(&[3, 4, 5, 6], 2, 2, 1);
    let vertical_first_ref = vertical_first.as_ref().expect("vertical first");
    let vertical = vertical_first_ref
        .concat_vertical_owned(vertical_second.as_ref().expect("vertical second"));
    let vertical_ref = vertical.as_ref().expect("vertical concatenation");
    assert_eq!(
        (
            vertical_ref.rows(),
            vertical_ref.cols(),
            vertical_ref.channels()
        ),
        (3, 2, 1)
    );
    assert_eq!(
        vertical_ref
            .to_u8()
            .as_ref()
            .expect("vertical bytes")
            .as_slice(),
        [1, 2, 3, 4, 5, 6]
    );

    let bgr = ffi::mat_from_u8(&[7, 8, 9], 1, 1, 3);
    let bgr_ref = bgr.as_ref().expect("BGR image");
    let aliased_horizontal = bgr_ref.concat_horizontal_owned(bgr_ref);
    assert_eq!(
        aliased_horizontal
            .as_ref()
            .expect("aliased horizontal")
            .to_u8()
            .as_ref()
            .expect("aliased horizontal bytes")
            .as_slice(),
        [7, 8, 9, 7, 8, 9]
    );
    let aliased_vertical = bgr_ref.concat_vertical_owned(bgr_ref);
    assert_eq!(
        aliased_vertical
            .as_ref()
            .expect("aliased vertical")
            .to_u8()
            .as_ref()
            .expect("aliased vertical bytes")
            .as_slice(),
        [7, 8, 9, 7, 8, 9]
    );

    let bgra_first = ffi::mat_from_u8(&[1, 2, 3, 4], 1, 1, 4);
    let bgra_second = ffi::mat_from_u8(&[5, 6, 7, 8], 1, 1, 4);
    let bgra_joined = bgra_first
        .as_ref()
        .expect("first BGRA")
        .concat_horizontal_owned(bgra_second.as_ref().expect("second BGRA"));
    assert_eq!(
        bgra_joined
            .as_ref()
            .expect("BGRA concatenation")
            .to_u8()
            .as_ref()
            .expect("BGRA bytes")
            .as_slice(),
        [1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        horizontal_first_ref
            .to_u8()
            .as_ref()
            .expect("unchanged horizontal first")
            .as_slice(),
        [1, 2, 3, 4]
    );

    let wrong_rows = ffi::mat_zeros(1, 1, 1);
    assert!(horizontal_first_ref
        .concat_horizontal_owned(wrong_rows.as_ref().expect("wrong rows"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let wrong_columns = ffi::mat_zeros(2, 1, 1);
    assert!(vertical_first_ref
        .concat_vertical_owned(wrong_columns.as_ref().expect("wrong columns"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);
    let wrong_layout = ffi::mat_zeros(2, 1, 3);
    assert!(horizontal_first_ref
        .concat_horizontal_owned(wrong_layout.as_ref().expect("wrong layout"))
        .is_null());
    assert_eq!(ffi::take_last_failure(), 1);

    drop(wrong_layout);
    drop(wrong_columns);
    drop(wrong_rows);
    drop(bgra_joined);
    drop(bgra_second);
    drop(bgra_first);
    drop(aliased_vertical);
    drop(aliased_horizontal);
    drop(bgr);
    drop(vertical);
    drop(vertical_second);
    drop(vertical_first);
    drop(horizontal);
    drop(horizontal_second);
    drop(horizontal_first);
    assert_eq!(ffi::live_mat_count(), 0);
}
