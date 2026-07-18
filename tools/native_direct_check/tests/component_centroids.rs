//! Direct connected-component centroid ownership contract.

use terlan_opencv_native::ffi;

/// Proves foreground-only centroid projection across connectivity and empty input.
#[test]
fn component_centroids_are_validated_and_copied_in_label_order() {
    assert_eq!(ffi::live_mat_count(), 0);
    assert_eq!(ffi::live_component_analysis_count(), 0);

    let mask = ffi::mat_from_u8(&[0, 255, 0, 0, 255, 0, 0, 0, 255], 3, 3, 1);
    let mask_ref = mask.as_ref().expect("component mask");
    let four = mask_ref.connected_components_owned(4);
    let eight = mask_ref.connected_components_owned(8);
    let four_values = four
        .as_ref()
        .expect("four-neighbor components")
        .centroids_owned();
    let eight_values = eight
        .as_ref()
        .expect("eight-neighbor components")
        .centroids_owned();

    assert_eq!(
        four_values.as_ref().expect("four centroids").as_slice(),
        [1.0, 0.5, 2.0, 2.0]
    );
    let eight_slice = eight_values.as_ref().expect("eight centroids").as_slice();
    assert_eq!(eight_slice.len(), 2);
    assert!((eight_slice[0] - 4.0 / 3.0).abs() <= 1.0e-9);
    assert_eq!(eight_slice[1], 1.0);

    let empty_mask = ffi::mat_zeros(2, 2, 1);
    let empty = empty_mask
        .as_ref()
        .expect("empty mask")
        .connected_components_owned(4);
    assert!(empty
        .as_ref()
        .expect("empty components")
        .centroids_owned()
        .as_ref()
        .expect("empty centroids")
        .is_empty());

    drop(empty);
    drop(empty_mask);
    drop(eight_values);
    drop(four_values);
    drop(eight);
    drop(four);
    drop(mask);
    assert_eq!(ffi::live_component_analysis_count(), 0);
    assert_eq!(ffi::live_mat_count(), 0);
}
