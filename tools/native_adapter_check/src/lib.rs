//! Full-cycle validation for the generated OpenCV native helper.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod adaptive_equalization_contract;
mod affine_sift_contract;
mod agast_keypoints_contract;
mod akaze_features_contract;
mod akaze_matching_contract;
mod arithmetic_contract;
mod binary_matching_contract_support;
mod bitwise_contract;
mod blend_mask_contract;
mod box_blur_contract;
mod brisk_features_contract;
mod brisk_matching_contract;
mod calib3d_contract;
mod calib3d_projection_contract;
mod calib3d_solve_pose_contract;
mod channel_contract;
mod comparison_contract;
mod components_contract;
mod concatenation_contract;
mod contour_approximation_contract;
mod contour_convex_hulls_contract;
mod contour_convexity_defects_contract;
mod contour_filled_mask_contract;
mod contour_signed_distances_contract;
mod contours_contract;
mod contract_support;
mod corner_eigensystem_contract;
mod corner_min_eigenvalue_contract;
mod detect_circles_contract;
mod detect_lines_contract;
mod distance_transform_contract;
mod divide_contract;
mod dnn_blob_contract;
mod dnn_blob_contract_support;
mod dnn_class_index_mask_contract;
mod dnn_class_score_detections_contract;
mod dnn_classification_contract;
mod dnn_confidence_mask_contract;
mod dnn_field_major_detection_contract_support;
mod dnn_inference_contract;
mod dnn_objectness_detections_contract;
mod dnn_segmentation_contract;
mod dnn_ssd_detections_contract;
mod dnn_transposed_class_score_detections_contract;
mod dnn_transposed_objectness_detections_contract;
mod dnn_xyxy_detections_contract;
mod draw_circle_contract;
mod draw_cross_contract;
mod draw_line_contract;
mod draw_rectangle_contract;
mod drawing_contract_support;
mod ellipse_contract;
mod euclidean_distance_contract;
mod extrema_contract;
mod fast_keypoints_contract;
mod features_contract;
mod fill_circle_contract;
mod fill_rectangle_contract;
mod filters_contract;
mod float_matching_contract_support;
mod flood_fill_contract;
mod gradient_magnitude_contract;
mod harris_response_contract;
mod integral_contract;
mod invert_contract;
mod kaze_features_contract;
mod kaze_matching_contract;
mod line_segments_contract;
mod lut_contract;
mod mask_contract;
mod mask_selection_contract;
mod mean_absolute_error_contract;
mod mean_shift_contract;
mod merge_channels_contract;
mod ml_bayes_contract;
mod ml_boost_contract;
mod ml_decision_tree_contract;
mod ml_gaussian_mixture_contract;
mod ml_knn_contract;
mod ml_logistic_contract;
mod ml_mlp_contract;
mod ml_random_forest_contract;
mod ml_sgd_svm_contract;
mod ml_svm_contract;
mod morphology_contract;
mod mser_regions_contract;
mod multiply_contract;
mod non_zero_approximate_hull_contract;
mod non_zero_bounds_contract;
mod non_zero_centroid_contract;
mod non_zero_convex_hull_contract;
mod non_zero_enclosing_circle_contract;
mod non_zero_enclosing_triangle_contract;
mod non_zero_fit_ellipse_contract;
mod non_zero_fit_line_contract;
mod non_zero_min_area_box_contract;
mod orb_features_contract;
mod orb_matching_contract;
mod otsu_threshold_contract;
mod pre_corner_response_contract;
mod protocol;
mod pyramid_contract;
mod replace_channel_contract;
mod saturated_arithmetic_contract;
mod scale_abs_contract;
mod shape_match_scores_contract;
mod sharpen_contract;
mod sift_features_contract;
mod sift_matching_contract;
mod simple_blob_contract;
mod sobel_contract;
mod statistics_contract;
mod template_match_contract;
mod threshold_contract;
mod tile_contract;
mod transpose_contract;
mod triangle_threshold_contract;

use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use contract_support::*;
use protocol::{decode_reply, NativeReply};

const MAT_TYPE: &str = "opencv.Mat.Mat";

/// Runs the complete OCV0.2 behavior and ownership contract.
pub fn run_native_contract(helper_path: impl AsRef<OsStr>) -> Result<(), String> {
    let mut helper = NativeHelper::spawn(helper_path)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)?;
    adaptive_equalization_contract::run(&mut helper)?;
    affine_sift_contract::run(&mut helper)?;
    agast_keypoints_contract::run(&mut helper)?;
    akaze_features_contract::run(&mut helper)?;
    akaze_matching_contract::run(&mut helper)?;
    blend_mask_contract::run(&mut helper)?;
    bitwise_contract::run(&mut helper)?;
    box_blur_contract::run(&mut helper)?;
    brisk_features_contract::run(&mut helper)?;
    brisk_matching_contract::run(&mut helper)?;
    calib3d_contract::run(&mut helper)?;
    calib3d_projection_contract::run(&mut helper)?;
    calib3d_solve_pose_contract::run(&mut helper)?;
    channel_contract::run(&mut helper)?;
    comparison_contract::run(&mut helper)?;
    components_contract::run(&mut helper)?;
    contour_approximation_contract::run(&mut helper)
        .map_err(|error| format!("contour approximation contract: {error}"))?;
    contour_convex_hulls_contract::run(&mut helper)
        .map_err(|error| format!("contour convex-hulls contract: {error}"))?;
    contour_convexity_defects_contract::run(&mut helper)
        .map_err(|error| format!("contour convexity-defects contract: {error}"))?;
    contour_filled_mask_contract::run(&mut helper)
        .map_err(|error| format!("contour filled-mask contract: {error}"))?;
    contour_signed_distances_contract::run(&mut helper)
        .map_err(|error| format!("contour signed-distances contract: {error}"))?;
    concatenation_contract::run(&mut helper)?;
    contours_contract::run(&mut helper)?;
    corner_eigensystem_contract::run(&mut helper)?;
    corner_min_eigenvalue_contract::run(&mut helper)?;
    distance_transform_contract::run(&mut helper)?;
    detect_circles_contract::run(&mut helper)?;
    detect_lines_contract::run(&mut helper)?;
    divide_contract::run(&mut helper)?;
    dnn_blob_contract::run(&mut helper)?;
    dnn_class_index_mask_contract::run(&mut helper)?;
    dnn_classification_contract::run(&mut helper)?;
    dnn_class_score_detections_contract::run(&mut helper)?;
    dnn_confidence_mask_contract::run(&mut helper)?;
    dnn_inference_contract::run(&mut helper)?;
    dnn_objectness_detections_contract::run(&mut helper)?;
    dnn_segmentation_contract::run(&mut helper)?;
    dnn_ssd_detections_contract::run(&mut helper)?;
    dnn_transposed_class_score_detections_contract::run(&mut helper)?;
    dnn_transposed_objectness_detections_contract::run(&mut helper)?;
    dnn_xyxy_detections_contract::run(&mut helper)?;
    draw_circle_contract::run(&mut helper)?;
    draw_cross_contract::run(&mut helper)?;
    draw_line_contract::run(&mut helper)?;
    draw_rectangle_contract::run(&mut helper)?;
    ellipse_contract::run(&mut helper)?;
    euclidean_distance_contract::run(&mut helper)?;
    extrema_contract::run(&mut helper)?;
    fast_keypoints_contract::run(&mut helper)?;
    features_contract::run(&mut helper)?;
    fill_circle_contract::run(&mut helper)?;
    fill_rectangle_contract::run(&mut helper)?;
    filters_contract::run(&mut helper)?;
    flood_fill_contract::run(&mut helper)?;
    gradient_magnitude_contract::run(&mut helper)?;
    harris_response_contract::run(&mut helper)?;
    integral_contract::run(&mut helper)?;
    invert_contract::run(&mut helper)?;
    kaze_features_contract::run(&mut helper)?;
    kaze_matching_contract::run(&mut helper)?;
    line_segments_contract::run(&mut helper)?;
    lut_contract::run(&mut helper)?;
    arithmetic_contract::run(&mut helper)?;
    mask_contract::run(&mut helper)?;
    mask_selection_contract::run(&mut helper)?;
    mean_absolute_error_contract::run(&mut helper)?;
    mean_shift_contract::run(&mut helper)?;
    merge_channels_contract::run(&mut helper)?;
    ml_bayes_contract::run(&mut helper)?;
    ml_boost_contract::run(&mut helper)?;
    ml_decision_tree_contract::run(&mut helper)?;
    ml_gaussian_mixture_contract::run(&mut helper)?;
    ml_knn_contract::run(&mut helper)?;
    ml_logistic_contract::run(&mut helper)?;
    ml_mlp_contract::run(&mut helper)?;
    ml_random_forest_contract::run(&mut helper)?;
    ml_sgd_svm_contract::run(&mut helper)?;
    ml_svm_contract::run(&mut helper)?;
    morphology_contract::run(&mut helper)?;
    multiply_contract::run(&mut helper)?;
    mser_regions_contract::run(&mut helper)?;
    orb_features_contract::run(&mut helper)?;
    orb_matching_contract::run(&mut helper)?;
    non_zero_approximate_hull_contract::run(&mut helper)?;
    non_zero_bounds_contract::run(&mut helper)?;
    non_zero_centroid_contract::run(&mut helper)?;
    non_zero_convex_hull_contract::run(&mut helper)?;
    non_zero_enclosing_triangle_contract::run(&mut helper)?;
    non_zero_enclosing_circle_contract::run(&mut helper)?;
    non_zero_fit_ellipse_contract::run(&mut helper)?;
    non_zero_fit_line_contract::run(&mut helper)?;
    non_zero_min_area_box_contract::run(&mut helper)?;
    otsu_threshold_contract::run(&mut helper)?;
    pre_corner_response_contract::run(&mut helper)?;
    pyramid_contract::run(&mut helper)?;
    replace_channel_contract::run(&mut helper)?;
    scale_abs_contract::run(&mut helper)?;
    saturated_arithmetic_contract::run(&mut helper)?;
    sharpen_contract::run(&mut helper)?;
    sift_features_contract::run(&mut helper)?;
    sift_matching_contract::run(&mut helper)?;
    simple_blob_contract::run(&mut helper)?;
    shape_match_scores_contract::run(&mut helper)?;
    sobel_contract::run(&mut helper)?;
    statistics_contract::run(&mut helper)?;
    template_match_contract::run(&mut helper)?;
    threshold_contract::run(&mut helper)?;
    tile_contract::run(&mut helper)?;
    triangle_threshold_contract::run(&mut helper)?;
    transpose_contract::run(&mut helper)?;

    let mut source = vec![0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0];
    let original = source.clone();
    let input = bytes_arg(&source);
    let mat = expect_handle(helper.call(
        "opencv.mat.from_u8",
        &[input, int_arg(2), int_arg(2), atom_arg("bgr")],
    )?)?;
    source.fill(17);

    expect_int(helper.call("opencv.mat.rows", &[mat.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[mat.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.channels", &[mat.arg()])?, 3)?;
    expect_bool(helper.call("opencv.mat.is_contiguous", &[mat.arg()])?, true)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[mat.arg()])?, &original)?;
    expect_error_code(
        helper.call("opencv.mat.rows", &[mat.arg_with_owner("foreign-worker")])?,
        "cross_owner_handle",
    )?;
    expect_error_code(
        helper.call(
            "opencv.mat.rows",
            &[mat.arg_with_type("other.Package.Resource")],
        )?,
        "handle_type_mismatch",
    )?;

    let clone = expect_handle(helper.call("opencv.mat.clone", &[mat.arg()])?)?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 2)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[clone.arg()])?, &original)?;

    let resized = expect_handle(helper.call(
        "opencv.imgproc.resize",
        &[mat.arg(), int_arg(1), int_arg(1), atom_arg("nearest")],
    )?)?;
    expect_int(helper.call("opencv.mat.rows", &[resized.arg()])?, 1)?;
    expect_int(helper.call("opencv.mat.cols", &[resized.arg()])?, 1)?;
    expect_int(helper.call("opencv.mat.channels", &[resized.arg()])?, 3)?;

    let gray = expect_handle(helper.call(
        "opencv.imgproc.cvt_color",
        &[mat.arg(), atom_arg("bgr_to_gray")],
    )?)?;
    expect_int(helper.call("opencv.mat.channels", &[gray.arg()])?, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[gray.arg()])?,
        &[0, 255, 76, 150],
    )?;
    let equalized = expect_handle(helper.call("opencv.imgproc.equalize_hist", &[gray.arg()])?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[equalized.arg()])?,
        &[0, 255, 85, 170],
    )?;
    let histogram = expect_ints(helper.call("opencv.imgproc.histogram", &[gray.arg()])?)?;
    if histogram.len() != 256
        || histogram[0] != 1
        || histogram[76] != 1
        || histogram[150] != 1
        || histogram[255] != 1
        || histogram.iter().sum::<i64>() != 4
    {
        return Err(format!("unexpected grayscale histogram: {histogram:?}"));
    }
    let thresholded = expect_handle(helper.call(
        "opencv.imgproc.threshold",
        &[
            gray.arg(),
            float_arg(100.0),
            float_arg(200.0),
            atom_arg("binary_threshold"),
        ],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[thresholded.arg()])?,
        &[0, 200, 0, 200],
    )?;
    let normalized = expect_handle(helper.call(
        "opencv.imgproc.normalize_min_max",
        &[thresholded.arg(), float_arg(0.0), float_arg(255.0)],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[normalized.arg()])?,
        &[0, 255, 0, 255],
    )?;
    let eroded = expect_handle(helper.call(
        "opencv.imgproc.erode",
        &[thresholded.arg(), int_arg(3), int_arg(3), int_arg(1)],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[eroded.arg()])?,
        &[0, 0, 0, 0],
    )?;
    let dilated = expect_handle(helper.call(
        "opencv.imgproc.dilate",
        &[thresholded.arg(), int_arg(3), int_arg(3), int_arg(1)],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[dilated.arg()])?,
        &[200, 200, 200, 200],
    )?;
    let edge_source = expect_handle(helper.call(
        "opencv.mat.zeros",
        &[int_arg(3), int_arg(3), atom_arg("gray")],
    )?)?;
    let edges = expect_handle(helper.call(
        "opencv.imgproc.canny",
        &[
            edge_source.arg(),
            float_arg(10.0),
            float_arg(20.0),
            int_arg(3),
            bool_arg(false),
        ],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[edges.arg()])?,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0],
    )?;
    let edges_l2 = expect_handle(helper.call(
        "opencv.imgproc.canny",
        &[
            edge_source.arg(),
            float_arg(10.0),
            float_arg(20.0),
            int_arg(3),
            bool_arg(true),
        ],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[edges_l2.arg()])?,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0],
    )?;

    let blurred = expect_handle(helper.call(
        "opencv.imgproc.gaussian_blur",
        &[mat.arg(), int_arg(3), int_arg(3), float_arg(0.0)],
    )?)?;
    expect_int(helper.call("opencv.mat.rows", &[blurred.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[blurred.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.channels", &[blurred.arg()])?, 3)?;

    let cropped = expect_handle(helper.call(
        "opencv.imgproc.crop",
        &[mat.arg(), int_arg(1), int_arg(0), int_arg(1), int_arg(2)],
    )?)?;
    expect_int(helper.call("opencv.mat.rows", &[cropped.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[cropped.arg()])?, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[cropped.arg()])?,
        &[255, 255, 255, 0, 255, 0],
    )?;
    let padded = expect_handle(helper.call(
        "opencv.imgproc.pad",
        &[
            cropped.arg(),
            int_arg(1),
            int_arg(0),
            int_arg(1),
            int_arg(0),
            atom_arg("constant"),
        ],
    )?)?;
    expect_int(helper.call("opencv.mat.rows", &[padded.arg()])?, 3)?;
    expect_int(helper.call("opencv.mat.cols", &[padded.arg()])?, 2)?;
    let flipped =
        expect_handle(helper.call("opencv.imgproc.flip", &[mat.arg(), atom_arg("horizontal")])?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[flipped.arg()])?,
        &[255, 255, 255, 0, 0, 0, 0, 255, 0, 0, 0, 255],
    )?;
    let rotated = expect_handle(helper.call(
        "opencv.imgproc.rotate",
        &[cropped.arg(), atom_arg("clockwise_90")],
    )?)?;
    expect_int(helper.call("opencv.mat.rows", &[rotated.arg()])?, 1)?;
    expect_int(helper.call("opencv.mat.cols", &[rotated.arg()])?, 2)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[rotated.arg()])?,
        &[0, 255, 0, 255, 255, 255],
    )?;
    let identity = affine_arg([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let affine = expect_handle(helper.call(
        "opencv.imgproc.warp_affine",
        &[
            mat.arg(),
            identity,
            int_arg(2),
            int_arg(2),
            atom_arg("nearest"),
            atom_arg("constant"),
        ],
    )?)?;
    expect_bytes(helper.call("opencv.mat.to_u8", &[affine.arg()])?, &original)?;
    let perspective_identity = perspective_arg([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let perspective = expect_handle(helper.call(
        "opencv.imgproc.warp_perspective",
        &[
            mat.arg(),
            perspective_identity,
            int_arg(2),
            int_arg(2),
            atom_arg("nearest"),
            atom_arg("constant"),
        ],
    )?)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[perspective.arg()])?,
        &original,
    )?;

    let encoded =
        expect_bytes_value(helper.call("opencv.codecs.encode", &[gray.arg(), atom_arg("bmp")])?)?;
    if !encoded.starts_with(b"BM") {
        return Err("encoded BMP does not have its required signature".to_string());
    }
    let decoded = expect_handle(helper.call(
        "opencv.codecs.decode",
        &[bytes_arg(&encoded), atom_arg("grayscale")],
    )?)?;
    expect_int(helper.call("opencv.mat.rows", &[decoded.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.cols", &[decoded.arg()])?, 2)?;
    expect_int(helper.call("opencv.mat.channels", &[decoded.arg()])?, 1)?;
    expect_bytes(
        helper.call("opencv.mat.to_u8", &[decoded.arg()])?,
        &[0, 255, 76, 150],
    )?;
    expect_error_code(
        helper.call(
            "opencv.codecs.decode",
            &[bytes_arg(b"not-bmp"), atom_arg("grayscale")],
        )?,
        "opencv.invalid_encoded_image",
    )?;
    expect_error_code(
        helper.call(
            "opencv.codecs.decode",
            &[bytes_arg(b"BM"), atom_arg("invented")],
        )?,
        "invalid_enum_value",
    )?;
    expect_error_code(
        helper.call("opencv.codecs.encode", &[mat.arg(), atom_arg("invented")])?,
        "invalid_enum_value",
    )?;
    for arguments in [
        vec![mat.arg(), int_arg(2), int_arg(0), int_arg(1), int_arg(1)],
        vec![
            mat.arg(),
            int_arg(0),
            int_arg(0),
            int_arg(i64::MAX),
            int_arg(1),
        ],
    ] {
        expect_error_code(
            helper.call("opencv.imgproc.crop", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.pad",
            &[
                mat.arg(),
                int_arg(-1),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                atom_arg("constant"),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.pad",
            &[
                mat.arg(),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                int_arg(0),
                atom_arg("invented"),
            ],
        )?,
        "invalid_enum_value",
    )?;

    expect_error_code(
        helper.call(
            "opencv.mat.from_u8",
            &[bytes_arg(&[1, 2]), int_arg(2), int_arg(2), atom_arg("bgr")],
        )?,
        "opencv.invalid_argument",
    )?;
    for arguments in [
        vec![bytes_arg(&[0; 13]), int_arg(2), int_arg(2), atom_arg("bgr")],
        vec![bytes_arg(&[]), int_arg(-1), int_arg(1), atom_arg("bgr")],
        vec![
            bytes_arg(&[]),
            int_arg(i64::MAX),
            int_arg(1),
            atom_arg("bgr"),
        ],
    ] {
        expect_error_code(
            helper.call("opencv.mat.from_u8", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    for arguments in [
        vec![
            mat.arg(),
            int_arg(i64::MAX),
            int_arg(1),
            atom_arg("nearest"),
        ],
        vec![mat.arg(), int_arg(0), int_arg(1), atom_arg("nearest")],
    ] {
        expect_error_code(
            helper.call("opencv.imgproc.resize", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.cvt_color",
            &[mat.arg(), atom_arg("gray_to_bgr")],
        )?,
        "opencv.invalid_argument",
    )?;
    for arguments in [
        vec![mat.arg(), int_arg(2), int_arg(3), float_arg(0.0)],
        vec![mat.arg(), int_arg(257), int_arg(3), float_arg(0.0)],
        vec![mat.arg(), int_arg(3), int_arg(3), float_arg(-0.1)],
        vec![mat.arg(), int_arg(3), int_arg(3), float_arg(f64::NAN)],
        vec![mat.arg(), int_arg(3), int_arg(3), float_arg(f64::INFINITY)],
    ] {
        expect_error_code(
            helper.call("opencv.imgproc.gaussian_blur", &arguments)?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.resize",
            &[mat.arg(), int_arg(1), int_arg(1), atom_arg("invented")],
        )?,
        "invalid_enum_value",
    )?;
    for operation in ["opencv.imgproc.flip", "opencv.imgproc.rotate"] {
        expect_error_code(
            helper.call(operation, &[mat.arg(), atom_arg("invented")])?,
            "invalid_enum_value",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.threshold",
            &[
                gray.arg(),
                float_arg(100.0),
                float_arg(200.0),
                atom_arg("invented"),
            ],
        )?,
        "invalid_enum_value",
    )?;
    for (threshold_value, maximum_value) in [
        (-0.1, 200.0),
        (255.1, 200.0),
        (f64::NAN, 200.0),
        (f64::INFINITY, 200.0),
        (100.0, -0.1),
        (100.0, 255.1),
        (100.0, f64::NAN),
        (100.0, f64::INFINITY),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.threshold",
                &[
                    gray.arg(),
                    float_arg(threshold_value),
                    float_arg(maximum_value),
                    atom_arg("binary_threshold"),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for operation in ["opencv.imgproc.erode", "opencv.imgproc.dilate"] {
        for (kernel_width, kernel_height, iterations) in [
            (0, 3, 1),
            (2, 3, 1),
            (3, 0, 1),
            (3, 2, 1),
            (257, 3, 1),
            (3, 257, 1),
            (3, 3, 0),
            (3, 3, 65),
        ] {
            expect_error_code(
                helper.call(
                    operation,
                    &[
                        gray.arg(),
                        int_arg(kernel_width),
                        int_arg(kernel_height),
                        int_arg(iterations),
                    ],
                )?,
                "opencv.invalid_argument",
            )?;
        }
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.canny",
            &[
                mat.arg(),
                float_arg(10.0),
                float_arg(20.0),
                int_arg(3),
                bool_arg(false),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    for (low_threshold, high_threshold, aperture_size) in [
        (-0.1, 20.0, 3),
        (10.0, 65536.0, 3),
        (f64::NAN, 20.0, 3),
        (10.0, f64::NAN, 3),
        (f64::INFINITY, 20.0, 3),
        (10.0, f64::INFINITY, 3),
        (20.0, 10.0, 3),
        (10.0, 20.0, 1),
        (10.0, 20.0, 4),
        (10.0, 20.0, 9),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.canny",
                &[
                    gray.arg(),
                    float_arg(low_threshold),
                    float_arg(high_threshold),
                    int_arg(aperture_size),
                    bool_arg(false),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for (minimum, maximum) in [
        (-0.1, 255.0),
        (0.0, 255.1),
        (f64::NAN, 255.0),
        (0.0, f64::NAN),
        (f64::INFINITY, 255.0),
        (0.0, f64::INFINITY),
        (20.0, 20.0),
        (20.0, 10.0),
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.normalize_min_max",
                &[gray.arg(), float_arg(minimum), float_arg(maximum)],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    for transform in [
        [f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0],
        [f64::INFINITY, 0.0, 0.0, 0.0, 1.0, 0.0],
        [1_000_001.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.warp_affine",
                &[
                    mat.arg(),
                    affine_arg(transform),
                    int_arg(2),
                    int_arg(2),
                    atom_arg("nearest"),
                    atom_arg("constant"),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call("opencv.imgproc.equalize_hist", &[mat.arg()])?,
        "opencv.invalid_argument",
    )?;
    expect_error_code(
        helper.call("opencv.imgproc.histogram", &[mat.arg()])?,
        "opencv.invalid_argument",
    )?;
    let oversized_histogram_source = expect_handle(helper.call(
        "opencv.mat.zeros",
        &[int_arg(4097), int_arg(4096), atom_arg("gray")],
    )?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.histogram",
            &[oversized_histogram_source.arg()],
        )?,
        "opencv.invalid_argument",
    )?;
    expect_unit(helper.call("opencv.mat.dispose", &[oversized_histogram_source.arg()])?)?;
    expect_error_code(
        helper.call(
            "opencv.imgproc.warp_affine",
            &[
                mat.arg(),
                affine_arg([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                int_arg(0),
                int_arg(2),
                atom_arg("nearest"),
                atom_arg("constant"),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    for (interpolation, border_mode) in [("invented", "constant"), ("nearest", "invented")] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.warp_affine",
                &[
                    mat.arg(),
                    affine_arg([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                    int_arg(2),
                    int_arg(2),
                    atom_arg(interpolation),
                    atom_arg(border_mode),
                ],
            )?,
            "invalid_enum_value",
        )?;
    }
    for transform in [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, f64::NAN, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, f64::INFINITY, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1_000_001.0, 0.0, 1.0],
    ] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.warp_perspective",
                &[
                    mat.arg(),
                    perspective_arg(transform),
                    int_arg(2),
                    int_arg(2),
                    atom_arg("nearest"),
                    atom_arg("constant"),
                ],
            )?,
            "opencv.invalid_argument",
        )?;
    }
    expect_error_code(
        helper.call(
            "opencv.imgproc.warp_perspective",
            &[
                mat.arg(),
                perspective_arg([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                int_arg(0),
                int_arg(2),
                atom_arg("nearest"),
                atom_arg("constant"),
            ],
        )?,
        "opencv.invalid_argument",
    )?;
    for (interpolation, border_mode) in [("invented", "constant"), ("nearest", "invented")] {
        expect_error_code(
            helper.call(
                "opencv.imgproc.warp_perspective",
                &[
                    mat.arg(),
                    perspective_arg([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    int_arg(2),
                    int_arg(2),
                    atom_arg(interpolation),
                    atom_arg(border_mode),
                ],
            )?,
            "invalid_enum_value",
        )?;
    }
    expect_int(helper.call("opencv.mat.live_count", &[])?, 20)?;

    for handle in [
        &equalized,
        &perspective,
        &affine,
        &edges_l2,
        &edges,
        &edge_source,
        &normalized,
        &dilated,
        &eroded,
        &thresholded,
        &rotated,
        &flipped,
        &padded,
        &cropped,
        &decoded,
        &blurred,
        &gray,
        &resized,
        &clone,
    ] {
        expect_unit(helper.call("opencv.mat.dispose", &[handle.arg()])?)?;
    }
    for _ in 0..100 {
        let temporary = expect_handle(helper.call("opencv.mat.clone", &[mat.arg()])?)?;
        expect_unit(helper.call("opencv.mat.dispose", &[temporary.arg()])?)?;
        expect_error_code(
            helper.call("opencv.mat.rows", &[temporary.arg()])?,
            "stale_handle",
        )?;
    }
    expect_unit(helper.call("opencv.mat.dispose", &[mat.arg()])?)?;
    expect_error_code(
        helper.call("opencv.mat.rows", &[mat.arg()])?,
        "stale_handle",
    )?;
    expect_int(helper.call("opencv.mat.live_count", &[])?, 0)
}

/// One live helper process with correlated request sequencing.
struct NativeHelper {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next_request_id: u64,
}

impl NativeHelper {
    /// Starts the generated helper with isolated standard streams.
    fn spawn(path: impl AsRef<OsStr>) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("cannot start generated helper: {error}"))?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| "generated helper stdin is unavailable".to_string())?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| "generated helper stdout is unavailable".to_string())?;
        Ok(Self {
            child,
            input,
            output: BufReader::new(output),
            next_request_id: 0,
        })
    }

    /// Executes one protocol call and decodes its correlated reply.
    fn call(&mut self, operation: &str, args: &[String]) -> Result<NativeReply, String> {
        self.next_request_id += 1;
        let mut fields = vec![
            "call".to_string(),
            self.next_request_id.to_string(),
            STANDARD.encode(operation),
        ];
        fields.extend(args.iter().cloned());
        writeln!(self.input, "{}", fields.join(" "))
            .and_then(|()| self.input.flush())
            .map_err(|error| format!("cannot write helper request: {error}"))?;
        let mut line = String::new();
        let read = self
            .output
            .read_line(&mut line)
            .map_err(|error| format!("cannot read helper reply: {error}"))?;
        if read == 0 {
            return Err("generated helper exited without replying".to_string());
        }
        let reply = decode_reply(line.trim_end(), self.next_request_id).map_err(|error| {
            format!("operation {operation} with arguments {args:?} failed: {error}")
        })?;
        if std::env::var_os("TERLAN_OPENCV_TRACE_HELPER_ERRORS").is_some()
            && matches!(reply, NativeReply::Error { .. })
        {
            eprintln!("helper error from {operation} {args:?}: {reply:?}");
        }
        Ok(reply)
    }
}

impl Drop for NativeHelper {
    /// Stops the helper even when a contract assertion fails.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Stable identity for one generated opaque resource handle.
#[derive(Debug)]
struct NativeHandle {
    owner: String,
    id: u64,
    generation: u64,
    type_name: String,
}

impl NativeHandle {
    /// Encodes this handle as a helper argument.
    fn arg(&self) -> String {
        format!(
            "h:{}:{}:{}:{}",
            STANDARD.encode(&self.owner),
            self.id,
            self.generation,
            STANDARD.encode(&self.type_name)
        )
    }

    /// Encodes this handle with a forged worker identity for rejection checks.
    fn arg_with_owner(&self, owner: &str) -> String {
        format!(
            "h:{}:{}:{}:{}",
            STANDARD.encode(owner),
            self.id,
            self.generation,
            STANDARD.encode(&self.type_name)
        )
    }

    /// Encodes this handle with a forged resource type for rejection checks.
    fn arg_with_type(&self, type_name: &str) -> String {
        format!(
            "h:{}:{}:{}:{}",
            STANDARD.encode(&self.owner),
            self.id,
            self.generation,
            STANDARD.encode(type_name)
        )
    }
}
