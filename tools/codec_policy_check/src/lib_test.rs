//! Tests for the package-owned OpenCV codec policy.

use super::*;
use std::path::PathBuf;

/// Resolves the package root from the checker crate.
fn package_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("checker remains under tools")
        .to_path_buf()
}

/// Parses the committed codec policy.
fn policy() -> Value {
    read_toml(&package_root().join("bindings/opencv.codecs.toml")).expect("codec policy")
}

/// Requires a stable validation failure.
fn expect_error(result: Result<(), String>, expected: &str) {
    let error = result.expect_err("mutation must fail");
    assert!(error.contains(expected), "unexpected error: {error}");
}

/// Finds one public module by its stable qualified name.
fn module_index(manifest: &Value, expected: &str) -> usize {
    manifest["modules"]
        .as_array()
        .expect("modules")
        .iter()
        .position(|module| module["module"] == expected)
        .unwrap_or_else(|| panic!("missing {expected}"))
}

/// Accepts the complete committed codec contract.
#[test]
fn committed_package_is_valid() {
    validate_package(&package_root()).expect("committed package");
}

/// Rejects ownership, filesystem, availability, and error-family drift.
#[test]
fn boundary_policy_drift_is_rejected() {
    for (key, value, expected) in [
        ("input_boundary", "borrowed_bytes", "input_boundary"),
        ("output_boundary", "native_view", "output_boundary"),
        ("filesystem_policy", "ambient", "filesystem"),
        ("availability_source", "runtime_guess", "availability"),
        ("unavailable_error", "io.error", "opencv.*"),
    ] {
        let mut value_policy = policy();
        value_policy[key] = Value::String(value.to_string());
        expect_error(validate_policy(&value_policy), expected);
    }
}

/// Rejects byte-budget expansion or removal.
#[test]
fn byte_limit_drift_is_rejected() {
    for (key, expected) in [
        ("max_encoded_bytes", "encoded byte limit"),
        ("max_decoded_bytes", "decoded byte limit"),
    ] {
        let mut value_policy = policy();
        value_policy[key] = Value::from(1_u64);
        expect_error(validate_policy(&value_policy), expected);
    }
}

/// Rejects drift in deterministic codec parameters.
#[test]
fn deterministic_codec_parameter_drift_is_rejected() {
    for (key, expected) in [
        ("jpeg_encode_quality", "JPEG quality"),
        ("png_compression_level", "PNG compression"),
        ("webp_encode_quality", "WebP quality"),
    ] {
        let mut value_policy = policy();
        value_policy[key] = Value::from(1_u64);
        expect_error(validate_policy(&value_policy), expected);
    }
    let mut value_policy = policy();
    value_policy["tiff_compression"] = Value::String("lzw".to_string());
    expect_error(validate_policy(&value_policy), "TIFF compression");
}

/// Rejects new public or missing forbidden filesystem operations.
#[test]
fn callable_inventory_drift_is_rejected() {
    let mut value_policy = policy();
    value_policy["public_functions"] = serde_json::json!(["decode", "encode", "imread"]);
    expect_error(validate_policy(&value_policy), "only decode and encode");

    let mut value_policy = policy();
    value_policy["forbidden_functions"] = serde_json::json!(["imread"]);
    expect_error(validate_policy(&value_policy), "imread and imwrite");
}

/// Rejects additional, renamed, unavailable, or one-way required codecs.
#[test]
fn codec_format_drift_is_rejected() {
    let mut value_policy = policy();
    value_policy["formats"][0]["name"] = Value::String("webp".to_string());
    expect_error(validate_policy(&value_policy), "bmp");

    let mut value_policy = policy();
    value_policy["formats"][0]["encode"] = Value::Bool(false);
    expect_error(validate_policy(&value_policy), "bmp encoding");

    let mut value_policy = policy();
    let duplicate = value_policy["formats"][0].clone();
    value_policy["formats"]
        .as_array_mut()
        .expect("formats")
        .push(duplicate);
    expect_error(
        validate_policy(&value_policy),
        "exactly BMP, PNG, JPEG, WebP, and TIFF",
    );
}

/// Rejects missing, duplicate, renamed, or renumbered decode modes.
#[test]
fn decode_mode_drift_is_rejected() {
    let mut value_policy = policy();
    value_policy["decode_modes"][0]["value"] = Value::from(2);
    expect_error(validate_policy(&value_policy), "decode modes");

    let mut value_policy = policy();
    value_policy["decode_modes"]
        .as_array_mut()
        .expect("modes")
        .pop();
    expect_error(validate_policy(&value_policy), "decode modes");
}

/// Rejects missing module declarations in every build description.
#[test]
fn module_scope_drift_is_rejected() {
    let lock = read_toml(&package_root().join("upstream/opencv.lock.toml")).expect("lock");
    let manifest =
        read_json(&package_root().join("bindings/opencv.native-binding.json")).expect("manifest");
    let cmake = read_text(&package_root().join("cmake/opencv-cpu.cmake")).expect("cmake");

    let mut changed = lock.clone();
    changed["build"]["modules"] = serde_json::json!(["core", "imgproc"]);
    expect_error(validate_module_scope(&changed, &manifest, &cmake), "lock");

    let mut changed = manifest.clone();
    changed["build"]["linked_libraries"]
        .as_array_mut()
        .expect("libraries")
        .retain(|library| library["name"] != "opencv_imgcodecs");
    expect_error(validate_module_scope(&lock, &changed, &cmake), "build plan");

    let extended_cmake = cmake.replace(
        "core,imgproc,imgcodecs,features,calib3d,ml,dnn",
        "core,imgproc,imgcodecs,features,calib3d,ml,dnn,video",
    );
    validate_module_scope(&lock, &manifest, &extended_cmake)
        .expect("modules after imgcodecs remain valid");

    let changed_cmake = cmake.replace("imgcodecs,", "");
    expect_error(
        validate_module_scope(&lock, &manifest, &changed_cmake),
        "CMake build scope",
    );
}

/// Rejects path-shaped functions and arguments from the public package surface.
#[test]
fn filesystem_surface_is_rejected() {
    let manifest =
        read_json(&package_root().join("bindings/opencv.native-binding.json")).expect("manifest");
    let mut changed = manifest.clone();
    changed["modules"][1]["functions"][0]["name"] = Value::String("imread".to_string());
    expect_error(
        validate_public_surface(&changed),
        "filesystem codec function",
    );

    let mut changed = manifest;
    changed["modules"][1]["functions"][0]["args"][0]["name"] = Value::String("path".to_string());
    expect_error(validate_public_surface(&changed), "path-shaped");
}

/// Rejects missing codec operations and any copied-byte signature drift.
#[test]
fn codec_module_contract_is_exact() {
    let manifest =
        read_json(&package_root().join("bindings/opencv.native-binding.json")).expect("manifest");
    validate_public_surface(&manifest).expect("codec module");
    let codec_index = module_index(&manifest, "opencv.Codecs");

    let mut changed = manifest.clone();
    changed["modules"][codec_index]["functions"]
        .as_array_mut()
        .expect("functions")
        .pop();
    expect_error(validate_public_surface(&changed), "exactly two");

    let mut changed = manifest;
    changed["modules"][codec_index]["functions"][0]["args"][0]["ty"] =
        Value::String("String".to_string());
    expect_error(validate_public_surface(&changed), "argument contract");
}

/// Rejects incomplete probes and any file-oriented codec call.
#[test]
fn link_probe_contract_is_strict() {
    let probe = read_text(&package_root().join("tests/cpp/opencv_link_probe.cc")).expect("probe");
    validate_link_probe(&probe).expect("complete probe");
    expect_error(
        validate_link_probe(&probe.replace("cv::imdecode", "decode")),
        "imdecode",
    );
    expect_error(
        validate_link_probe(&(probe + "\ncv::imwrite(path, image);\n")),
        "imwrite",
    );
}
