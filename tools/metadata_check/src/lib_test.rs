//! Tests for the committed OpenCV metadata contract.

use std::fs;

use serde_json::Value;

use super::{
    package_root, snake_name, stable_symbol_id, validate_generated_aot_adapter, validate_offline,
    COMPILE_DATABASE, GENERATED_BOUNDARY, GENERATED_HELPER, HEADER, MANIFEST,
};

/// Copies every input required by the offline validator into an isolated root.
fn copy_fixture(name: &str) -> std::path::PathBuf {
    let source = package_root();
    let root = std::env::temp_dir().join(format!(
        "opencv_metadata_{name}_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("bindings/cpp")).expect("create temporary package");
    for path in [MANIFEST, COMPILE_DATABASE, HEADER] {
        fs::copy(source.join(path), root.join(path)).expect("copy metadata fixture input");
    }
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(source.join(MANIFEST)).expect("read source manifest"),
    )
    .expect("parse source manifest");
    for path in manifest["build"]["adapter_headers"]
        .as_array()
        .expect("adapter headers")
    {
        let path = path.as_str().expect("adapter header path");
        let target = root.join("bindings").join(path);
        fs::create_dir_all(target.parent().expect("adapter header parent"))
            .expect("create adapter header parent");
        fs::copy(source.join("bindings").join(path), target)
            .expect("copy adapter header fixture input");
    }
    root
}

/// Copies generated AOT adapter evidence into an isolated package root.
fn copy_generated_fixture(name: &str) -> std::path::PathBuf {
    let source = package_root();
    let root = std::env::temp_dir().join(format!(
        "opencv_generated_aot_{name}_{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let _ = fs::remove_dir_all(&root);
    for path in [GENERATED_BOUNDARY, GENERATED_HELPER] {
        let target = root.join(path);
        fs::create_dir_all(target.parent().expect("generated input parent"))
            .expect("create generated input parent");
        fs::copy(source.join(path), target).expect("copy generated AOT adapter input");
    }
    root
}

/// Writes one structured JSON mutation into a copied fixture.
fn write_json(root: &std::path::Path, path: &str, value: &Value) {
    fs::write(
        root.join(path),
        serde_json::to_vec_pretty(value).expect("render mutated JSON"),
    )
    .expect("write mutated JSON");
}

/// Keeps generated IDs compatible with the package's established resource names.
#[test]
fn generated_symbol_ids_are_stable() {
    assert_eq!(snake_name("OpenCvMat"), "opencv_mat");
    assert_eq!(snake_name("MatrixDepth"), "matrix_depth");
    let symbol = serde_json::json!({
        "kind": "method",
        "cpp_name": "depth_code",
        "receiver": "terlan_opencv::OpenCvMat"
    });
    assert_eq!(
        stable_symbol_id(symbol.as_object().expect("symbol object")).expect("stable ID"),
        "method.opencv_mat.depth_code"
    );
}

/// Requires a deterministic offline rejection and removes the fixture.
fn expect_rejection(root: std::path::PathBuf, expected: &str) {
    let error = validate_offline(&root).expect_err("metadata drift must fail");
    assert!(error.contains(expected), "unexpected error: {error}");
    fs::remove_dir_all(root).expect("remove temporary package");
}

/// Accepts the committed adapter metadata and compile provenance.
#[test]
fn committed_metadata_is_consistent() {
    validate_offline(&package_root()).expect("committed metadata must validate");
}

/// Rejects compile-command drift before live extraction is attempted.
#[test]
fn compile_provenance_drift_is_rejected() {
    let root = copy_fixture("compile_drift");
    let mut database: Value = serde_json::from_str(
        &fs::read_to_string(root.join(COMPILE_DATABASE)).expect("read database"),
    )
    .expect("parse database");
    database[0]["arguments"][2] = Value::String("-std=c++11".into());
    write_json(&root, COMPILE_DATABASE, &database);
    expect_rejection(root, "compile_commands.json");
}

/// Rejects producer, namespace, header, source-list, and language drift.
#[test]
fn metadata_identity_drift_is_rejected() {
    let mutations = [
        (
            "producer",
            vec!["cpp_metadata", "producer", "name"],
            "other",
            "producer",
        ),
        (
            "namespace",
            vec!["cpp_metadata", "namespace"],
            "other",
            "namespace",
        ),
        (
            "header",
            vec!["cpp_metadata", "header"],
            "other.hpp",
            "header",
        ),
        (
            "language",
            vec!["cpp_metadata", "compile", "language_standard"],
            "c++11",
            "language standard",
        ),
    ];
    for (name, path, replacement, expected) in mutations {
        let root = copy_fixture(name);
        let mut manifest: Value =
            serde_json::from_str(&fs::read_to_string(root.join(MANIFEST)).expect("read manifest"))
                .expect("parse manifest");
        let mut target = &mut manifest;
        for segment in path {
            target = &mut target[segment];
        }
        *target = Value::String(replacement.into());
        write_json(&root, MANIFEST, &manifest);
        expect_rejection(root, expected);
    }

    let root = copy_fixture("sources");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(root.join(MANIFEST)).expect("read manifest"))
            .expect("parse manifest");
    manifest["cpp_metadata"]["sources"] = serde_json::json!(["cpp/other.cc"]);
    write_json(&root, MANIFEST, &manifest);
    expect_rejection(root, "source list");
}

/// Rejects duplicate declarations and stale source coordinates.
#[test]
fn declaration_identity_and_location_drift_is_rejected() {
    let root = copy_fixture("duplicate_symbol");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(root.join(MANIFEST)).expect("read manifest"))
            .expect("parse manifest");
    let duplicate = manifest["cpp_metadata"]["symbols"][0].clone();
    manifest["cpp_metadata"]["symbols"]
        .as_array_mut()
        .expect("symbols")
        .push(duplicate);
    write_json(&root, MANIFEST, &manifest);
    expect_rejection(root, "duplicate metadata symbol id");

    for (name, field, value, expected) in [
        (
            "source_path",
            "path",
            serde_json::json!("other.hpp"),
            "source path",
        ),
        ("source_line", "line", serde_json::json!(1), "source line"),
        (
            "source_column",
            "column",
            serde_json::json!(0),
            "source column",
        ),
        (
            "stale_source_column",
            "column",
            serde_json::json!(1),
            "source column does not identify",
        ),
    ] {
        let root = copy_fixture(name);
        let mut manifest: Value =
            serde_json::from_str(&fs::read_to_string(root.join(MANIFEST)).expect("read manifest"))
                .expect("parse manifest");
        manifest["cpp_metadata"]["symbols"][0]["source"][field] = value;
        write_json(&root, MANIFEST, &manifest);
        expect_rejection(root, expected);
    }
}

/// Rejects ambiguous compile databases and headers that collapse include modes.
#[test]
fn compile_database_and_header_boundary_drift_is_rejected() {
    let root = copy_fixture("extra_command");
    let mut database: Value = serde_json::from_str(
        &fs::read_to_string(root.join(COMPILE_DATABASE)).expect("read database"),
    )
    .expect("parse database");
    let duplicate = database[0].clone();
    database.as_array_mut().expect("commands").push(duplicate);
    write_json(&root, COMPILE_DATABASE, &database);
    expect_rejection(root, "one command");

    let root = copy_fixture("header_boundary");
    let header = fs::read_to_string(root.join(HEADER))
        .expect("read header")
        .replace("TERLAN_OPENCV_METADATA", "TERLAN_METADATA_REMOVED");
    fs::write(root.join(HEADER), header).expect("write header mutation");
    expect_rejection(root, "separate metadata and runtime");
}

/// Rejects removal of every committed input required for offline provenance.
#[test]
fn missing_offline_metadata_inputs_are_rejected() {
    for (name, path, expected) in [
        ("missing_manifest", MANIFEST, "opencv.native-binding.json"),
        (
            "missing_compile_database",
            COMPILE_DATABASE,
            "compile_commands.json",
        ),
        ("missing_header", HEADER, "cannot read adapter header"),
    ] {
        let root = copy_fixture(name);
        fs::remove_file(root.join(path)).expect("remove required metadata input");
        expect_rejection(root, expected);
    }
}

/// Accepts the generated public adapter ABI and its bounded helper.
#[test]
fn generated_aot_adapter_is_public_and_bounded() {
    validate_generated_aot_adapter(&package_root())
        .expect("generated OpenCV adapter must satisfy public AOT ABI");
}

/// Rejects legacy metadata, ABI drift, and mismatched C++ targets.
#[test]
fn generated_aot_metadata_drift_is_rejected() {
    for (name, from, to, expected) in [
        (
            "legacy_runtime",
            "[public_adapter]",
            "[runtime]",
            "legacy [runtime]",
        ),
        (
            "abi_version",
            "adapter_abi_version = 1",
            "adapter_abi_version = 2",
            "adapter_abi_version",
        ),
        (
            "frame_bound",
            "max_frame_bytes = 1048576",
            "max_frame_bytes = 1048577",
            "max_frame_bytes",
        ),
        (
            "cpp_target",
            "target = \"x86_64-unknown-linux-gnu\"",
            "target = \"aarch64-unknown-linux-gnu\"",
            "C++ metadata target differs",
        ),
    ] {
        let root = copy_generated_fixture(name);
        let path = root.join(GENERATED_BOUNDARY);
        let source = fs::read_to_string(&path).expect("read generated metadata");
        let mutated = if name == "cpp_target" {
            source.replacen(from, to, 1)
        } else {
            source.replace(from, to)
        };
        assert_ne!(source, mutated, "fixture mutation must apply");
        fs::write(path, mutated).expect("write mutated generated metadata");
        let error =
            validate_generated_aot_adapter(&root).expect_err("AOT metadata drift must fail");
        assert!(error.contains(expected), "unexpected error: {error}");
        fs::remove_dir_all(root).expect("remove generated fixture");
    }
}

/// Rejects removal of helper flow control and private runtime ABI leakage.
#[test]
fn generated_aot_helper_drift_is_rejected() {
    for (name, mutation, expected) in [
        (
            "missing_monotonicity",
            ("\"request_not_monotonic\"", "\"request_replayed\""),
            "request_not_monotonic",
        ),
        (
            "missing_chunked_transfer",
            ("struct InboundTransfer", "struct RemovedInboundTransfer"),
            "struct InboundTransfer",
        ),
        (
            "private_runtime",
            (
                "use std::collections::HashMap;",
                "use std::collections::HashMap;\n// TvmRef",
            ),
            "TvmRef",
        ),
    ] {
        let root = copy_generated_fixture(name);
        let path = root.join(GENERATED_HELPER);
        let source = fs::read_to_string(&path).expect("read generated helper");
        let mutated = source.replace(mutation.0, mutation.1);
        assert_ne!(source, mutated, "fixture mutation must apply");
        fs::write(path, mutated).expect("write mutated generated helper");
        let error = validate_generated_aot_adapter(&root).expect_err("helper ABI drift must fail");
        assert!(error.contains(expected), "unexpected error: {error}");
        fs::remove_dir_all(root).expect("remove generated fixture");
    }
}
