//! Offline validation for the pinned OpenCV package surface.

use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const EXPECTED_COMMIT: &str = "fe38fc608f6acb8b68953438a62305d8318f4fcd";
const EXPECTED_MODULES: [&str; 7] = [
    "core",
    "imgproc",
    "imgcodecs",
    "features",
    "calib3d",
    "ml",
    "dnn",
];
const EXPECTED_CPP_SHAPES: [&str; 21] = [
    "borrowed_bytes_input",
    "borrowed_float_list_input",
    "borrowed_int_list_input",
    "copied_value_record_input",
    "cross_module_resource_result",
    "deterministic_disposal",
    "external_dynamic_linkage",
    "finite_enum_input",
    "free_function",
    "immutable_method",
    "multiple_resource_types",
    "opaque_unique_resource",
    "owned_bytes_output",
    "owned_float_list_output",
    "owned_int_list_output",
    "owned_value_projection",
    "primitive_bool",
    "primitive_float",
    "primitive_int",
    "secondary_resource_borrow",
    "stable_failure_translation",
];
const EXPECTED_CPP_REJECTIONS: [&str; 10] = [
    "cpp.callback.unsupported",
    "cpp.exception.crossing",
    "cpp.inheritance.unsupported",
    "cpp.lifetime.borrowed",
    "cpp.overload.ambiguous",
    "cpp.ownership.unknown",
    "cpp.pointer.unsupported",
    "cpp.template.unspecialized",
    "cpp.type.unmapped",
    "cpp.variadic.unsupported",
];

/// Resolve the package root from this tool's manifest directory.
fn package_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("surface checker must remain under tools/surface_check")
        .to_path_buf()
}

/// Read and parse a JSON object from `path`.
fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    require(
        value.is_object(),
        format!("{}: root must be an object", path.display()),
    )?;
    Ok(value)
}

/// Read and parse a TOML document from `path` into a structured value.
fn read_toml(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    basic_toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Return success when `condition` holds or a deterministic contract error.
fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

/// Obtain a required object member.
fn object<'a>(value: &'a Value, key: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object"))
}

/// Obtain a required array member.
fn array<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))
}

/// Obtain a required non-empty string member.
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    let text = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string"))?;
    require(!text.is_empty(), format!("{key} must not be empty"))?;
    Ok(text)
}

/// Convert an array member into a stable ordered set of strings.
fn string_set(value: &Value, key: &str) -> Result<BTreeSet<String>, String> {
    array(value, key)?
        .iter()
        .map(|item| {
            item.as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} entries must be non-empty strings"))
        })
        .collect()
}

/// Validate that a commit is a lowercase, full Git SHA-1.
fn is_full_commit(commit: &str) -> bool {
    commit.len() == 40
        && commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate the immutable OpenCV source and build identity.
fn validate_pin(pin: &Value) -> Result<(), String> {
    require(
        string(pin, "schema")? == "terlan.opencv.upstream.v1",
        "invalid upstream schema",
    )?;
    let commit = string(pin, "commit")?;
    require(
        is_full_commit(commit),
        "upstream commit must be a full lowercase SHA-1",
    )?;
    require(commit == EXPECTED_COMMIT, "unexpected upstream commit")?;
    require(
        string(pin, "tag")? == "4.13.0",
        "unexpected OpenCV release tag",
    )?;
    let build = Value::Object(object(pin, "build")?.clone());
    require(
        string_set(&build, "modules")? == expected_modules(),
        "upstream lock must select only core, imgproc, imgcodecs, features, calib3d, ml, and dnn",
    )?;
    require(
        string(&build, "profile")? == "cpu-only",
        "upstream profile must be cpu-only",
    )?;
    require(
        string(&build, "language_standard")? == "c++14",
        "upstream lock must use c++14",
    )
}

/// Return the expected module set in deterministic order.
fn expected_modules() -> BTreeSet<String> {
    EXPECTED_MODULES.into_iter().map(str::to_owned).collect()
}

/// Validate the package-owned OpenCV CMake cache policy.
fn validate_cmake(source: &str) -> Result<(), String> {
    require(
        source.contains("set(BUILD_LIST \"core,imgproc,imgcodecs,features2d,calib3d,ml,dnn\""),
        "CMake policy must build only core, imgproc, imgcodecs, features2d, calib3d, ml, and dnn",
    )?;
    for option in [
        "WITH_CUDA",
        "WITH_ADE",
        "WITH_OPENCL",
        "WITH_FFMPEG",
        "WITH_GSTREAMER",
        "WITH_V4L",
        "BUILD_JAVA",
        "BUILD_opencv_gapi",
        "BUILD_opencv_python3",
    ] {
        require(
            source.contains(&format!("set({option} OFF")),
            format!("CMake policy must disable {option}"),
        )?;
    }
    require(
        source.contains("set(CMAKE_CXX_STANDARD 14"),
        "CMake policy must pin C++14",
    )
}

/// Validate declaration provenance and collect selected symbols and transitive types.
fn validate_metadata(metadata: &Value) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    require(
        string(metadata, "schema")? == "terlan.opencv.metadata.v1",
        "invalid metadata schema",
    )?;
    let upstream = Value::Object(object(metadata, "upstream")?.clone());
    require(
        string(&upstream, "commit")? == EXPECTED_COMMIT,
        "metadata commit does not match lock",
    )?;
    require(
        string_set(&upstream, "modules")? == expected_modules(),
        "metadata must contain only core, imgproc, imgcodecs, features, calib3d, ml, and dnn",
    )?;
    let compile = Value::Object(object(metadata, "compile")?.clone());
    require(
        string(&compile, "language_standard")? == "c++14",
        "metadata must use c++14",
    )?;

    let declarations = array(metadata, "declarations")?;
    require(
        !declarations.is_empty(),
        "metadata declarations must not be empty",
    )?;
    let mut ids = BTreeSet::new();
    let mut upstream_ids = BTreeSet::new();
    let mut types = BTreeSet::new();
    for (index, declaration) in declarations.iter().enumerate() {
        require(
            declaration.is_object(),
            format!("declaration {index} must be an object"),
        )?;
        let id = string(declaration, "id")?;
        let upstream_id = string(declaration, "upstream_id")?;
        require(
            ids.insert(id.to_owned()),
            format!("duplicate declaration id: {id}"),
        )?;
        require(
            upstream_ids.insert(upstream_id.to_owned()),
            format!("duplicate upstream id: {upstream_id}"),
        )?;
        require(
            EXPECTED_MODULES.contains(&string(declaration, "module")?),
            format!("{id}: module is outside the build scope"),
        )?;
        string(declaration, "signature").map_err(|error| format!("{id}: {error}"))?;
        let source = Value::Object(object(declaration, "source")?.clone());
        require(
            string(&source, "path")?.starts_with("modules/"),
            format!("{id}: invalid source path"),
        )?;
        require(
            source
                .get("line")
                .and_then(Value::as_u64)
                .is_some_and(|line| line > 0),
            format!("{id}: invalid source line"),
        )?;
        require(
            source
                .get("column")
                .and_then(Value::as_u64)
                .is_some_and(|column| column > 0),
            format!("{id}: invalid source column"),
        )?;
        let referenced = string_set(declaration, "referenced_types")?;
        require(
            !referenced.is_empty(),
            format!("{id}: referenced_types must not be empty"),
        )?;
        types.extend(referenced);
    }
    Ok((ids, types))
}

/// Validate one policy entry per declaration and every transitive C++ type.
fn validate_mapping(
    mapping: &Value,
    symbol_ids: &BTreeSet<String>,
    referenced_types: &BTreeSet<String>,
) -> Result<(), String> {
    require(
        string(mapping, "schema")? == "terlan.opencv.mapping.v1",
        "invalid mapping schema",
    )?;
    require(
        string(mapping, "upstream_commit")? == EXPECTED_COMMIT,
        "mapping commit does not match lock",
    )?;
    let mut mapped_ids = BTreeSet::new();
    for symbol in array(mapping, "symbols")? {
        let id = string(symbol, "id")?;
        require(
            mapped_ids.insert(id.to_owned()),
            format!("duplicate mapping id: {id}"),
        )?;
        require(
            matches!(
                string(symbol, "disposition")?,
                "bind" | "adapter_internal" | "reject"
            ),
            format!("{id}: invalid symbol disposition"),
        )?;
    }
    require(
        &mapped_ids == symbol_ids,
        format!(
            "symbol mapping mismatch: missing={:?}, extra={:?}",
            symbol_ids.difference(&mapped_ids).collect::<Vec<_>>(),
            mapped_ids.difference(symbol_ids).collect::<Vec<_>>()
        ),
    )?;

    let allowed = BTreeSet::from([
        "adapter_internal",
        "checked_scalar",
        "contained_exception",
        "copied_container",
        "copied_value",
        "finite_enum",
        "opaque_resource",
        "reject",
        "scalar",
    ]);
    let mut mapped_types = BTreeSet::new();
    for ty in array(mapping, "types")? {
        let cpp = string(ty, "cpp")?;
        require(
            mapped_types.insert(cpp.to_owned()),
            format!("duplicate C++ type mapping: {cpp}"),
        )?;
        require(
            allowed.contains(string(ty, "disposition")?),
            format!("{cpp}: invalid type disposition"),
        )?;
    }
    let missing = referenced_types
        .difference(&mapped_types)
        .collect::<Vec<_>>();
    require(
        missing.is_empty(),
        format!("unmapped transitive C++ types: {missing:?}"),
    )
}

/// Validate deterministic reasons for declarations intentionally outside the surface.
fn validate_skipped(skipped: &Value) -> Result<(), String> {
    require(
        string(skipped, "schema")? == "terlan.opencv.skipped.v1",
        "invalid skipped-symbol schema",
    )?;
    require(
        string(skipped, "upstream_commit")? == EXPECTED_COMMIT,
        "skipped-symbol commit does not match lock",
    )?;
    let mut upstream_ids = BTreeSet::new();
    for symbol in array(skipped, "symbols")? {
        let upstream_id = string(symbol, "upstream_id")?;
        require(
            upstream_ids.insert(upstream_id.to_owned()),
            format!("duplicate skipped upstream id: {upstream_id}"),
        )?;
        string(symbol, "reason").map_err(|error| format!("{upstream_id}: {error}"))?;
    }
    require(
        !upstream_ids.is_empty(),
        "skipped-symbol inventory must not be empty",
    )
}

/// Validate real-package evidence for every admitted generic C++ binding shape.
fn validate_cpp_shape_coverage(
    root: &Path,
    coverage: &Value,
    native_manifest: &Value,
) -> Result<(), String> {
    require(
        string(coverage, "schema")? == "terlan.opencv.cpp-binding-shapes.v1",
        "invalid C++ binding-shape schema",
    )?;
    require(
        string(coverage, "upstream_commit")? == EXPECTED_COMMIT,
        "C++ binding-shape commit does not match the OpenCV pin",
    )?;
    require(
        string(coverage, "compiler_gate")? == "make cpp-binding-generator-check",
        "C++ binding-shape compiler gate drift",
    )?;

    let operations = array(native_manifest, "modules")?
        .iter()
        .flat_map(|module| {
            module
                .get("functions")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .map(|function| string(function, "operation").map(str::to_owned))
        .collect::<Result<BTreeSet<_>, _>>()?;

    let mut shapes = BTreeSet::new();
    for shape in array(coverage, "shapes")? {
        let id = string(shape, "id")?;
        require(
            shapes.insert(id.to_owned()),
            format!("duplicate C++ binding shape: {id}"),
        )?;
        string(shape, "rationale").map_err(|error| format!("{id}: {error}"))?;
        let shape_operations = string_set(shape, "operations")?;
        require(
            !shape_operations.is_empty(),
            format!("{id}: at least one OpenCV operation is required"),
        )?;
        for operation in shape_operations {
            require(
                operations.contains(&operation),
                format!("{id}: unknown OpenCV operation `{operation}`"),
            )?;
        }
        let evidence = string_set(shape, "evidence")?;
        require(
            !evidence.is_empty(),
            format!("{id}: executable evidence is required"),
        )?;
        for path in evidence {
            require(
                root.join(&path).is_file(),
                format!("{id}: evidence path does not exist: {path}"),
            )?;
        }
    }
    require(
        shapes == EXPECTED_CPP_SHAPES.into_iter().map(str::to_owned).collect(),
        "generic C++ binding-shape coverage is incomplete or contains unknown shapes",
    )?;

    let mut rejections = BTreeSet::new();
    for rejection in array(coverage, "rejections")? {
        let family = string(rejection, "family")?;
        require(
            rejections.insert(family.to_owned()),
            format!("duplicate C++ rejection family: {family}"),
        )?;
        string(rejection, "rationale").map_err(|error| format!("{family}: {error}"))?;
    }
    require(
        rejections
            == EXPECTED_CPP_REJECTIONS
                .into_iter()
                .map(str::to_owned)
                .collect(),
        "generic C++ rejection-family coverage is incomplete or contains unknown families",
    )
}

/// Run every offline OCV0.1 contract validation.
fn check_surface(root: &Path) -> Result<(), String> {
    let pin = read_toml(&root.join("upstream/opencv.lock.toml"))?;
    let metadata = read_json(&root.join("bindings/opencv.metadata.json"))?;
    let mapping = read_toml(&root.join("bindings/opencv.mapping.toml"))?;
    let skipped = read_json(&root.join("bindings/skipped-symbols.json"))?;
    let shape_coverage = read_toml(&root.join("bindings/opencv.cpp-binding-shapes.toml"))?;
    let native_manifest = read_json(&root.join("bindings/opencv.native-binding.json"))?;
    let cmake = fs::read_to_string(root.join("cmake/opencv-cpu.cmake"))
        .map_err(|error| format!("cmake/opencv-cpu.cmake: {error}"))?;
    validate_pin(&pin)?;
    validate_cmake(&cmake)?;
    let (symbols, types) = validate_metadata(&metadata)?;
    validate_mapping(&mapping, &symbols, &types)?;
    validate_skipped(&skipped)?;
    validate_cpp_shape_coverage(root, &shape_coverage, &native_manifest)
}

/// Execute the surface checker as a command-line gate.
fn main() {
    if let Err(error) = check_surface(&package_root()) {
        eprintln!("opencv-surface-check: {error}");
        std::process::exit(1);
    }
    println!(
        "opencv-surface-check: OpenCV 4.13.0 scope is pinned, selected declarations are mapped, and all 21 admitted generic C++ binding shapes are proven"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One named mutation and the stable diagnostic it must trigger.
    type MetadataMutation = (&'static str, fn(&mut Value), &'static str);

    /// Parses one committed JSON surface input.
    fn committed_json(path: &str) -> Value {
        read_json(&package_root().join(path)).expect("parse committed JSON")
    }

    /// Parses one committed TOML surface input.
    fn committed_toml(path: &str) -> Value {
        read_toml(&package_root().join(path)).expect("parse committed TOML")
    }

    /// Requires a deterministic validation failure containing `expected`.
    fn expect_error(result: Result<(), String>, expected: &str) {
        let error = result.expect_err("mutated surface must fail");
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    /// Verify full lowercase commits and reject truncated or uppercase identities.
    #[test]
    fn commit_validation_is_strict() {
        assert!(is_full_commit(EXPECTED_COMMIT));
        assert!(!is_full_commit(&EXPECTED_COMMIT[..39]));
        assert!(!is_full_commit(&EXPECTED_COMMIT.to_uppercase()));
    }

    /// Verify the CMake gate accepts the committed policy.
    #[test]
    fn committed_cmake_policy_is_valid() {
        let source = fs::read_to_string(package_root().join("cmake/opencv-cpu.cmake")).unwrap();
        validate_cmake(&source).unwrap();
    }

    /// Verify the full committed surface remains internally consistent.
    #[test]
    fn committed_surface_is_valid() {
        check_surface(&package_root()).unwrap();
    }

    /// Reject every mutable part of the pinned upstream identity.
    #[test]
    fn upstream_pin_drift_is_rejected() {
        for (field, value, expected) in [
            (
                "commit",
                serde_json::json!("0000000000000000000000000000000000000000"),
                "commit",
            ),
            ("tag", serde_json::json!("4.12.0"), "tag"),
        ] {
            let mut pin = committed_toml("upstream/opencv.lock.toml");
            pin[field] = value;
            expect_error(validate_pin(&pin), expected);
        }

        let mut pin = committed_toml("upstream/opencv.lock.toml");
        pin["build"]["modules"] = serde_json::json!(["core"]);
        expect_error(
            validate_pin(&pin),
            "core, imgproc, imgcodecs, features, calib3d, ml, and dnn",
        );

        let mut pin = committed_toml("upstream/opencv.lock.toml");
        pin["build"]["language_standard"] = serde_json::json!("c++11");
        expect_error(validate_pin(&pin), "c++14");
    }

    /// Reject module, declaration, source, and referenced-type metadata drift.
    #[test]
    fn reviewed_upstream_metadata_drift_is_rejected() {
        let mutations: [MetadataMutation; 2] = [
            (
                "commit",
                |metadata: &mut Value| metadata["upstream"]["commit"] = serde_json::json!("other"),
                "commit",
            ),
            (
                "language",
                |metadata: &mut Value| {
                    metadata["compile"]["language_standard"] = serde_json::json!("c++11")
                },
                "c++14",
            ),
        ];
        for (_name, mutate, expected) in mutations {
            let mut metadata = committed_json("bindings/opencv.metadata.json");
            mutate(&mut metadata);
            expect_error(validate_metadata(&metadata).map(|_| ()), expected);
        }

        let mut metadata = committed_json("bindings/opencv.metadata.json");
        let duplicate = metadata["declarations"][0].clone();
        metadata["declarations"]
            .as_array_mut()
            .expect("declarations")
            .push(duplicate);
        expect_error(
            validate_metadata(&metadata).map(|_| ()),
            "duplicate declaration",
        );

        for (field, value, expected) in [
            (
                "path",
                serde_json::json!("include/other.hpp"),
                "source path",
            ),
            ("line", serde_json::json!(0), "source line"),
            ("column", serde_json::json!(0), "source column"),
        ] {
            let mut metadata = committed_json("bindings/opencv.metadata.json");
            metadata["declarations"][0]["source"][field] = value;
            expect_error(validate_metadata(&metadata).map(|_| ()), expected);
        }

        let mut metadata = committed_json("bindings/opencv.metadata.json");
        metadata["declarations"][0]["referenced_types"] = serde_json::json!([]);
        expect_error(validate_metadata(&metadata).map(|_| ()), "referenced_types");
    }

    /// Reject missing, extra, duplicate, and invalid mapping decisions.
    #[test]
    fn mapping_completeness_drift_is_rejected() {
        let metadata = committed_json("bindings/opencv.metadata.json");
        let (symbols, types) = validate_metadata(&metadata).expect("valid metadata");

        let mut mapping = committed_toml("bindings/opencv.mapping.toml");
        mapping["symbols"].as_array_mut().expect("symbols").pop();
        expect_error(
            validate_mapping(&mapping, &symbols, &types),
            "symbol mapping mismatch",
        );

        let mut mapping = committed_toml("bindings/opencv.mapping.toml");
        let duplicate = mapping["symbols"][0].clone();
        mapping["symbols"]
            .as_array_mut()
            .expect("symbols")
            .push(duplicate);
        expect_error(
            validate_mapping(&mapping, &symbols, &types),
            "duplicate mapping id",
        );

        let mut mapping = committed_toml("bindings/opencv.mapping.toml");
        mapping["symbols"][0]["disposition"] = serde_json::json!("invented");
        expect_error(
            validate_mapping(&mapping, &symbols, &types),
            "invalid symbol disposition",
        );

        let mut mapping = committed_toml("bindings/opencv.mapping.toml");
        mapping["types"]
            .as_array_mut()
            .expect("types")
            .retain(|ty| ty["cpp"] != "cv::Mat");
        expect_error(
            validate_mapping(&mapping, &symbols, &types),
            "unmapped transitive",
        );

        let mut mapping = committed_toml("bindings/opencv.mapping.toml");
        mapping["types"][0]["disposition"] = serde_json::json!("invented");
        expect_error(
            validate_mapping(&mapping, &symbols, &types),
            "invalid type disposition",
        );
    }

    /// Reject skipped-symbol drift and incomplete reasons.
    #[test]
    fn skipped_symbol_inventory_drift_is_rejected() {
        let mut skipped = committed_json("bindings/skipped-symbols.json");
        skipped["upstream_commit"] = serde_json::json!("other");
        expect_error(validate_skipped(&skipped), "commit");

        let mut skipped = committed_json("bindings/skipped-symbols.json");
        skipped["symbols"] = serde_json::json!([]);
        expect_error(validate_skipped(&skipped), "must not be empty");

        let mut skipped = committed_json("bindings/skipped-symbols.json");
        let duplicate = skipped["symbols"][0].clone();
        skipped["symbols"]
            .as_array_mut()
            .expect("symbols")
            .push(duplicate);
        expect_error(validate_skipped(&skipped), "duplicate skipped upstream id");

        let mut skipped = committed_json("bindings/skipped-symbols.json");
        skipped["symbols"][0]["reason"] = serde_json::json!("");
        expect_error(validate_skipped(&skipped), "reason");
    }

    /// Reject missing real-package shape evidence and unknown operations.
    #[test]
    fn cpp_binding_shape_coverage_is_executable_and_complete() {
        let native_manifest = committed_json("bindings/opencv.native-binding.json");
        let coverage = committed_toml("bindings/opencv.cpp-binding-shapes.toml");
        validate_cpp_shape_coverage(&package_root(), &coverage, &native_manifest).unwrap();

        let mut missing_shape = coverage.clone();
        missing_shape["shapes"]
            .as_array_mut()
            .expect("shapes")
            .pop();
        expect_error(
            validate_cpp_shape_coverage(&package_root(), &missing_shape, &native_manifest),
            "coverage is incomplete",
        );

        let mut unknown_operation = coverage;
        unknown_operation["shapes"][0]["operations"] =
            serde_json::json!(["opencv.missing.operation"]);
        expect_error(
            validate_cpp_shape_coverage(&package_root(), &unknown_operation, &native_manifest),
            "unknown OpenCV operation",
        );
    }

    /// Reject scope expansion and weakened CMake language policy.
    #[test]
    fn cmake_policy_drift_is_rejected() {
        let source = fs::read_to_string(package_root().join("cmake/opencv-cpu.cmake"))
            .expect("read CMake policy");
        for (from, to, expected) in [
            (
                "core,imgproc,imgcodecs,features2d,calib3d,ml,dnn",
                "core,imgproc,imgcodecs,features2d,calib3d,ml",
                "core, imgproc, imgcodecs, features2d, calib3d, ml, and dnn",
            ),
            ("set(WITH_CUDA OFF", "set(WITH_CUDA ON", "WITH_CUDA"),
            ("CMAKE_CXX_STANDARD 14", "CMAKE_CXX_STANDARD 11", "C++14"),
        ] {
            expect_error(validate_cmake(&source.replace(from, to)), expected);
        }
    }
}
