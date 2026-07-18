//! Offline and live validation for normalized OpenCV adapter metadata.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

const MANIFEST: &str = "bindings/opencv.native-binding.json";
const COMPILE_DATABASE: &str = "bindings/cpp/compile_commands.json";
const HEADER: &str = "bindings/cpp/opencv_adapter.hpp";
const PRIMARY_HEADER_SOURCE: &str = "cpp/opencv_adapter.hpp";

/// Resolves the package root from this checker's manifest directory.
pub fn package_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("metadata checker must remain under tools/metadata_check")
        .to_path_buf()
}

/// Validates committed metadata, compile provenance, and source locations offline.
pub fn validate_offline(root: &Path) -> Result<(), String> {
    let manifest = read_json(&root.join(MANIFEST))?;
    let build = object(&manifest, "build")?;
    require(
        array(build, "adapter_headers")?.as_slice()
            == [
                Value::String("cpp/opencv_contrast_validation.hpp".into()),
                Value::String("cpp/opencv_drawing_validation.hpp".into()),
                Value::String("cpp/opencv_dnn.hpp".into()),
                Value::String("cpp/opencv_dnn_detection_validation.hpp".into()),
                Value::String("cpp/opencv_dnn_internal.hpp".into()),
                Value::String("cpp/opencv_dnn_validation.hpp".into()),
                Value::String("cpp/opencv_failure.hpp".into()),
                Value::String("cpp/opencv_feature_internal.hpp".into()),
                Value::String("cpp/opencv_internal.hpp".into()),
                Value::String("cpp/opencv_mask_analysis.hpp".into()),
                Value::String("cpp/opencv_ml.hpp".into()),
                Value::String("cpp/opencv_ml_internal.hpp".into()),
                Value::String("cpp/opencv_region_internal.hpp".into()),
                Value::String("cpp/opencv_validation.hpp".into()),
            ],
        "adapter header list drift",
    )?;
    let metadata = object(&manifest, "cpp_metadata")?;
    require(
        string(metadata, "schema")? == "terlan.cpp.metadata.v1",
        "unexpected C++ metadata schema",
    )?;
    let producer = map_object(metadata, "producer")?;
    require(
        string(producer, "name")? == "clang-libtooling",
        "metadata producer must be clang-libtooling",
    )?;
    require(
        string(producer, "format")? == "normalized-ast-json",
        "metadata format drift",
    )?;
    require(
        string(metadata, "namespace")? == "terlan_opencv",
        "metadata namespace drift",
    )?;
    require(
        string(metadata, "header")? == "cpp/opencv_adapter.hpp",
        "metadata header drift",
    )?;
    let sources = array(metadata, "sources")?;
    require(
        sources.as_slice()
            == [
                Value::String("cpp/opencv_metadata.cc".into()),
                Value::String("cpp/opencv_adapter.cc".into()),
                Value::String("cpp/opencv_arithmetic.cc".into()),
                Value::String("cpp/opencv_binary_features.cc".into()),
                Value::String("cpp/opencv_blobs.cc".into()),
                Value::String("cpp/opencv_calib3d.cc".into()),
                Value::String("cpp/opencv_contours.cc".into()),
                Value::String("cpp/opencv_contrast.cc".into()),
                Value::String("cpp/opencv_corners.cc".into()),
                Value::String("cpp/opencv_drawing.cc".into()),
                Value::String("cpp/opencv_dnn.cc".into()),
                Value::String("cpp/opencv_dnn_inference.cc".into()),
                Value::String("cpp/opencv_dnn_segmentation.cc".into()),
                Value::String("cpp/opencv_feature_internal.cc".into()),
                Value::String("cpp/opencv_features.cc".into()),
                Value::String("cpp/opencv_filters.cc".into()),
                Value::String("cpp/opencv_float_features.cc".into()),
                Value::String("cpp/opencv_hough.cc".into()),
                Value::String("cpp/opencv_masks.cc".into()),
                Value::String("cpp/opencv_mser.cc".into()),
                Value::String("cpp/opencv_ml.cc".into()),
                Value::String("cpp/opencv_ml_clustering.cc".into()),
                Value::String("cpp/opencv_ml_neural.cc".into()),
                Value::String("cpp/opencv_region_internal.cc".into()),
                Value::String("cpp/opencv_statistics.cc".into()),
            ],
        "metadata source list drift",
    )?;

    let database = read_json(&root.join(COMPILE_DATABASE))?;
    let commands = database
        .as_array()
        .ok_or_else(|| "compile database root must be an array".to_string())?;
    require(
        commands.len() == 1,
        "compile database must contain one command",
    )?;
    let command = commands[0]
        .as_object()
        .ok_or_else(|| "compile command must be an object".to_string())?;
    let compile = map_object(metadata, "compile")?;
    require(
        string(compile, "language_standard")? == "c++14",
        "metadata language standard drift",
    )?;
    require(
        compile.get("arguments") == command.get("arguments"),
        "metadata arguments differ from compile_commands.json",
    )?;
    require(
        command.get("file").and_then(Value::as_str) == Some("opencv_metadata.cc"),
        "compile database must select opencv_metadata.cc",
    )?;

    let header = fs::read_to_string(root.join(HEADER))
        .map_err(|error| format!("cannot read adapter header: {error}"))?;
    require(
        header.contains("TERLAN_OPENCV_METADATA") && header.contains("rust/cxx.h"),
        "adapter header must separate metadata and runtime CXX includes",
    )?;
    let mut declaration_sources = BTreeMap::from([(PRIMARY_HEADER_SOURCE.to_string(), header)]);
    for path in array(build, "adapter_headers")? {
        let path = path
            .as_str()
            .ok_or_else(|| "adapter header path must be a string".to_string())?;
        let source = fs::read_to_string(root.join("bindings").join(path))
            .map_err(|error| format!("cannot read adapter header {path}: {error}"))?;
        declaration_sources.insert(path.to_string(), source);
    }
    let symbols = array(metadata, "symbols")?;
    require(
        !symbols.is_empty(),
        "metadata symbol list must not be empty",
    )?;
    let mut symbol_ids = BTreeSet::new();
    for symbol in symbols {
        let symbol = symbol
            .as_object()
            .ok_or_else(|| "metadata symbol must be an object".to_string())?;
        let id = string(symbol, "id")?;
        require(
            symbol_ids.insert(id),
            format!("duplicate metadata symbol id `{id}`"),
        )?;
        let cpp_name = string(symbol, "cpp_name")?;
        let source = map_object(symbol, "source")?;
        let source_path = string(source, "path")?;
        let source_text = declaration_sources
            .get(source_path)
            .ok_or_else(|| format!("{id}: source path is not a reviewed adapter header"))?;
        let line = source
            .get("line")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{id}: source line must be positive"))?
            as usize;
        let column = source
            .get("column")
            .and_then(Value::as_u64)
            .filter(|column| *column > 0)
            .ok_or_else(|| format!("{id}: source column must be positive"))?
            as usize;
        let source_line = source_text.lines().nth(line.saturating_sub(1));
        require(
            line > 0 && source_line.is_some_and(|text| text.contains(cpp_name)),
            format!("{id}: source line does not contain `{cpp_name}`"),
        )?;
        require(
            source_line
                .and_then(|text| text.get(column - 1..))
                .is_some_and(|text| text.starts_with(cpp_name)),
            format!("{id}: source column does not identify `{cpp_name}`"),
        )?;
        require(
            !string(symbol, "documentation")?.starts_with("Undocumented"),
            format!("{id}: declaration is undocumented"),
        )?;
    }
    Ok(())
}

/// Reproduces metadata with the maintained extractor and compares declarations.
pub fn validate_live(root: &Path, extractor: &Path) -> Result<(), String> {
    validate_offline(root)?;
    require(
        extractor.is_file(),
        "maintained metadata extractor is missing",
    )?;
    let cpp = root.join("bindings/cpp");
    require(
        root.join("upstream/install/include/opencv4/opencv2/core.hpp")
            .is_file(),
        "pinned OpenCV install is required for live extraction",
    )?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let live = std::env::temp_dir().join(format!("terlan-opencv-metadata-{nonce}"));
    fs::create_dir_all(&live)
        .map_err(|error| format!("cannot create live metadata directory: {error}"))?;
    let output = live.join("metadata.json");
    let mut database = read_json(&root.join(COMPILE_DATABASE))?;
    for command in database
        .as_array_mut()
        .ok_or_else(|| "compile database root must be an array".to_string())?
    {
        command
            .as_object_mut()
            .ok_or_else(|| "compile command must be an object".to_string())?
            .insert(
                "directory".to_string(),
                Value::String(cpp.to_string_lossy().into_owned()),
            );
    }
    fs::write(
        live.join("compile_commands.json"),
        serde_json::to_vec_pretty(&database)
            .map_err(|error| format!("cannot encode live compilation database: {error}"))?,
    )
    .map_err(|error| format!("cannot write live compilation database: {error}"))?;
    let mut invocation = Command::new(extractor);
    invocation
        .current_dir(&cpp)
        .arg("-p")
        .arg(&live)
        .arg("--output")
        .arg(&output)
        .arg("--header")
        .arg("opencv_adapter.hpp")
        .arg("--header")
        .arg("opencv_ml.hpp")
        .arg("--header")
        .arg("opencv_dnn.hpp")
        .arg("--namespace")
        .arg("terlan_opencv")
        .arg(cpp.join("opencv_metadata.cc"));
    if let Some(include_path) = cpp_include_path()? {
        invocation.env("CPLUS_INCLUDE_PATH", include_path);
    }
    let status = invocation
        .status()
        .map_err(|error| format!("cannot start maintained extractor: {error}"))?;
    if !status.success() {
        let _ = fs::remove_dir_all(&live);
        return Err("maintained extractor failed".to_string());
    }

    let manifest = read_json(&root.join(MANIFEST))?;
    let expected = Value::Object(object(&manifest, "cpp_metadata")?.clone());
    let actual = read_json(&output)?;
    let expected = normalize_metadata(expected, false)?;
    let actual = normalize_metadata(actual, true)?;
    if actual != expected {
        fs::write(
            live.join("expected-normalized.json"),
            serde_json::to_vec_pretty(&expected)
                .map_err(|error| format!("cannot encode expected metadata: {error}"))?,
        )
        .map_err(|error| format!("cannot write expected metadata: {error}"))?;
        fs::write(
            live.join("actual-normalized.json"),
            serde_json::to_vec_pretty(&actual)
                .map_err(|error| format!("cannot encode actual metadata: {error}"))?,
        )
        .map_err(|error| format!("cannot write actual metadata: {error}"))?;
        return Err(format!(
            "live Clang declarations, types, or source locations differ from committed metadata; inspect {}",
            live.display()
        ));
    }
    let _ = fs::remove_dir_all(&live);
    Ok(())
}

/// Returns host C++ include roots for LibTooling when Clang and GCC differ.
fn cpp_include_path() -> Result<Option<OsString>, String> {
    if let Some(value) = std::env::var_os("CPLUS_INCLUDE_PATH") {
        return Ok(Some(value));
    }
    let output = match Command::new("g++")
        .args(["-E", "-x", "c++", "-v", "-"])
        .stdin(Stdio::null())
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot inspect host C++ include search: {error}")),
    };
    require(output.status.success(), "host C++ include probe failed")?;
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    let mut collecting = false;
    let mut roots = Vec::new();
    for line in diagnostics.lines() {
        let line = line.trim();
        if line == "#include <...> search starts here:" {
            collecting = true;
        } else if line == "End of search list." {
            break;
        } else if collecting {
            let path = Path::new(line);
            if path.is_dir() && path.components().any(|part| part.as_os_str() == "c++") {
                roots.push(path.to_path_buf());
            }
        }
    }
    require(!roots.is_empty(), "host C++ compiler reported no include roots")?;
    std::env::join_paths(roots)
        .map(Some)
        .map_err(|error| format!("cannot encode host C++ include search: {error}"))
}

/// Normalizes installation-specific and package-relative metadata fields.
fn normalize_metadata(mut metadata: Value, rebase_paths: bool) -> Result<Value, String> {
    let root = metadata
        .as_object_mut()
        .ok_or_else(|| "metadata root must be an object".to_string())?;
    object_mut(root, "producer")?.remove("version");
    object_mut(root, "compile")?.insert(
        "include_roots".to_string(),
        Value::Array(vec![Value::String(".".into())]),
    );
    if rebase_paths {
        root.insert(
            "header".to_string(),
            Value::String("cpp/opencv_adapter.hpp".into()),
        );
    }
    // Declaration parity uses one header-owning translation unit; other
    // metadata sources remain independently checked build inputs offline.
    root.insert(
        "sources".to_string(),
        Value::Array(vec![Value::String("cpp/opencv_metadata.cc".into())]),
    );
    let symbols = root
        .get_mut("symbols")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "metadata symbols must be an array".to_string())?;
    for symbol in symbols.iter_mut() {
        let symbol = symbol
            .as_object_mut()
            .ok_or_else(|| "metadata symbol must be an object".to_string())?;
        symbol.remove("id");
        symbol.remove("annotations");
        if symbol.get("variadic") == Some(&Value::Bool(false)) {
            symbol.remove("variadic");
        }
        if rebase_paths {
            let source = object_mut(symbol, "source")?;
            let path = string(source, "path")?;
            source.insert("path".to_string(), Value::String(format!("cpp/{path}")));
        }
        normalize_symbol_types(symbol)?;
    }
    symbols.sort_by_key(symbol_key);
    Ok(metadata)
}

/// Normalizes target aliases without discarding declared type spellings.
fn normalize_symbol_types(symbol: &mut Map<String, Value>) -> Result<(), String> {
    if let Some(returns) = symbol.get_mut("returns") {
        normalize_type(returns)?;
    }
    for collection in ["parameters", "fields"] {
        if let Some(items) = symbol.get_mut(collection).and_then(Value::as_array_mut) {
            if collection == "fields" {
                items.retain(|item| {
                    item.get("access")
                        .and_then(Value::as_str)
                        .is_none_or(|access| access == "public" || access == "none")
                });
                for item in items.iter_mut() {
                    if let Some(item) = item.as_object_mut() {
                        item.remove("access");
                    }
                }
            }
            for item in items {
                let ty = item
                    .get_mut("ty")
                    .ok_or_else(|| format!("{collection} entry has no type"))?;
                normalize_type(ty)?;
            }
        }
    }
    if symbol
        .get("fields")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty)
    {
        symbol.remove("fields");
    }
    let receiver = symbol
        .get("receiver")
        .and_then(Value::as_str)
        .and_then(|receiver| receiver.strip_prefix("terlan_opencv::"))
        .map(str::to_string);
    if let Some(receiver) = receiver {
        symbol.insert("receiver".to_string(), Value::String(receiver));
    }
    Ok(())
}

/// Canonicalizes target-specific fixed-width aliases and namespace qualification.
fn normalize_type(value: &mut Value) -> Result<(), String> {
    let ty = value
        .as_object_mut()
        .ok_or_else(|| "type metadata must be an object".to_string())?;
    let rendered_spelling = string(ty, "spelling")?.replace("class ", "");
    ty.insert(
        "spelling".to_string(),
        Value::String(rendered_spelling.clone()),
    );
    let spelling = rendered_spelling.replace(' ', "");
    let canonical = match spelling.as_str() {
        "std::int64_t" => "std::int64_t".to_string(),
        "rust::Slice<conststd::uint8_t>" => "rust::Slice<const std::uint8_t>".to_string(),
        "rust::Slice<conststd::int64_t>" => "rust::Slice<const std::int64_t>".to_string(),
        "std::unique_ptr<std::vector<std::uint8_t>>" => {
            "std::unique_ptr<std::vector<std::uint8_t>>".to_string()
        }
        "std::unique_ptr<std::vector<std::int64_t>>" => {
            "std::unique_ptr<std::vector<std::int64_t>>".to_string()
        }
        _ => string(ty, "canonical")?.replace("terlan_opencv::", ""),
    };
    ty.insert("canonical".to_string(), Value::String(canonical));
    Ok(())
}

/// Returns a deterministic declaration comparison key.
fn symbol_key(value: &Value) -> String {
    let object = value.as_object();
    format!(
        "{}:{}:{}",
        object
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        object
            .and_then(|value| value.get("cpp_name"))
            .and_then(Value::as_str)
            .unwrap_or(""),
        object
            .and_then(|value| value.get("source"))
            .and_then(|value| value.get("line"))
            .and_then(Value::as_u64)
            .unwrap_or_default()
    )
}

/// Reads one UTF-8 JSON document.
fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// Returns a required immutable JSON object member.
fn object<'a>(value: &'a Value, key: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object"))
}

/// Returns a required immutable object member from an existing JSON object.
fn map_object<'a>(
    value: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{key} must be an object"))
}

/// Returns a required mutable JSON object member.
fn object_mut<'a>(
    value: &'a mut Map<String, Value>,
    key: &str,
) -> Result<&'a mut Map<String, Value>, String> {
    value
        .get_mut(key)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| format!("{key} must be an object"))
}

/// Returns a required JSON array member.
fn array<'a>(value: &'a Map<String, Value>, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be an array"))
}

/// Returns a required non-empty JSON string member.
fn string<'a>(value: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

/// Returns success when a package contract condition holds.
fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
