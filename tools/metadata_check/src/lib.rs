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
const GENERATED_BOUNDARY: &str = "generated/native/terlan-native.toml";
const GENERATED_HELPER: &str = "generated/native/rust/src/bin/native_boundary_helper.rs";

/// Validates the generated OpenCV package against the public AOT adapter ABI.
pub fn validate_generated_aot_adapter(root: &Path) -> Result<(), String> {
    let metadata_path = root.join(GENERATED_BOUNDARY);
    let metadata_text = fs::read_to_string(&metadata_path)
        .map_err(|error| format!("{}: {error}", metadata_path.display()))?;
    let metadata = metadata_text
        .parse::<toml::Value>()
        .map_err(|error| format!("{}: {error}", metadata_path.display()))?;
    require(
        metadata.get("runtime").is_none(),
        "legacy [runtime] adapter metadata must not survive AOT generation",
    )?;
    let public = toml_table(&metadata, "public_adapter")?;
    require_toml_integer(public, "adapter_abi_version", 1)?;
    require_toml_text(public, "target", "x86_64-unknown-linux-gnu")?;
    require_toml_text(public, "calling_convention", "system_v")?;
    require_toml_text(public, "execution_context", "explicit")?;
    require_toml_text(public, "ownership", "opaque_handles")?;
    require_toml_text(public, "capability_lifetimes", "explicit")?;
    require_toml_text(public, "resource_lifetimes", "execution_context_scoped")?;
    require_toml_integer(public, "max_frame_bytes", 1_048_576)?;
    require_toml_integer(public, "max_transfer_bytes", 16_777_216)?;
    require_toml_text(public, "status_model", "status_values")?;
    require_toml_text(public, "callback_reentrancy", "forbidden")?;
    require_toml_text(public, "async_completion", "single_shot")?;
    require_toml_text(public, "handle_scope", "worker_random_256")?;
    require_toml_text(public, "cross_owner", "reject")?;
    require_toml_bool(public, "raw_pointers", false)?;
    require_toml_bool(public, "exceptions_cross_boundary", false)?;
    require_toml_bool(public, "native_failure_payloads", false)?;

    let cpp = toml_table(&metadata, "cpp_metadata")?;
    let cpp_target = toml_text(cpp, "target")?;
    require(
        cpp_target == toml_text(public, "target")?,
        "C++ metadata target differs from public adapter target",
    )?;

    let helper_path = root.join(GENERATED_HELPER);
    let helper = fs::read_to_string(&helper_path)
        .map_err(|error| format!("{}: {error}", helper_path.display()))?;
    for required in [
        "const MAX_ADAPTER_FRAME_BYTES: usize = 1048576;",
        "const MAX_ADAPTER_TRANSFER_BYTES: usize = 16777216;",
        ".take((MAX_ADAPTER_FRAME_BYTES + 1) as u64)",
        "struct InboundTransfer",
        "\"reply_chunk {request_id} {index} {final_chunk}",
        "\"transfer_too_large\"",
        "last_request_id: Option<u64>",
        "\"request_not_monotonic\"",
        "#![forbid(unsafe_code)]",
    ] {
        require(
            helper.contains(required),
            format!("generated helper omits public adapter invariant `{required}`"),
        )?;
    }
    let private_runtime_terms = [
        "TvmRef",
        "ManagedExecutionRuntime",
        "PureNativeExecutionShard",
        "Cranelift",
        "actor_heap",
        "actor heap",
        "continuation",
        "thread_identity",
        "thread identity",
    ];
    for term in private_runtime_terms {
        require(
            !metadata_text.contains(term) && !helper.contains(term),
            format!("generated public adapter leaks private AOT runtime term `{term}`"),
        )?;
    }
    Ok(())
}

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
                Value::String("cpp/opencv_metadata_rust.hpp".into()),
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
                Value::String("cpp/opencv_core_arrays.cc".into()),
                Value::String("cpp/opencv_core_linear_algebra.cc".into()),
                Value::String("cpp/opencv_core_runtime.cc".into()),
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
    let actual = extract_live_metadata(root, extractor)?;
    let manifest = read_json(&root.join(MANIFEST))?;
    let expected = Value::Object(object(&manifest, "cpp_metadata")?.clone());
    let expected = normalize_metadata(expected, false)?;
    let actual = normalize_metadata(actual, true)?;
    if actual != expected {
        return Err(
            "live Clang declarations, types, or source locations differ from committed metadata"
                .to_string(),
        );
    }
    Ok(())
}

/// Refreshes Clang-owned adapter metadata while preserving package-owned symbol IDs.
pub fn refresh_metadata(root: &Path, extractor: &Path) -> Result<(), String> {
    let mut actual = extract_live_metadata(root, extractor)?;
    let mut manifest = read_json(&root.join(MANIFEST))?;
    let manifest_root = manifest
        .as_object_mut()
        .ok_or_else(|| "binding manifest root must be an object".to_string())?;
    let existing = manifest_root
        .get("cpp_metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| "binding manifest cpp_metadata must be an object".to_string())?
        .clone();
    let existing_symbols = existing
        .get("symbols")
        .and_then(Value::as_array)
        .ok_or_else(|| "committed metadata symbols must be an array".to_string())?;
    let mut identities = BTreeMap::new();
    for symbol in existing_symbols {
        let symbol = symbol
            .as_object()
            .ok_or_else(|| "committed metadata symbol must be an object".to_string())?;
        identities.insert(
            string(symbol, "overload_set")?.to_string(),
            (
                string(symbol, "id")?.to_string(),
                symbol
                    .get("annotations")
                    .cloned()
                    .unwrap_or_else(|| Value::Array(Vec::new())),
            ),
        );
    }

    let actual_root = actual
        .as_object_mut()
        .ok_or_else(|| "live metadata root must be an object".to_string())?;
    let producer_version = existing
        .get("producer")
        .and_then(Value::as_object)
        .and_then(|producer| producer.get("version"))
        .cloned()
        .ok_or_else(|| "committed metadata producer version is missing".to_string())?;
    object_mut(actual_root, "producer")?.insert("version".to_string(), producer_version);
    object_mut(actual_root, "compile")?.insert(
        "include_roots".to_string(),
        Value::Array(vec![Value::String(".".to_string())]),
    );
    actual_root.insert(
        "header".to_string(),
        Value::String(PRIMARY_HEADER_SOURCE.to_string()),
    );
    actual_root.insert(
        "sources".to_string(),
        existing
            .get("sources")
            .cloned()
            .ok_or_else(|| "committed metadata source list is missing".to_string())?,
    );
    let symbols = actual_root
        .get_mut("symbols")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "live metadata symbols must be an array".to_string())?;
    for symbol in symbols.iter_mut() {
        let symbol = symbol
            .as_object_mut()
            .ok_or_else(|| "live metadata symbol must be an object".to_string())?;
        let overload = string(symbol, "overload_set")?.to_string();
        let (id, annotations) = identities
            .get(&overload)
            .cloned()
            .unwrap_or((stable_symbol_id(symbol)?, Value::Array(Vec::new())));
        symbol.insert("id".to_string(), Value::String(id));
        symbol.insert("annotations".to_string(), annotations);
        let source = object_mut(symbol, "source")?;
        let path = string(source, "path")?.to_string();
        source.insert("path".to_string(), Value::String(format!("cpp/{path}")));
    }
    symbols.sort_by_key(|symbol| {
        symbol
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    });
    manifest_root.insert("cpp_metadata".to_string(), actual);
    let encoded = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("cannot encode refreshed binding manifest: {error}"))?;
    fs::write(root.join(MANIFEST), encoded)
        .map_err(|error| format!("cannot write refreshed binding manifest: {error}"))
}

/// Extracts one raw maintained-Clang metadata document from current adapter headers.
fn extract_live_metadata(root: &Path, extractor: &Path) -> Result<Value, String> {
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
    let manifest = read_json(&root.join(MANIFEST))?;
    let metadata = object(&manifest, "cpp_metadata")?;
    let sources = array(metadata, "sources")?
        .iter()
        .map(|source| {
            source
                .as_str()
                .and_then(|source| source.strip_prefix("cpp/"))
                .map(str::to_string)
                .ok_or_else(|| "metadata source must use the cpp/ prefix".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    require(
        !sources.is_empty(),
        "metadata source list must not be empty",
    )?;
    let database = expanded_compile_database(root, &cpp, &sources)?;
    fs::write(
        live.join("compile_commands.json"),
        serde_json::to_vec_pretty(&database)
            .map_err(|error| format!("cannot encode live compilation database: {error}"))?,
    )
    .map_err(|error| format!("cannot write live compilation database: {error}"))?;

    let include_path = cpp_include_path()?;
    let mut documents = Vec::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        let output = live.join(format!("metadata-{index}.json"));
        run_extractor(
            extractor,
            &cpp,
            &live,
            &output,
            source,
            include_path.as_ref(),
        )?;
        documents.push(read_json(&output)?);
    }

    let actual = merge_direct_calls(documents)?;
    fs::remove_dir_all(&live)
        .map_err(|error| format!("cannot remove live metadata directory: {error}"))?;
    Ok(actual)
}

/// Expands the reviewed compile command across every adapter translation unit.
fn expanded_compile_database(root: &Path, cpp: &Path, sources: &[String]) -> Result<Value, String> {
    let database = read_json(&root.join(COMPILE_DATABASE))?;
    let template = database
        .as_array()
        .and_then(|commands| commands.first())
        .and_then(Value::as_object)
        .ok_or_else(|| "compile database must contain one object template".to_string())?;
    let mut commands = Vec::with_capacity(sources.len());
    for source in sources {
        let mut command = template.clone();
        command.insert(
            "directory".to_string(),
            Value::String(cpp.to_string_lossy().into_owned()),
        );
        command.insert("file".to_string(), Value::String(source.clone()));
        let arguments = command
            .get_mut("arguments")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| "compile command arguments must be an array".to_string())?;
        let mut replaced = false;
        for argument in arguments {
            if argument.as_str() == Some("opencv_metadata.cc") {
                *argument = Value::String(source.clone());
                replaced = true;
            }
        }
        require(
            replaced,
            "compile command must name opencv_metadata.cc as its source",
        )?;
        commands.push(Value::Object(command));
    }
    Ok(Value::Array(commands))
}

/// Runs one maintained-Clang extraction for a selected adapter source.
fn run_extractor(
    extractor: &Path,
    cpp: &Path,
    database: &Path,
    output: &Path,
    source: &str,
    include_path: Option<&OsString>,
) -> Result<(), String> {
    let mut invocation = Command::new(extractor);
    invocation
        .current_dir(cpp)
        .arg("-p")
        .arg(database)
        .arg("--output")
        .arg(output)
        .arg("--header")
        .arg("opencv_adapter.hpp")
        .arg("--header")
        .arg("opencv_ml.hpp")
        .arg("--header")
        .arg("opencv_dnn.hpp")
        .arg("--namespace")
        .arg("terlan_opencv")
        .arg(cpp.join(source));
    if let Some(include_path) = include_path {
        invocation.env("CPLUS_INCLUDE_PATH", include_path);
    }
    let status = invocation
        .status()
        .map_err(|error| format!("cannot start maintained extractor for {source}: {error}"))?;
    require(
        status.success(),
        format!("maintained extractor failed for {source}"),
    )
}

/// Merges per-source direct calls into the primary declaration document.
fn merge_direct_calls(mut documents: Vec<Value>) -> Result<Value, String> {
    require(
        !documents.is_empty(),
        "metadata extraction produced no documents",
    )?;
    let mut calls_by_symbol = BTreeMap::<String, BTreeMap<String, Value>>::new();
    for document in &documents {
        let root = document
            .as_object()
            .ok_or_else(|| "metadata document root must be an object".to_string())?;
        for symbol in array(root, "symbols")? {
            let symbol = symbol
                .as_object()
                .ok_or_else(|| "metadata symbol must be an object".to_string())?;
            let owner = string(symbol, "id")?;
            let Some(calls) = symbol.get("direct_calls").and_then(Value::as_array) else {
                continue;
            };
            let merged = calls_by_symbol.entry(owner.to_string()).or_default();
            for call in calls {
                let call = call
                    .as_object()
                    .ok_or_else(|| "direct call must be an object".to_string())?;
                merged.insert(string(call, "id")?.to_string(), Value::Object(call.clone()));
            }
        }
    }

    let mut primary = documents.remove(0);
    let root = primary
        .as_object_mut()
        .ok_or_else(|| "primary metadata root must be an object".to_string())?;
    let symbols = root
        .get_mut("symbols")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "primary metadata symbols must be an array".to_string())?;
    for symbol in symbols {
        let symbol = symbol
            .as_object_mut()
            .ok_or_else(|| "primary metadata symbol must be an object".to_string())?;
        let id = string(symbol, "id")?.to_string();
        if let Some(calls) = calls_by_symbol.remove(&id) {
            symbol.insert(
                "direct_calls".to_string(),
                Value::Array(calls.into_values().collect()),
            );
        }
    }
    Ok(primary)
}

/// Derives a deterministic package symbol ID for a newly extracted declaration.
fn stable_symbol_id(symbol: &Map<String, Value>) -> Result<String, String> {
    let kind = string(symbol, "kind")?;
    let name = snake_name(string(symbol, "cpp_name")?);
    if kind == "method" {
        let receiver = string(symbol, "receiver")?
            .rsplit("::")
            .next()
            .ok_or_else(|| "method receiver is empty".to_string())?;
        return Ok(format!("method.{}.{}", snake_name(receiver), name));
    }
    Ok(format!("{kind}.{name}"))
}

/// Converts one C++ identifier into the package's stable snake-case spelling.
fn snake_name(name: &str) -> String {
    let name = name
        .strip_prefix("OpenCv")
        .map_or_else(|| name.to_string(), |suffix| format!("Opencv{suffix}"));
    let mut result = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() && index != 0 {
            result.push('_');
        }
        result.push(character.to_ascii_lowercase());
    }
    result
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
    require(
        !roots.is_empty(),
        "host C++ compiler reported no include roots",
    )?;
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

/// Returns a required TOML table.
fn toml_table<'a>(
    value: &'a toml::Value,
    key: &str,
) -> Result<&'a toml::map::Map<String, toml::Value>, String> {
    value
        .get(key)
        .and_then(toml::Value::as_table)
        .ok_or_else(|| format!("{key} must be a TOML table"))
}

/// Returns a required non-empty TOML string.
fn toml_text<'a>(
    value: &'a toml::map::Map<String, toml::Value>,
    key: &str,
) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(toml::Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty TOML string"))
}

/// Requires one exact TOML string.
fn require_toml_text(
    value: &toml::map::Map<String, toml::Value>,
    key: &str,
    expected: &str,
) -> Result<(), String> {
    let actual = toml_text(value, key)?;
    require(
        actual == expected,
        format!("{key} must be `{expected}`, found `{actual}`"),
    )
}

/// Requires one exact TOML integer.
fn require_toml_integer(
    value: &toml::map::Map<String, toml::Value>,
    key: &str,
    expected: i64,
) -> Result<(), String> {
    let actual = value
        .get(key)
        .and_then(toml::Value::as_integer)
        .ok_or_else(|| format!("{key} must be a TOML integer"))?;
    require(
        actual == expected,
        format!("{key} must be {expected}, found {actual}"),
    )
}

/// Requires one exact TOML Boolean.
fn require_toml_bool(
    value: &toml::map::Map<String, toml::Value>,
    key: &str,
    expected: bool,
) -> Result<(), String> {
    let actual = value
        .get(key)
        .and_then(toml::Value::as_bool)
        .ok_or_else(|| format!("{key} must be a TOML Boolean"))?;
    require(
        actual == expected,
        format!("{key} must be {expected}, found {actual}"),
    )
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
