//! Declaration identity auditing for the pinned OpenCV public API.

mod candidates;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema used by the normalized upstream inventory.
const SNAPSHOT_SCHEMA: &str = "terlan.opencv.api-inventory.v1";
/// Schema used by the evidence-backed parity mapping ledger.
const MAPPING_SCHEMA: &str = "terlan.opencv.api-parity.v1";
/// Exact OpenCV source revision represented by both parity documents.
const UPSTREAM_COMMIT: &str = "fe38fc608f6acb8b68953438a62305d8318f4fcd";
/// OpenCV modules selected for the complete portable runtime denominator.
const PUBLIC_MODULES: &[&str] = &[
    "calib3d",
    "core",
    "dnn",
    "features2d",
    "flann",
    "gapi",
    "highgui",
    "imgcodecs",
    "imgproc",
    "ml",
    "objdetect",
    "photo",
    "stitching",
    "video",
    "videoio",
];

/// Maintained Clang extractor output needed for normalization.
#[derive(Deserialize)]
struct ExtractorDocument {
    /// Extracted declarations selected from public headers.
    symbols: Vec<ExtractedSymbol>,
}

/// One declaration emitted by the maintained Clang extractor.
#[derive(Deserialize)]
struct ExtractedSymbol {
    /// Canonical C++ declaration identity.
    id: String,
    /// Declaration family such as function, method, record, or enum.
    kind: String,
    /// Unqualified C++ declaration name.
    cpp_name: String,
    /// Qualified overload-set identity.
    overload_set: String,
    /// Exact pinned source location.
    source: ExtractedSource,
}

/// Source location emitted by the maintained Clang extractor.
#[derive(Deserialize)]
struct ExtractedSource {
    /// Public header basename.
    path: String,
    /// One-based source line.
    line: u64,
    /// One-based source column.
    column: u64,
}

/// Committed normalized OpenCV parity denominator.
#[derive(Deserialize, Eq, PartialEq, Serialize)]
struct Inventory {
    /// Inventory schema.
    schema: String,
    /// Exact pinned OpenCV revision.
    upstream_commit: String,
    /// Total identity count.
    total: usize,
    /// Deterministic identity count by module.
    modules: BTreeMap<String, usize>,
    /// Exact normalized declarations.
    identities: Vec<Identity>,
}

/// One source-stable OpenCV API identity.
#[derive(Deserialize, Eq, PartialEq, Serialize)]
struct Identity {
    /// Unique identity including source location.
    id: String,
    /// Canonical C++ identity before source disambiguation.
    upstream_id: String,
    /// Declaration family.
    kind: String,
    /// Owning OpenCV module.
    module: String,
    /// Unqualified declaration name.
    cpp_name: String,
    /// Qualified overload set.
    overload_set: String,
    /// Exact pinned source location.
    source: Source,
}

/// Normalized source location stored in the inventory.
#[derive(Deserialize, Eq, PartialEq, Serialize)]
struct Source {
    /// Public header basename.
    path: String,
    /// One-based source line.
    line: u64,
    /// One-based source column.
    column: u64,
}

/// Evidence-backed Terlan mappings for upstream identities.
#[derive(Deserialize)]
struct MappingDocument {
    /// Mapping schema.
    schema: String,
    /// Exact pinned OpenCV revision.
    upstream_commit: String,
    /// Behavior-preserving Terlan mappings.
    #[serde(default)]
    mappings: Vec<Mapping>,
}

/// One direct, reshaped, or composed Terlan behavior mapping.
#[derive(Deserialize)]
struct Mapping {
    /// Exact upstream identities covered by this operation.
    upstream: Vec<String>,
    /// Public Terlan operation.
    terlan: String,
    /// Additional public operations that collectively preserve the behavior.
    #[serde(default)]
    additional_terlan: Vec<String>,
    /// Direct, reshaped, or composed mapping classification.
    disposition: String,
    /// Explanation of behavior preservation.
    rationale: String,
    /// Permanent executable evidence paths.
    evidence: Vec<String>,
}

/// Runs the requested parity command and renders deterministic failures.
fn main() {
    if let Err(message) = run(env::args().skip(1).collect()) {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

/// Dispatches snapshot generation, live reproduction, reporting, or strict validation.
fn run(arguments: Vec<String>) -> Result<(), String> {
    let root = package_root()?;
    match arguments.as_slice() {
        [flag, extractor, output] if flag == "--write-snapshot" => {
            write_snapshot(Path::new(extractor), Path::new(output))
        }
        [flag, extractor] if flag == "--refresh" => refresh(&root, Path::new(extractor)),
        [flag, extractor] if flag == "--reproduce" => reproduce(&root, Path::new(extractor)),
        [flag] if flag == "--candidates" => candidates::report(&root),
        [flag] if flag == "--refresh-reconciliation" => {
            candidates::refresh_reconciliation(&root)
        }
        [] => report(&root, false),
        [flag] if flag == "--strict" => report(&root, true),
        _ => Err(
            "usage: terlan-opencv-api-parity [--candidates | --refresh-reconciliation | --strict | --refresh EXTRACTOR | --reproduce EXTRACTOR | --write-snapshot EXTRACTOR_JSON OUTPUT]".to_string(),
        ),
    }
}

/// Resolves the package root from the parity tool location.
fn package_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| "API parity tool must remain under tools/api_parity".to_string())
}

/// Writes a normalized inventory from maintained Clang extractor output.
fn write_snapshot(extractor: &Path, output: &Path) -> Result<(), String> {
    let inventory = inventory_from_extractor(extractor)?;
    let encoded = serde_json::to_string_pretty(&inventory)
        .map_err(|error| format!("cannot encode API inventory: {error}"))?;
    fs::write(output, format!("{encoded}\n"))
        .map_err(|error| format!("cannot write {}: {error}", output.display()))
}

/// Converts maintained Clang output into a deterministic parity inventory.
fn inventory_from_extractor(extractor: &Path) -> Result<Inventory, String> {
    let extracted: ExtractorDocument = read_json(extractor)?;
    let mut identities = extracted
        .symbols
        .into_iter()
        .map(normalize_identity)
        .collect::<Result<Vec<_>, _>>()?;
    identities.sort_by(|left, right| left.id.cmp(&right.id));
    let unique = identities
        .iter()
        .map(|identity| identity.id.as_str())
        .collect::<BTreeSet<_>>();
    require(
        unique.len() == identities.len(),
        "normalized upstream identities are not unique",
    )?;
    let mut modules = BTreeMap::new();
    for identity in &identities {
        *modules.entry(identity.module.clone()).or_insert(0) += 1;
    }
    Ok(Inventory {
        schema: SNAPSHOT_SCHEMA.to_string(),
        upstream_commit: UPSTREAM_COMMIT.to_string(),
        total: identities.len(),
        modules,
        identities,
    })
}

/// Replaces the committed inventory from one pinned maintained-Clang extraction.
fn refresh(root: &Path, extractor: &Path) -> Result<(), String> {
    let inventory = extract_pinned_inventory(root, extractor)?;
    let encoded = serde_json::to_string_pretty(&inventory)
        .map_err(|error| format!("cannot encode API inventory: {error}"))?;
    fs::write(
        root.join("bindings/opencv.api-inventory.json"),
        format!("{encoded}\n"),
    )
    .map_err(|error| format!("cannot write API inventory: {error}"))?;
    println!(
        "OpenCV API inventory refreshed: {} identities across {} modules",
        inventory.total,
        inventory.modules.len()
    );
    Ok(())
}

/// Reproduces the committed inventory from the pinned source checkout.
fn reproduce(root: &Path, extractor: &Path) -> Result<(), String> {
    let actual = extract_pinned_inventory(root, extractor)?;
    let expected: Inventory = read_json(&root.join("bindings/opencv.api-inventory.json"))?;
    require(
        actual == expected,
        "committed OpenCV API inventory differs from maintained Clang extraction",
    )?;
    println!(
        "OpenCV API inventory reproduced: {} identities across {} modules",
        actual.total,
        actual.modules.len()
    );
    Ok(())
}

/// Extracts one normalized inventory from the pinned OpenCV checkout.
fn extract_pinned_inventory(root: &Path, extractor: &Path) -> Result<Inventory, String> {
    require(
        extractor.is_file(),
        "maintained metadata extractor is missing",
    )?;
    let upstream = root.join("upstream/opencv");
    let revision = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&upstream)
        .output()
        .map_err(|error| format!("cannot inspect pinned OpenCV checkout: {error}"))?;
    require(revision.status.success(), "cannot read OpenCV revision")?;
    require(
        String::from_utf8_lossy(&revision.stdout).trim() == UPSTREAM_COMMIT,
        "OpenCV checkout is not at the pinned revision",
    )?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let temporary = env::temp_dir().join(format!("terlan-opencv-api-{nonce}"));
    fs::create_dir_all(&temporary)
        .map_err(|error| format!("cannot create API inventory directory: {error}"))?;
    let mut database: serde_json::Value =
        read_json(&root.join("bindings/api/compile_commands.json"))?;
    for command in database
        .as_array_mut()
        .ok_or_else(|| "API compile database must be an array".to_string())?
    {
        command
            .as_object_mut()
            .ok_or_else(|| "API compile command must be an object".to_string())?
            .insert(
                "directory".to_string(),
                serde_json::Value::String(root.join("bindings/api").display().to_string()),
            );
    }
    fs::write(
        temporary.join("compile_commands.json"),
        serde_json::to_vec_pretty(&database)
            .map_err(|error| format!("cannot encode API compile database: {error}"))?,
    )
    .map_err(|error| format!("cannot write API compile database: {error}"))?;
    let output = temporary.join("opencv-public-api.json");
    let mut command = Command::new(extractor);
    command
        .current_dir(root.join("bindings/api"))
        .arg("-p")
        .arg(&temporary)
        .arg("--namespace")
        .arg("cv")
        .arg("--public-only")
        .arg("--output")
        .arg(&output);
    command
        .arg("--exact-headers")
        .arg("--header-root")
        .arg(&upstream);
    for header in public_headers(&upstream)? {
        command.arg("--header").arg(header);
    }
    command.arg(root.join("bindings/api/opencv_public_api_inventory.cc"));
    if let Some(include_path) = cpp_include_path()? {
        command.env("CPLUS_INCLUDE_PATH", include_path);
    }
    let status = command
        .status()
        .map_err(|error| format!("cannot start maintained metadata extractor: {error}"))?;
    require(
        status.success(),
        format!("OpenCV API extraction failed with status {status}"),
    )?;
    let actual = inventory_from_extractor(&output);
    let cleanup = fs::remove_dir_all(&temporary);
    cleanup.map_err(|error| format!("cannot remove API inventory directory: {error}"))?;
    actual
}

/// Finds host C++ standard-library include roots for embedded Clang.
fn cpp_include_path() -> Result<Option<OsString>, String> {
    if let Some(value) = env::var_os("CPLUS_INCLUDE_PATH") {
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
    env::join_paths(roots)
        .map(Some)
        .map_err(|error| format!("cannot encode host C++ include search: {error}"))
}

/// Enumerates the public CPU C++ headers represented by the umbrella translation unit.
fn public_headers(upstream: &Path) -> Result<Vec<PathBuf>, String> {
    let mut headers = Vec::new();
    for module in PUBLIC_MODULES {
        collect_public_headers(
            &upstream
                .join("modules")
                .join(module)
                .join("include/opencv2"),
            &mut headers,
        )?;
    }
    headers.sort();
    require(!headers.is_empty(), "OpenCV public header set is empty")?;
    Ok(headers)
}

/// Recursively collects selected public headers under one OpenCV module root.
fn collect_public_headers(directory: &Path, headers: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(directory).map_err(|error| {
        format!(
            "cannot read public headers in {}: {error}",
            directory.display()
        )
    })?;
    for entry in entries {
        let path = entry
            .map_err(|error| format!("cannot read entry in {}: {error}", directory.display()))?
            .path();
        if path.is_dir() {
            collect_public_headers(&path, headers)?;
        } else if is_public_cpu_header(&path) {
            headers.push(path);
        }
    }
    Ok(())
}

/// Returns whether a source header belongs to the portable public C++ surface.
fn is_public_cpu_header(path: &Path) -> bool {
    let rendered = path.to_string_lossy().replace('\\', "/");
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let cpp_header = name.ends_with(".hpp") || rendered.contains("/flann/") && name.ends_with(".h");
    cpp_header
        && !name.ends_with(".inl.hpp")
        && !name.ends_with(".impl.hpp")
        && !name.contains("private")
        && !name.starts_with("cv_cpu_")
        && !rendered.contains("/private/")
        && !rendered.contains("/detail/")
        && !rendered.contains("/hal/")
        && !rendered.contains("/cuda/")
        && !rendered.contains("/opencl/")
        && !rendered.contains("/openvx/")
        && !rendered.contains("/gpu/")
        && !rendered.contains("/ocl/")
        && !rendered.contains("/oak/")
        && !matches!(
            name,
            "cuda.hpp"
                | "cuda_stream_accessor.hpp"
                | "cuda_types.hpp"
                | "directx.hpp"
                | "eigen.hpp"
                | "ocl.hpp"
                | "opengl.hpp"
                | "ovx.hpp"
                | "va_intel.hpp"
        )
}

/// Normalizes one extracted declaration into a source-stable identity.
fn normalize_identity(symbol: ExtractedSymbol) -> Result<Identity, String> {
    let module =
        if symbol.id.contains(":cv::gapi::") || symbol.overload_set.starts_with("cv::gapi::") {
            "gapi"
        } else {
            module_for_header(&symbol.source.path)?
        };
    let id = format!(
        "{}@{}:{}:{}",
        symbol.id, symbol.source.path, symbol.source.line, symbol.source.column
    );
    Ok(Identity {
        id,
        upstream_id: symbol.id,
        kind: symbol.kind,
        module: module.to_string(),
        cpp_name: symbol.cpp_name,
        overload_set: symbol.overload_set,
        source: Source {
            path: symbol.source.path,
            line: symbol.source.line,
            column: symbol.source.column,
        },
    })
}

/// Classifies a selected public header into its owning OpenCV module.
fn module_for_header(header: &str) -> Result<&'static str, String> {
    for module in PUBLIC_MODULES {
        if header.starts_with(&format!("modules/{module}/include/")) {
            return Ok(module);
        }
    }
    Err(format!("unclassified OpenCV public header `{header}`"))
}

/// Reports mapped and pending identities and optionally enforces full audit coverage.
fn report(root: &Path, strict: bool) -> Result<(), String> {
    let inventory: Inventory = read_json(&root.join("bindings/opencv.api-inventory.json"))?;
    require(
        inventory.schema == SNAPSHOT_SCHEMA,
        "invalid inventory schema",
    )?;
    require(
        inventory.upstream_commit == UPSTREAM_COMMIT,
        "inventory upstream commit drift",
    )?;
    require(
        inventory.total == inventory.identities.len(),
        "inventory total does not match identities",
    )?;
    let mappings: MappingDocument = read_toml(&root.join("bindings/opencv.api-parity.toml"))?;
    require(mappings.schema == MAPPING_SCHEMA, "invalid mapping schema")?;
    require(
        mappings.upstream_commit == UPSTREAM_COMMIT,
        "mapping upstream commit drift",
    )?;
    let known = inventory
        .identities
        .iter()
        .map(|identity| identity.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut mapped = BTreeSet::new();
    for mapping in &mappings.mappings {
        validate_mapping(mapping)?;
        validate_evidence(root, mapping)?;
        for upstream in &mapping.upstream {
            require(
                known.contains(upstream.as_str()),
                format!("mapping references unknown upstream identity `{upstream}`"),
            )?;
            require(
                mapped.insert(upstream.as_str()),
                format!("duplicate upstream mapping `{upstream}`"),
            )?;
        }
    }
    candidates::validate_reconciliation(root, &inventory, &mappings)?;
    println!("OpenCV 4.13.0 API identity audit");
    for (module, total) in &inventory.modules {
        let covered = inventory
            .identities
            .iter()
            .filter(|identity| identity.module == *module && mapped.contains(identity.id.as_str()))
            .count();
        println!("{module}: {covered}/{total}");
    }
    let pending = inventory.total - mapped.len();
    println!(
        "total: {}/{} mapped, {} pending",
        mapped.len(),
        inventory.total,
        pending
    );
    if strict && pending != 0 {
        return Err(format!(
            "OpenCV API identity audit is incomplete: {pending} upstream identities remain unmapped"
        ));
    }
    Ok(())
}

/// Validates one behavior mapping and its executable evidence.
fn validate_mapping(mapping: &Mapping) -> Result<(), String> {
    require(
        !mapping.upstream.is_empty(),
        "mapping upstream must not be empty",
    )?;
    require(
        !mapping.terlan.trim().is_empty(),
        "mapping Terlan API is empty",
    )?;
    let operations = std::iter::once(mapping.terlan.as_str())
        .chain(mapping.additional_terlan.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    require(
        operations.len() == mapping.additional_terlan.len() + 1
            && operations
                .iter()
                .all(|operation| !operation.trim().is_empty()),
        format!(
            "{}: additional Terlan operations must be nonempty and unique",
            mapping.terlan
        ),
    )?;
    require(
        matches!(
            mapping.disposition.as_str(),
            "direct" | "reshaped" | "composed"
        ),
        format!("invalid mapping disposition `{}`", mapping.disposition),
    )?;
    require(
        !mapping.rationale.trim().is_empty(),
        format!("{}: mapping rationale is empty", mapping.terlan),
    )?;
    require(
        !mapping.evidence.is_empty() && mapping.evidence.iter().all(|item| !item.trim().is_empty()),
        format!("{}: executable evidence is required", mapping.terlan),
    )
}

/// Requires every declared executable evidence path to exist in the package.
fn validate_evidence(root: &Path, mapping: &Mapping) -> Result<(), String> {
    for evidence in &mapping.evidence {
        require(
            root.join(evidence).is_file(),
            format!(
                "{}: evidence path does not exist: {evidence}",
                mapping.terlan
            ),
        )?;
    }
    Ok(())
}

/// Reads and deserializes one JSON document.
fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&source).map_err(|error| format!("{}: {error}", path.display()))
}

/// Reads and deserializes one TOML document.
fn read_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    toml::from_str(&source).map_err(|error| format!("{}: {error}", path.display()))
}

/// Converts a failed invariant into a deterministic diagnostic.
fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbol(id: &str, overload_set: &str, path: &str) -> ExtractedSymbol {
        ExtractedSymbol {
            id: id.to_string(),
            kind: "function".to_string(),
            cpp_name: "operation".to_string(),
            overload_set: overload_set.to_string(),
            source: ExtractedSource {
                path: path.to_string(),
                line: 7,
                column: 11,
            },
        }
    }

    #[test]
    fn identity_includes_source_location_and_module() {
        let identity = normalize_identity(symbol(
            "function:cv::resize()",
            "cv::resize",
            "modules/imgproc/include/opencv2/imgproc.hpp",
        ))
        .expect("normalize identity");

        assert_eq!(
            identity.id,
            "function:cv::resize()@modules/imgproc/include/opencv2/imgproc.hpp:7:11".to_string()
        );
        assert_eq!(identity.module, "imgproc");
    }

    #[test]
    fn gapi_namespace_overrides_ambiguous_header_name() {
        let identity = normalize_identity(symbol(
            "function:cv::gapi::resize()",
            "cv::gapi::resize",
            "modules/gapi/include/opencv2/gapi/imgproc.hpp",
        ))
        .expect("normalize G-API identity");

        assert_eq!(identity.module, "gapi");
    }

    #[test]
    fn unknown_public_header_is_rejected() {
        assert_eq!(
            module_for_header("private.hpp").unwrap_err(),
            "unclassified OpenCV public header `private.hpp`"
        );
    }

    #[test]
    fn mappings_require_behavior_and_evidence() {
        let valid = Mapping {
            upstream: vec![
                "function:cv::resize()@modules/imgproc/include/opencv2/imgproc.hpp:7:11"
                    .to_string(),
            ],
            terlan: "opencv.Imgproc.resize".to_string(),
            additional_terlan: Vec::new(),
            disposition: "reshaped".to_string(),
            rationale: "Returns the output matrix instead of mutating an output parameter."
                .to_string(),
            evidence: vec!["tests/opencv/ImgprocTest.terl".to_string()],
        };
        assert!(validate_mapping(&valid).is_ok());

        let invalid = Mapping {
            evidence: Vec::new(),
            ..valid
        };
        assert_eq!(
            validate_mapping(&invalid).unwrap_err(),
            "opencv.Imgproc.resize: executable evidence is required"
        );
    }

    #[test]
    fn evidence_paths_must_resolve_inside_the_package() {
        let mapping = Mapping {
            upstream: vec!["function:cv::resize()".to_string()],
            terlan: "opencv.Imgproc.resize".to_string(),
            additional_terlan: Vec::new(),
            disposition: "reshaped".to_string(),
            rationale: "Returns one owned output matrix.".to_string(),
            evidence: vec!["missing-evidence.terl".to_string()],
        };
        assert_eq!(
            validate_evidence(Path::new("/definitely/missing/package"), &mapping).unwrap_err(),
            "opencv.Imgproc.resize: evidence path does not exist: missing-evidence.terl"
        );
    }

    #[test]
    fn portable_header_policy_excludes_internal_and_accelerated_headers() {
        assert!(is_public_cpu_header(Path::new(
            "modules/core/include/opencv2/core/mat.hpp"
        )));
        assert!(is_public_cpu_header(Path::new(
            "modules/flann/include/opencv2/flann/kdtree_index.h"
        )));
        assert!(!is_public_cpu_header(Path::new(
            "modules/core/include/opencv2/core/private.hpp"
        )));
        assert!(!is_public_cpu_header(Path::new(
            "modules/core/include/opencv2/core/cuda/utility.hpp"
        )));
        assert!(!is_public_cpu_header(Path::new(
            "modules/core/include/opencv2/core/mat.inl.hpp"
        )));
    }
}
