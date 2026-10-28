//! Builds and verifies relocatable Terlan OpenCV package artifacts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output};
use walkdir::WalkDir;

mod release;

use release::{build_release_plan, verify_release_archive};

/// Stable schema written into every assembled package artifact.
const ARTIFACT_SCHEMA: &str = "terlan.opencv.artifact.v1";

/// OpenCV shared libraries required by the current package surface.
const OPENCV_LIBRARIES: &[&str] = &[
    "calib3d",
    "core",
    "dnn",
    "features2d",
    "flann",
    "imgcodecs",
    "imgproc",
    "ml",
];

/// Dynamic libraries provided by the Linux target rather than the package.
const LINUX_SYSTEM_LIBRARIES: &[&str] = &[
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libstdc++.so.6",
    "linux-vdso.so.1",
];

/// Parsed root package metadata needed by the artifact identity.
#[derive(Debug, Deserialize)]
struct RootManifest {
    /// Public package section.
    package: RootPackage,
}

/// Public package identity read from `terlan.toml`.
#[derive(Debug, Deserialize)]
struct RootPackage {
    /// Published package name.
    name: String,
    /// Published package version.
    version: String,
    /// Public module namespace exposed by the package.
    namespace: String,
}

/// Parsed pinned OpenCV source identity.
#[derive(Debug, Deserialize)]
struct UpstreamLock {
    /// Upstream source repository.
    repository: String,
    /// OpenCV release tag.
    tag: String,
    /// Exact OpenCV Git revision.
    commit: String,
}

/// Serializable package identity included in the artifact manifest.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PackageIdentity {
    /// Terlan package name.
    name: String,
    /// Terlan package version.
    version: String,
    /// Source revision used to assemble the artifact.
    revision: String,
    /// Whether source changes existed outside the recorded revision.
    dirty: bool,
}

/// Serializable OpenCV identity included in the artifact manifest.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct OpenCvIdentity {
    /// OpenCV release tag.
    version: String,
    /// Exact OpenCV source revision.
    revision: String,
}

/// Relative runtime locations consumed after artifact extraction.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct RuntimeLayout {
    /// Runtime guard executable relative to the artifact root.
    guard: String,
    /// Relocatable native worker launcher relative to the artifact root.
    worker: String,
    /// Directory containing private OpenCV libraries.
    library_dir: String,
    /// Runtime executable bindings consumed by Terlan package execution.
    environment: BTreeMap<String, String>,
}

/// Complete target artifact manifest.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ArtifactManifest {
    /// Stable manifest schema identifier.
    schema: String,
    /// Target triple supported by this artifact.
    target: String,
    /// Package source identity.
    package: PackageIdentity,
    /// OpenCV source identity.
    opencv: OpenCvIdentity,
    /// Relative Terlan package directory.
    terlan_package: String,
    /// Runtime executable and library locations.
    runtime: RuntimeLayout,
    /// License files shipped with the artifact.
    licenses: Vec<String>,
}

/// Fully resolved inputs used to assemble one artifact.
#[derive(Debug)]
struct BuildInputs {
    /// Package repository root.
    root: PathBuf,
    /// Output directory receiving staging and archive files.
    output: PathBuf,
    /// Supported target triple.
    target: String,
    /// Parsed root package metadata.
    package: RootManifest,
    /// Parsed OpenCV lock metadata.
    upstream: UpstreamLock,
}

/// Program entry point with stable error rendering.
fn main() {
    if let Err(error) = run(env::args().skip(1).collect()) {
        eprintln!("terlan-opencv-package-artifact: {error}");
        std::process::exit(1);
    }
}

/// Dispatches package artifact build and release-validation commands.
fn run(arguments: Vec<String>) -> Result<(), String> {
    match arguments.as_slice() {
        [command, root, output, target] if command == "build" => {
            let archive = build_artifact(Path::new(root), Path::new(output), target)?;
            println!("{}", archive.display());
            Ok(())
        }
        [command, archive, extract_dir] if command == "verify" => {
            let root = verify_archive(Path::new(archive), Path::new(extract_dir))?;
            println!("{}", root.display());
            Ok(())
        }
        [command, root, tag, revision, target] if command == "release-plan" => {
            let plan = build_release_plan(Path::new(root), tag, revision, target)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&plan)
                    .map_err(|error| format!("cannot serialize release plan: {error}"))?
            );
            Ok(())
        }
        [command, archive, extract_dir, tag, revision, target]
            if command == "release-verify" =>
        {
            let root = verify_release_archive(
                Path::new(archive),
                Path::new(extract_dir),
                tag,
                revision,
                target,
            )?;
            println!("{}", root.display());
            Ok(())
        }
        _ => Err(
            "usage: package-artifact build ROOT OUTPUT TARGET | verify ARCHIVE EXTRACT_DIR | release-plan ROOT TAG REVISION TARGET | release-verify ARCHIVE EXTRACT_DIR TAG REVISION TARGET".to_string(),
        ),
    }
}

/// Assembles one deterministic package archive and returns its path.
fn build_artifact(root: &Path, output: &Path, target: &str) -> Result<PathBuf, String> {
    let inputs = load_build_inputs(root, output, target)?;
    let artifact_name = format!(
        "{}-{}-{}",
        inputs.package.package.name, inputs.package.package.version, inputs.target
    );
    let staging = inputs.output.join(&artifact_name);
    remove_dir_if_present(&staging)?;
    fs::create_dir_all(&staging).map_err(|error| {
        format!(
            "cannot create staging directory {}: {error}",
            staging.display()
        )
    })?;

    copy_package_payload(&inputs, &staging)?;
    copy_runtime_payload(&inputs, &staging)?;
    copy_library_payload(&inputs, &staging)?;
    copy_license_payload(&inputs, &staging)?;
    write_artifact_manifest(&inputs, &staging)?;
    validate_dynamic_dependencies(&staging)?;
    sanitize_embedded_build_paths(&inputs, &staging)?;
    reject_embedded_build_paths(&inputs, &staging)?;
    write_payload_checksums(&staging)?;

    let archive = inputs.output.join(format!("{artifact_name}.tar.zst"));
    create_archive(&inputs.output, &artifact_name, &archive)?;
    write_archive_checksum(&archive)?;
    Ok(archive)
}

/// Loads and validates package and upstream metadata for one build.
fn load_build_inputs(root: &Path, output: &Path, target: &str) -> Result<BuildInputs, String> {
    if target != "x86_64-unknown-linux-gnu" {
        return Err(format!("unsupported artifact target `{target}`"));
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve package root {}: {error}", root.display()))?;
    let package = read_toml::<RootManifest>(&root.join("terlan.toml"))?;
    let upstream = read_toml::<UpstreamLock>(&root.join("upstream/opencv.lock.toml"))?;
    let actual_upstream = git_output(&root.join("upstream/opencv"), &["rev-parse", "HEAD"])?;
    if actual_upstream != upstream.commit {
        return Err(format!(
            "OpenCV revision mismatch: expected {}, found {actual_upstream}",
            upstream.commit
        ));
    }
    fs::create_dir_all(output).map_err(|error| {
        format!(
            "cannot create artifact output {}: {error}",
            output.display()
        )
    })?;
    let output = output.canonicalize().map_err(|error| {
        format!(
            "cannot resolve artifact output {}: {error}",
            output.display()
        )
    })?;
    Ok(BuildInputs {
        root,
        output,
        target: target.to_string(),
        package,
        upstream,
    })
}

/// Copies generated Terlan sources, documentation, and package metadata.
fn copy_package_payload(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let destination = staging.join("package");
    write_file(
        &destination.join("terlan.toml"),
        format!(
            "[package]\nname = \"{}\"\nversion = \"{}\"\nnamespace = \"{}\"\n\n[build]\nsource_roots = [\"src\"]\nartifact = \"library\"\n",
            inputs.package.package.name,
            inputs.package.package.version,
            inputs.package.package.namespace
        )
        .as_bytes(),
    )?;
    copy_tree(&inputs.root.join("generated/src"), &destination.join("src"))?;
    copy_tree(
        &inputs.root.join("generated/docs"),
        &destination.join("docs"),
    )?;
    copy_file(
        &inputs
            .root
            .join("generated/bindings/native-binding-manifest.json"),
        &destination.join("bindings/native-binding-manifest.json"),
    )
}

/// Copies release executables into the relocatable artifact layout.
fn copy_runtime_payload(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let bin = staging.join("bin");
    copy_executable(
        &inputs
            .root
            .join("generated/native/rust/target/release/native-boundary-helper"),
        &bin.join("terlan-opencv-worker"),
    )?;
    copy_executable(
        &inputs
            .root
            .join("native/runtime_guard/target/release/terlan-opencv-runtime-guard"),
        &bin.join("terlan-opencv-runtime-guard"),
    )
}

/// Copies the finite OpenCV shared-library closure under real and SONAME names.
fn copy_library_payload(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let source = inputs.root.join("upstream/install/lib");
    let destination = staging.join("lib");
    fs::create_dir_all(&destination).map_err(|error| {
        format!(
            "cannot create library directory {}: {error}",
            destination.display()
        )
    })?;
    for module in OPENCV_LIBRARIES {
        let real_name = format!("libopencv_{module}.so.4.13.0");
        let soname = format!("libopencv_{module}.so.413");
        let real_source = source.join(&real_name);
        copy_file(&real_source, &destination.join(&real_name))?;
        copy_file(&real_source, &destination.join(&soname))?;
    }
    Ok(())
}

/// Copies package and upstream license texts into the artifact.
fn copy_license_payload(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let licenses = staging.join("licenses");
    copy_file(
        &inputs.root.join("LICENSE"),
        &licenses.join("terlan-opencv-Apache-2.0.txt"),
    )?;
    copy_file(
        &inputs.root.join("upstream/opencv/LICENSE"),
        &licenses.join("opencv-Apache-2.0.txt"),
    )?;
    copy_optional_license(
        &inputs
            .root
            .join("upstream/opencv/3rdparty/libjpeg-turbo/LICENSE.md"),
        &licenses.join("libjpeg-turbo-license.md"),
    )?;
    copy_optional_license(
        &inputs.root.join("upstream/opencv/3rdparty/libpng/LICENSE"),
        &licenses.join("libpng-license.txt"),
    )?;
    copy_optional_license(
        &inputs.root.join("upstream/opencv/3rdparty/zlib/README"),
        &licenses.join("zlib-readme.txt"),
    )?;
    copy_optional_license(
        &inputs.root.join("upstream/opencv/3rdparty/libwebp/COPYING"),
        &licenses.join("libwebp-license.txt"),
    )?;
    copy_optional_license(
        &inputs
            .root
            .join("upstream/opencv/3rdparty/libtiff/LICENSE.md"),
        &licenses.join("libtiff-license.md"),
    )
}

/// Writes the target and runtime contract consumed by package installation.
fn write_artifact_manifest(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let revision = git_output(&inputs.root, &["rev-parse", "HEAD"])?;
    let dirty = !git_output(&inputs.root, &["status", "--porcelain"])?.is_empty();
    let manifest = ArtifactManifest {
        schema: ARTIFACT_SCHEMA.to_string(),
        target: inputs.target.clone(),
        package: PackageIdentity {
            name: inputs.package.package.name.clone(),
            version: inputs.package.package.version.clone(),
            revision,
            dirty,
        },
        opencv: OpenCvIdentity {
            version: inputs.upstream.tag.clone(),
            revision: inputs.upstream.commit.clone(),
        },
        terlan_package: "package".to_string(),
        runtime: RuntimeLayout {
            guard: "bin/terlan-opencv-runtime-guard".to_string(),
            worker: "bin/terlan-opencv-worker".to_string(),
            library_dir: "lib".to_string(),
            environment: BTreeMap::from([(
                "TERLAN_OPENCV_NATIVE_BOUNDARY_HELPER_PATH".to_string(),
                "bin/terlan-opencv-runtime-guard".to_string(),
            )]),
        },
        licenses: collect_relative_files(&staging.join("licenses"), staging)?,
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("cannot serialize artifact manifest: {error}"))?;
    bytes.push(b'\n');
    write_file(&staging.join("artifact.json"), &bytes)
}

/// Rejects missing or host-specific non-system dynamic dependencies.
fn validate_dynamic_dependencies(staging: &Path) -> Result<(), String> {
    let library_dir = staging.join("lib");
    let mut bundled = BTreeSet::new();
    for entry in fs::read_dir(&library_dir)
        .map_err(|error| format!("cannot read {}: {error}", library_dir.display()))?
    {
        let entry = entry.map_err(|error| format!("cannot read library entry: {error}"))?;
        bundled.insert(entry.file_name().to_string_lossy().into_owned());
    }
    let mut binaries = vec![staging.join("bin/terlan-opencv-worker")];
    for module in OPENCV_LIBRARIES {
        binaries.push(library_dir.join(format!("libopencv_{module}.so.4.13.0")));
    }
    for binary in binaries {
        let dynamic = command_output(
            Command::new("readelf").arg("-d").arg(&binary),
            "inspect dynamic runtime paths",
        )?;
        validate_runtime_search_paths(&binary, &dynamic)?;
        let output = command_output(
            Command::new("ldd")
                .env("LD_LIBRARY_PATH", &library_dir)
                .arg(&binary),
            "inspect dynamic dependencies",
        )?;
        for dependency in parse_ldd_names(&output) {
            if !bundled.contains(&dependency)
                && !LINUX_SYSTEM_LIBRARIES.contains(&dependency.as_str())
            {
                return Err(format!(
                    "{} depends on unbundled library `{dependency}`",
                    binary.display()
                ));
            }
        }
    }
    Ok(())
}

/// Rejects absolute and working-directory-relative ELF runtime search paths.
fn validate_runtime_search_paths(binary: &Path, output: &str) -> Result<(), String> {
    for search_path in parse_dynamic_search_paths(output) {
        for entry in search_path.split(':') {
            if entry != "$ORIGIN" && !entry.starts_with("$ORIGIN/") {
                return Err(format!(
                    "{} contains unsafe runtime search path `{entry}`",
                    binary.display()
                ));
            }
        }
    }
    Ok(())
}

/// Parses RPATH and RUNPATH values from `readelf -d` output.
fn parse_dynamic_search_paths(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter(|line| line.contains("(RPATH)") || line.contains("(RUNPATH)"))
        .filter_map(|line| {
            let start = line.find('[')? + 1;
            let end = line[start..].find(']')? + start;
            Some(line[start..end].to_string())
        })
        .collect()
}

/// Replaces ambient source prefixes with equal-length deterministic tokens.
fn sanitize_embedded_build_paths(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let mut replacements = vec![(
        inputs.root.as_os_str().as_encoded_bytes().to_vec(),
        padded_replacement(
            b"terlan-opencv",
            inputs.root.as_os_str().as_encoded_bytes().len(),
        ),
    )];
    if let Some(home) = env::var_os("HOME") {
        let home = home.as_encoded_bytes().to_vec();
        replacements.push((home.clone(), padded_replacement(b"terlan-home", home.len())));
    }
    for path in collect_payload_paths(staging)? {
        let full_path = staging.join(&path);
        let mut bytes = fs::read(&full_path)
            .map_err(|error| format!("cannot inspect {}: {error}", full_path.display()))?;
        let mut changed = false;
        for (needle, replacement) in &replacements {
            changed |= replace_bytes(&mut bytes, needle, replacement);
        }
        if changed {
            fs::write(&full_path, bytes)
                .map_err(|error| format!("cannot sanitize {}: {error}", full_path.display()))?;
        }
    }
    Ok(())
}

/// Produces an equal-length printable replacement for one filesystem prefix.
fn padded_replacement(label: &[u8], length: usize) -> Vec<u8> {
    let mut replacement = vec![b'_'; length];
    let copied = label.len().min(length);
    replacement[..copied].copy_from_slice(&label[..copied]);
    replacement
}

/// Replaces every non-overlapping byte occurrence without changing payload size.
fn replace_bytes(bytes: &mut [u8], needle: &[u8], replacement: &[u8]) -> bool {
    if needle.is_empty() || needle.len() != replacement.len() {
        return false;
    }
    let mut changed = false;
    let mut offset = 0;
    while offset + needle.len() <= bytes.len() {
        if &bytes[offset..offset + needle.len()] == needle {
            bytes[offset..offset + needle.len()].copy_from_slice(replacement);
            changed = true;
            offset += needle.len();
        } else {
            offset += 1;
        }
    }
    changed
}

/// Rejects package-root and home-directory paths embedded in native payloads.
fn reject_embedded_build_paths(inputs: &BuildInputs, staging: &Path) -> Result<(), String> {
    let mut forbidden = vec![inputs.root.as_os_str().as_encoded_bytes().to_vec()];
    if let Some(home) = env::var_os("HOME") {
        forbidden.push(home.as_encoded_bytes().to_vec());
    }
    for path in collect_payload_paths(staging)? {
        let full_path = staging.join(&path);
        let bytes = fs::read(&full_path)
            .map_err(|error| format!("cannot inspect {}: {error}", full_path.display()))?;
        for needle in &forbidden {
            if contains_bytes(&bytes, needle) {
                return Err(format!(
                    "{} contains an ambient build path",
                    full_path.display()
                ));
            }
        }
    }
    Ok(())
}

/// Returns whether one byte slice contains another non-empty byte slice.
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

/// Parses dependency names from successful `ldd` output.
fn parse_ldd_names(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            if let Some((name, _)) = trimmed.split_once(" => ") {
                return Some(name.to_string());
            }
            trimmed.split_whitespace().next().map(|name| {
                Path::new(name)
                    .file_name()
                    .and_then(OsStr::to_str)
                    .unwrap_or(name)
                    .to_string()
            })
        })
        .collect()
}

/// Writes sorted SHA-256 entries for every payload file.
fn write_payload_checksums(staging: &Path) -> Result<(), String> {
    let files = collect_payload_paths(staging)?;
    let mut output = String::new();
    for relative in files {
        let digest = hash_path(&staging.join(&relative))?;
        output.push_str(&format!("{digest}  {}\n", relative.display()));
    }
    write_file(&staging.join("checksums.sha256"), output.as_bytes())
}

/// Creates a deterministic zstd-compressed tar archive with GNU tar.
fn create_archive(parent: &Path, name: &str, archive: &Path) -> Result<(), String> {
    if archive.exists() {
        fs::remove_file(archive)
            .map_err(|error| format!("cannot replace archive {}: {error}", archive.display()))?;
    }
    command_output(
        Command::new("tar")
            .arg("--sort=name")
            .arg("--mtime=@0")
            .arg("--owner=0")
            .arg("--group=0")
            .arg("--numeric-owner")
            .arg("--zstd")
            .arg("-cf")
            .arg(archive)
            .arg("-C")
            .arg(parent)
            .arg(name),
        "create deterministic artifact archive",
    )?;
    Ok(())
}

/// Writes the external checksum used before archive extraction.
fn write_archive_checksum(archive: &Path) -> Result<(), String> {
    let digest = hash_path(archive)?;
    let file_name = archive
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("invalid archive name {}", archive.display()))?;
    let checksum = PathBuf::from(format!("{}.sha256", archive.display()));
    write_file(&checksum, format!("{digest}  {file_name}\n").as_bytes())
}

/// Verifies the archive against its adjacent external SHA-256 file.
fn verify_archive_checksum(archive: &Path) -> Result<(), String> {
    let checksum_path = PathBuf::from(format!("{}.sha256", archive.display()));
    let checksum = fs::read_to_string(&checksum_path)
        .map_err(|error| format!("cannot read {}: {error}", checksum_path.display()))?;
    let mut lines = checksum.lines();
    let line = lines
        .next()
        .ok_or_else(|| format!("{} is empty", checksum_path.display()))?;
    if lines.next().is_some() {
        return Err(format!(
            "{} must contain exactly one checksum",
            checksum_path.display()
        ));
    }
    let (expected, file_name) = line
        .split_once("  ")
        .ok_or_else(|| format!("invalid archive checksum in {}", checksum_path.display()))?;
    let expected_name = archive
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("invalid archive name {}", archive.display()))?;
    if file_name != expected_name {
        return Err(format!(
            "archive checksum names `{file_name}` instead of `{expected_name}`"
        ));
    }
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("archive checksum must be a 64-character SHA-256 digest".to_string());
    }
    let actual = hash_path(archive)?;
    if !expected.eq_ignore_ascii_case(&actual) {
        return Err(format!("archive checksum mismatch for {expected_name}"));
    }
    Ok(())
}

/// Extracts and verifies one artifact archive, returning its root directory.
fn verify_archive(archive: &Path, extract_dir: &Path) -> Result<PathBuf, String> {
    verify_archive_checksum(archive)?;
    remove_dir_if_present(extract_dir)?;
    fs::create_dir_all(extract_dir).map_err(|error| {
        format!(
            "cannot create extraction directory {}: {error}",
            extract_dir.display()
        )
    })?;
    command_output(
        Command::new("tar")
            .arg("--zstd")
            .arg("-xf")
            .arg(archive)
            .arg("-C")
            .arg(extract_dir),
        "extract artifact archive",
    )?;
    let roots = fs::read_dir(extract_dir)
        .map_err(|error| format!("cannot inspect {}: {error}", extract_dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot inspect extracted artifact: {error}"))?;
    if roots.len() != 1 || !roots[0].path().is_dir() {
        return Err("artifact must contain exactly one root directory".to_string());
    }
    let root = roots[0].path();
    verify_payload_checksums(&root)?;
    let manifest: ArtifactManifest = serde_json::from_slice(
        &fs::read(root.join("artifact.json"))
            .map_err(|error| format!("cannot read artifact manifest: {error}"))?,
    )
    .map_err(|error| format!("cannot parse artifact manifest: {error}"))?;
    if manifest.schema != ARTIFACT_SCHEMA {
        return Err(format!("unsupported artifact schema `{}`", manifest.schema));
    }
    Ok(root)
}

/// Verifies every listed checksum and rejects unlisted payload files.
fn verify_payload_checksums(root: &Path) -> Result<(), String> {
    let checksum_path = root.join("checksums.sha256");
    let checksum_text = fs::read_to_string(&checksum_path)
        .map_err(|error| format!("cannot read {}: {error}", checksum_path.display()))?;
    let mut expected = BTreeMap::new();
    for (line_number, line) in checksum_text.lines().enumerate() {
        let (digest, relative) = line
            .split_once("  ")
            .ok_or_else(|| format!("invalid checksum line {}", line_number + 1))?;
        let relative = PathBuf::from(relative);
        validate_relative_path(&relative)?;
        if expected.insert(relative, digest.to_string()).is_some() {
            return Err(format!("duplicate checksum line {}", line_number + 1));
        }
    }
    let actual = collect_payload_paths(root)?;
    let actual_set = actual.iter().cloned().collect::<BTreeSet<_>>();
    let expected_set = expected.keys().cloned().collect::<BTreeSet<_>>();
    if actual_set != expected_set {
        return Err("artifact payload does not match checksums.sha256".to_string());
    }
    for relative in actual {
        let digest = hash_path(&root.join(&relative))?;
        if expected.get(&relative) != Some(&digest) {
            return Err(format!("checksum mismatch for {}", relative.display()));
        }
    }
    Ok(())
}

/// Collects every payload path except the checksum index itself.
fn collect_payload_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.map_err(|error| format!("cannot walk {}: {error}", root.display()))?;
        if entry.path() == root || entry.file_type().is_dir() {
            continue;
        }
        if !entry.file_type().is_file() {
            return Err(format!(
                "unsupported payload entry {}",
                entry.path().display()
            ));
        }
        let relative = entry
            .path()
            .strip_prefix(root)
            .map_err(|error| format!("cannot relativize payload path: {error}"))?
            .to_path_buf();
        if relative != Path::new("checksums.sha256") {
            validate_relative_path(&relative)?;
            paths.push(relative);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Collects sorted relative regular-file paths from one directory.
fn collect_relative_files(directory: &Path, root: &Path) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    for entry in WalkDir::new(directory) {
        let entry =
            entry.map_err(|error| format!("cannot walk {}: {error}", directory.display()))?;
        if entry.file_type().is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .map_err(|error| format!("cannot relativize license path: {error}"))?;
            files.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    files.sort();
    Ok(files)
}

/// Rejects absolute paths, parent traversal, and platform prefixes.
fn validate_relative_path(path: &Path) -> Result<(), String> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!("unsafe artifact path `{}`", path.display()));
    }
    Ok(())
}

/// Computes SHA-256 over one regular file.
fn hash_path(path: &Path) -> Result<String, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    if metadata.is_file() {
        let mut file = fs::File::open(path)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    } else {
        return Err(format!("unsupported payload file {}", path.display()));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Copies one directory tree without following symbolic links.
fn copy_tree(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.is_dir() {
        return Err(format!("missing source directory {}", source.display()));
    }
    for entry in WalkDir::new(source).follow_links(false) {
        let entry = entry.map_err(|error| format!("cannot walk {}: {error}", source.display()))?;
        let relative = entry
            .path()
            .strip_prefix(source)
            .map_err(|error| format!("cannot relativize source path: {error}"))?;
        let target = destination.join(relative);
        if entry.file_type().is_dir() {
            fs::create_dir_all(&target)
                .map_err(|error| format!("cannot create {}: {error}", target.display()))?;
        } else if entry.file_type().is_file() {
            copy_file(entry.path(), &target)?;
        } else {
            return Err(format!(
                "unsupported source entry {}",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

/// Copies one required file and creates its parent directory.
fn copy_file(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.is_file() {
        return Err(format!("missing required file {}", source.display()));
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    fs::copy(source, destination).map_err(|error| {
        format!(
            "cannot copy {} to {}: {error}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

/// Copies one optional third-party license when it exists.
fn copy_optional_license(source: &Path, destination: &Path) -> Result<(), String> {
    if source.is_file() {
        copy_file(source, destination)?;
    }
    Ok(())
}

/// Copies one executable and preserves executable permissions.
fn copy_executable(source: &Path, destination: &Path) -> Result<(), String> {
    copy_file(source, destination)?;
    set_executable(destination)
}

/// Adds executable bits to one copied runtime file.
fn set_executable(path: &Path) -> Result<(), String> {
    let mut permissions = fs::metadata(path)
        .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?
        .permissions();
    permissions.set_mode(permissions.mode() | 0o111);
    fs::set_permissions(path, permissions)
        .map_err(|error| format!("cannot set executable mode on {}: {error}", path.display()))
}

/// Writes one complete file after creating its parent directory.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut file = fs::File::create(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// Reads and deserializes one TOML file.
fn read_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    toml::from_str(&text).map_err(|error| format!("cannot parse {}: {error}", path.display()))
}

/// Executes one Git command and returns trimmed UTF-8 standard output.
fn git_output(directory: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = command_output(
        Command::new("git").arg("-C").arg(directory).args(arguments),
        "run Git command",
    )?;
    Ok(output.trim().to_string())
}

/// Executes one child process and returns UTF-8 output on success.
fn command_output(command: &mut Command, action: &str) -> Result<String, String> {
    let output: Output = command
        .output()
        .map_err(|error| format!("cannot {action}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cannot {action}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{action} returned non-UTF-8 output: {error}"))
}

/// Removes one previous build directory while leaving absent paths untouched.
fn remove_dir_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot remove {}: {error}", path.display())),
    }
}

#[cfg(test)]
#[path = "main_test.rs"]
mod tests;
