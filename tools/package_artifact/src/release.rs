//! Stable release identity and artifact validation.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{git_output, read_toml, verify_archive, ArtifactManifest, RootManifest, UpstreamLock};

/// Parsed immutable Terlan compiler source identity.
#[derive(Debug, Deserialize)]
struct CompilerLock {
    /// Compiler source repository.
    repository: String,
    /// Exact compiler Git revision.
    commit: String,
}

/// Machine-readable immutable inputs for one release build.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ReleasePlan {
    /// Package release tag.
    tag: String,
    /// Package source revision being released.
    revision: String,
    /// Target artifact triple.
    target: String,
    /// Expected archive file name.
    archive: String,
    /// Exact compiler repository.
    compiler_repository: String,
    /// Exact compiler revision.
    compiler_revision: String,
    /// Exact OpenCV repository.
    opencv_repository: String,
    /// Exact OpenCV revision.
    opencv_revision: String,
}

/// Builds and validates the immutable input plan for one release tag.
pub(super) fn build_release_plan(
    root: &Path,
    tag: &str,
    revision: &str,
    target: &str,
) -> Result<ReleasePlan, String> {
    validate_release_revision(revision)?;
    validate_release_target(target)?;
    let package = read_toml::<RootManifest>(&root.join("terlan.toml"))?;
    validate_release_tag(tag, &package.package.version)?;
    let actual_revision = git_output(root, &["rev-parse", "HEAD"])?;
    if actual_revision != revision {
        return Err(format!(
            "release revision mismatch: expected `{revision}`, found `{actual_revision}`"
        ));
    }
    let dirty = git_output(root, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    if !dirty.is_empty() {
        return Err(format!(
            "release source must be clean; first change: {}",
            dirty.lines().next().unwrap_or("unknown change")
        ));
    }
    let compiler = read_toml::<CompilerLock>(&root.join("upstream/terlan.lock.toml"))?;
    validate_release_revision(&compiler.commit)?;
    let opencv = read_toml::<UpstreamLock>(&root.join("upstream/opencv.lock.toml"))?;
    validate_release_revision(&opencv.commit)?;
    Ok(ReleasePlan {
        tag: tag.to_string(),
        revision: revision.to_string(),
        target: target.to_string(),
        archive: format!(
            "{}-{}-{target}.tar.zst",
            package.package.name, package.package.version
        ),
        compiler_repository: compiler.repository,
        compiler_revision: compiler.commit,
        opencv_repository: opencv.repository,
        opencv_revision: opencv.commit,
    })
}

/// Requires a stable semantic-version tag matching the package version.
pub(super) fn validate_release_tag(tag: &str, version: &str) -> Result<(), String> {
    if version.contains('-') || version.contains('+') {
        return Err(format!(
            "release package version `{version}` must not contain prerelease or build metadata"
        ));
    }
    let components = version.split('.').collect::<Vec<_>>();
    if components.len() != 3
        || components.iter().any(|component| {
            component.is_empty() || !component.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return Err(format!(
            "release package version `{version}` must be numeric major.minor.patch"
        ));
    }
    let expected = format!("v{version}");
    if tag != expected {
        return Err(format!(
            "release tag mismatch: expected `{expected}`, found `{tag}`"
        ));
    }
    Ok(())
}

/// Requires a full lowercase SHA-1 Git object identity.
pub(super) fn validate_release_revision(revision: &str) -> Result<(), String> {
    let valid = revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err("release revision must be a full lowercase Git SHA-1".to_string())
    }
}

/// Requires a currently supported release target.
pub(super) fn validate_release_target(target: &str) -> Result<(), String> {
    if target == "x86_64-unknown-linux-gnu" {
        Ok(())
    } else {
        Err(format!("unsupported artifact target `{target}`"))
    }
}

/// Verifies an artifact and binds it to its release tag, source, and target.
pub(super) fn verify_release_archive(
    archive: &Path,
    extract_dir: &Path,
    tag: &str,
    revision: &str,
    target: &str,
) -> Result<PathBuf, String> {
    validate_release_revision(revision)?;
    validate_release_target(target)?;
    let root = verify_archive(archive, extract_dir)?;
    let manifest: ArtifactManifest = serde_json::from_slice(
        &fs::read(root.join("artifact.json"))
            .map_err(|error| format!("cannot read artifact manifest: {error}"))?,
    )
    .map_err(|error| format!("cannot parse artifact manifest: {error}"))?;
    validate_release_tag(tag, &manifest.package.version)?;
    if manifest.package.dirty {
        return Err("release artifact was built from a dirty package source".to_string());
    }
    if manifest.package.revision != revision {
        return Err(format!(
            "release artifact revision mismatch: expected `{revision}`, found `{}`",
            manifest.package.revision
        ));
    }
    if manifest.target != target {
        return Err(format!(
            "release artifact target mismatch: expected `{target}`, found `{}`",
            manifest.target
        ));
    }
    let expected_name = format!(
        "{}-{}-{target}.tar.zst",
        manifest.package.name, manifest.package.version
    );
    let actual_name = archive
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("invalid archive name {}", archive.display()))?;
    if actual_name != expected_name {
        return Err(format!(
            "release artifact name mismatch: expected `{expected_name}`, found `{actual_name}`"
        ));
    }
    let root_name = root
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| format!("invalid artifact root {}", root.display()))?;
    if root_name != expected_name.trim_end_matches(".tar.zst") {
        return Err(format!(
            "release artifact root mismatch: expected `{}`, found `{root_name}`",
            expected_name.trim_end_matches(".tar.zst")
        ));
    }
    Ok(root)
}
