//! Revision and evidence validation for OpenCV package promotion.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Stable schema written by the OpenCV promotion gate.
pub const PROMOTION_SCHEMA: &str = "terlan.opencv.promotion.v1";

/// One immutable repository identity used by promotion.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RepositoryRevision {
    /// Logical role of the repository in the promotion run.
    pub role: String,
    /// Canonical source path inspected by the gate.
    pub path: PathBuf,
    /// Full committed Git object identity.
    pub commit: String,
    /// Whether the tracked and untracked worktree was empty.
    pub clean: bool,
}

/// Reproducibility evidence for one OpenCV promotion attempt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromotionEvidence {
    /// Evidence document schema.
    pub schema: String,
    /// Exact Terlan compiler repository identity.
    pub compiler: RepositoryRevision,
    /// Exact `terlan-opencv` package repository identity.
    pub package: RepositoryRevision,
    /// Exact upstream OpenCV repository identity.
    pub opencv: RepositoryRevision,
    /// Whether the fresh-workspace consumer execution completed.
    pub fresh_workspace_verified: bool,
}

/// Inspects all promotion repositories and validates their immutable contract.
pub fn collect_promotion_evidence(
    compiler: &Path,
    package: &Path,
    opencv: &Path,
    expected_opencv_commit: &str,
    fresh_workspace_verified: bool,
) -> Result<PromotionEvidence, String> {
    require_full_commit(expected_opencv_commit, "expected OpenCV commit")?;
    let package = inspect_repository("package", package)?;
    let compiler = inspect_repository("compiler", compiler)?;
    let opencv = inspect_repository("opencv", opencv)?;
    if opencv.commit != expected_opencv_commit {
        return Err(format!(
            "error[opencv_promotion_revision]: OpenCV commit `{}` does not match pinned commit `{expected_opencv_commit}`",
            opencv.commit
        ));
    }
    Ok(PromotionEvidence {
        schema: PROMOTION_SCHEMA.to_string(),
        compiler,
        package,
        opencv,
        fresh_workspace_verified,
    })
}

/// Writes deterministic, pretty-printed promotion evidence.
pub fn write_promotion_evidence(
    evidence: &PromotionEvidence,
    destination: &Path,
) -> Result<(), String> {
    let parent = destination.parent().ok_or_else(|| {
        format!(
            "error[opencv_promotion_report]: report path `{}` has no parent",
            destination.display()
        )
    })?;
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "error[opencv_promotion_report]: cannot create `{}`: {error}",
            parent.display()
        )
    })?;
    let mut bytes = serde_json::to_vec_pretty(evidence).map_err(|error| {
        format!("error[opencv_promotion_report]: cannot encode evidence: {error}")
    })?;
    bytes.push(b'\n');
    fs::write(destination, bytes).map_err(|error| {
        format!(
            "error[opencv_promotion_report]: cannot write `{}`: {error}",
            destination.display()
        )
    })
}

/// Reads and validates one clean repository with a committed HEAD.
fn inspect_repository(role: &str, path: &Path) -> Result<RepositoryRevision, String> {
    let canonical = path.canonicalize().map_err(|error| {
        format!(
            "error[opencv_promotion_repository]: {role} repository `{}` is unavailable: {error}",
            path.display()
        )
    })?;
    let commit = git_output(&canonical, &["rev-parse", "--verify", "HEAD"]).map_err(|error| {
        format!(
            "error[opencv_promotion_repository]: {role} repository `{}` has no committed HEAD: {error}",
            canonical.display()
        )
    })?;
    require_full_commit(&commit, &format!("{role} HEAD"))?;
    let status = git_output(
        &canonical,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        let first = status.lines().next().unwrap_or("unknown change");
        return Err(format!(
            "error[opencv_promotion_dirty]: {role} repository `{}` is not clean; first change: {first}",
            canonical.display()
        ));
    }
    Ok(RepositoryRevision {
        role: role.to_string(),
        path: canonical,
        commit,
        clean: true,
    })
}

/// Executes Git in one repository and returns trimmed standard output.
fn git_output(repository: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .map_err(|error| {
            format!(
                "error[opencv_promotion_git]: cannot execute git for `{}`: {error}",
                repository.display()
            )
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!(
                "error[opencv_promotion_git]: git {} failed for `{}`",
                arguments.join(" "),
                repository.display()
            )
        } else {
            stderr
        });
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| {
            format!(
                "error[opencv_promotion_git]: git output for `{}` is not UTF-8: {error}",
                repository.display()
            )
        })
}

/// Requires a full lowercase SHA-1 object identity.
fn require_full_commit(commit: &str, label: &str) -> Result<(), String> {
    let valid = commit.len() == 40
        && commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(format!(
            "error[opencv_promotion_revision]: {label} must be a full lowercase Git SHA-1"
        ))
    }
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;
