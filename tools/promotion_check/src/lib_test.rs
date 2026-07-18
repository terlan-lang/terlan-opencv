use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{collect_promotion_evidence, write_promotion_evidence, PROMOTION_SCHEMA};

/// Temporary filesystem root removed after each test.
struct TestWorkspace(PathBuf);

impl TestWorkspace {
    /// Creates a collision-resistant temporary test workspace.
    fn new(name: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "terlan-opencv-promotion-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create test workspace");
        Self(path)
    }

    /// Returns the temporary workspace path.
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Executes one Git fixture command and returns standard output.
fn git(repository: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .expect("execute git");
    assert!(
        output.status.success(),
        "git {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("UTF-8 git output")
        .trim()
        .to_string()
}

/// Creates a clean repository containing one committed file.
fn committed_repository(parent: &Path, name: &str) -> (PathBuf, String) {
    let repository = parent.join(name);
    fs::create_dir_all(&repository).expect("create repository");
    git(&repository, &["init", "--quiet"]);
    git(&repository, &["config", "user.name", "Terlan Test"]);
    git(
        &repository,
        &["config", "user.email", "tests@terlan.invalid"],
    );
    fs::write(repository.join("README.md"), format!("# {name}\n")).expect("write fixture");
    git(&repository, &["add", "README.md"]);
    git(&repository, &["commit", "--quiet", "-m", "fixture"]);
    let commit = git(&repository, &["rev-parse", "HEAD"]);
    (repository, commit)
}

/// Records exact identities from three clean repositories.
#[test]
fn collects_exact_clean_repository_revisions() {
    let workspace = TestWorkspace::new("clean");
    let (compiler, compiler_commit) = committed_repository(workspace.path(), "compiler");
    let (package, package_commit) = committed_repository(workspace.path(), "package");
    let (opencv, opencv_commit) = committed_repository(workspace.path(), "opencv");

    let evidence = collect_promotion_evidence(&compiler, &package, &opencv, &opencv_commit, true)
        .expect("valid promotion evidence");

    assert_eq!(evidence.schema, PROMOTION_SCHEMA);
    assert_eq!(evidence.compiler.commit, compiler_commit);
    assert_eq!(evidence.package.commit, package_commit);
    assert_eq!(evidence.opencv.commit, opencv_commit);
    assert!(evidence.fresh_workspace_verified);
}

/// Refuses an initialized repository that has no committed HEAD.
#[test]
fn rejects_repository_without_committed_head() {
    let workspace = TestWorkspace::new("missing-head");
    let (compiler, _) = committed_repository(workspace.path(), "compiler");
    let package = workspace.path().join("package");
    fs::create_dir_all(&package).expect("create package");
    git(&package, &["init", "--quiet"]);
    let (opencv, opencv_commit) = committed_repository(workspace.path(), "opencv");

    let error = collect_promotion_evidence(&compiler, &package, &opencv, &opencv_commit, false)
        .expect_err("missing HEAD must fail");

    assert!(error.contains("package repository"));
    assert!(error.contains("has no committed HEAD"));
}

/// Refuses tracked or untracked worktree changes.
#[test]
fn rejects_dirty_or_untracked_repository_content() {
    let workspace = TestWorkspace::new("dirty");
    let (compiler, _) = committed_repository(workspace.path(), "compiler");
    let (package, _) = committed_repository(workspace.path(), "package");
    let (opencv, opencv_commit) = committed_repository(workspace.path(), "opencv");
    fs::write(package.join("untracked.txt"), "not committed\n").expect("dirty package");

    let error = collect_promotion_evidence(&compiler, &package, &opencv, &opencv_commit, false)
        .expect_err("dirty package must fail");

    assert!(error.contains("error[opencv_promotion_dirty]"));
    assert!(error.contains("untracked.txt"));
}

/// Refuses an OpenCV revision different from the requested pin.
#[test]
fn rejects_unexpected_opencv_revision() {
    let workspace = TestWorkspace::new("wrong-opencv");
    let (compiler, _) = committed_repository(workspace.path(), "compiler");
    let (package, _) = committed_repository(workspace.path(), "package");
    let (opencv, _) = committed_repository(workspace.path(), "opencv");
    let wrong_commit = "0000000000000000000000000000000000000000";

    let error = collect_promotion_evidence(&compiler, &package, &opencv, wrong_commit, false)
        .expect_err("wrong OpenCV revision must fail");

    assert!(error.contains("does not match pinned commit"));
}

/// Writes evidence that can be decoded without information loss.
#[test]
fn writes_round_trippable_promotion_evidence() {
    let workspace = TestWorkspace::new("report");
    let (compiler, _) = committed_repository(workspace.path(), "compiler");
    let (package, _) = committed_repository(workspace.path(), "package");
    let (opencv, opencv_commit) = committed_repository(workspace.path(), "opencv");
    let evidence = collect_promotion_evidence(&compiler, &package, &opencv, &opencv_commit, false)
        .expect("inventory evidence");
    let report = workspace.path().join("reports/promotion.json");

    write_promotion_evidence(&evidence, &report).expect("write evidence");
    let decoded: super::PromotionEvidence =
        serde_json::from_slice(&fs::read(report).expect("read report")).expect("decode report");

    assert_eq!(decoded, evidence);
    assert!(!decoded.fresh_workspace_verified);
}
