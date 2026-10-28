//! Tests for OpenCV runtime availability containment.

use std::ffi::OsStr;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use super::{bundled_worker_path, request_id, unavailable_reply, RuntimeGuard};

/// Resolves the worker beside the installed runtime guard.
#[test]
fn bundled_worker_uses_guard_directory() {
    assert_eq!(
        bundled_worker_path(Path::new("/artifact/bin/terlan-opencv-runtime-guard")),
        Some(PathBuf::from("/artifact/bin/terlan-opencv-worker"))
    );
}

/// Extracts only valid correlated call identities.
#[test]
fn request_identity_rejects_malformed_input() {
    assert_eq!(request_id("call 42 operation"), 42);
    assert_eq!(request_id("reply 42 operation"), 0);
    assert_eq!(request_id("call invalid operation"), 0);
    assert_eq!(request_id("call"), 0);
}

/// Emits the stable package error with no host-specific loader detail.
#[test]
fn unavailable_reply_is_stable_and_correlated() {
    assert_eq!(
        unavailable_reply(7),
        format!(
            "reply 7 1 err {} {}",
            STANDARD.encode("opencv.unavailable"),
            STANDARD.encode("OpenCV native runtime is unavailable")
        )
    );
}

/// Keeps serving correlated errors when no worker can be started.
#[test]
fn missing_worker_does_not_terminate_guard() {
    let mut guard = RuntimeGuard::new(Some(OsStr::new("/definitely/missing/terlan-opencv-worker")));
    let mut output = Vec::new();
    guard
        .run(Cursor::new("call 1 first\n\ncall 2 second\n"), &mut output)
        .expect("guard must contain missing worker");
    let output = String::from_utf8(output).expect("guard output is UTF-8");
    assert_eq!(output.lines().count(), 2);
    assert!(output
        .lines()
        .next()
        .expect("first reply")
        .starts_with("reply 1 1 err "));
    assert!(output
        .lines()
        .nth(1)
        .expect("second reply")
        .starts_with("reply 2 1 err "));
}

/// Transparently forwards requests and replies from a healthy worker.
#[test]
fn healthy_worker_is_transparent() {
    let mut guard = RuntimeGuard::new(Some(OsStr::new("cat")));
    assert!(
        guard.worker.is_some(),
        "test requires the standard cat utility"
    );
    let mut output = Vec::new();
    guard
        .run(Cursor::new("reply 3 1 ok_unit\n"), &mut output)
        .expect("guard must proxy healthy worker");
    assert_eq!(output, b"reply 3 1 ok_unit\n");
}

/// Converts a worker that exits before replying into package unavailability.
#[test]
fn early_worker_exit_is_contained() {
    let mut guard = RuntimeGuard::new(Some(OsStr::new("false")));
    assert!(
        guard.worker.is_some(),
        "test requires the standard false utility"
    );
    let mut output = Vec::new();
    guard
        .run(Cursor::new("call 9 operation\n"), &mut output)
        .expect("guard must contain early worker exit");
    assert!(String::from_utf8(output)
        .expect("guard output is UTF-8")
        .starts_with("reply 9 1 err "));
}
