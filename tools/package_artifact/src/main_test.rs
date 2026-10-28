//! Unit tests for package artifact validation helpers.

use super::*;

/// Returns one valid release artifact manifest fixture.
fn release_manifest(version: &str, revision: &str, target: &str) -> ArtifactManifest {
    ArtifactManifest {
        schema: ARTIFACT_SCHEMA.to_string(),
        target: target.to_string(),
        package: PackageIdentity {
            name: "terlan-opencv".to_string(),
            version: version.to_string(),
            revision: revision.to_string(),
            dirty: false,
        },
        opencv: OpenCvIdentity {
            version: "4.13.0".to_string(),
            revision: "1111111111111111111111111111111111111111".to_string(),
        },
        terlan_package: "package".to_string(),
        runtime: RuntimeLayout {
            guard: "bin/guard".to_string(),
            worker: "bin/worker".to_string(),
            library_dir: "lib".to_string(),
            environment: BTreeMap::new(),
        },
        licenses: vec!["licenses/LICENSE".to_string()],
    }
}

/// Executes one Git command for a release-plan fixture.
fn fixture_git(repository: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .expect("execute fixture git");
    assert!(
        output.status.success(),
        "git {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("UTF-8 fixture git output")
        .trim()
        .to_string()
}

/// Parses linked, direct-loader, and virtual dynamic dependency rows.
#[test]
fn ldd_names_are_normalized() {
    let names = parse_ldd_names(
        "linux-vdso.so.1 (0x1)\nlibopencv_core.so.413 => /tmp/libopencv_core.so.413 (0x2)\n/lib64/ld-linux-x86-64.so.2 (0x3)\n",
    );
    assert_eq!(
        names,
        BTreeSet::from([
            "ld-linux-x86-64.so.2".to_string(),
            "libopencv_core.so.413".to_string(),
            "linux-vdso.so.1".to_string(),
        ])
    );
}

/// Accepts origin-relative runtime paths and rejects host or working paths.
#[test]
fn dynamic_search_paths_must_be_origin_relative() {
    let origin = "0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN:$ORIGIN/private]\n";
    assert_eq!(
        parse_dynamic_search_paths(origin),
        BTreeSet::from(["$ORIGIN:$ORIGIN/private".to_string()])
    );
    assert!(validate_runtime_search_paths(Path::new("lib.so"), origin).is_ok());

    for unsafe_path in [
        "0x000000000000001d (RUNPATH) Library runpath: [/tmp/build]\n",
        "0x000000000000000f (RPATH) Library rpath: [relative]\n",
    ] {
        assert!(validate_runtime_search_paths(Path::new("lib.so"), unsafe_path).is_err());
    }
}

/// Finds exact embedded byte sequences without treating binaries as text.
#[test]
fn embedded_path_search_handles_binary_payloads() {
    assert!(contains_bytes(b"\0/home/build\0", b"/home/build"));
    assert!(!contains_bytes(b"\0artifact\0", b"/home/build"));
    assert!(!contains_bytes(b"artifact", b""));
}

/// Scrubs every matching path while preserving binary payload length.
#[test]
fn embedded_path_sanitization_preserves_layout() {
    let mut bytes = b"/tmp/root/a\0/tmp/root/b".to_vec();
    let original_length = bytes.len();
    let replacement = padded_replacement(b"source", b"/tmp/root".len());
    assert!(replace_bytes(&mut bytes, b"/tmp/root", &replacement));
    assert_eq!(bytes.len(), original_length);
    assert!(!contains_bytes(&bytes, b"/tmp/root"));
    assert_eq!(bytes, b"source___/a\0source___/b");
    assert!(!replace_bytes(&mut bytes, b"different", b"short"));
}

/// Rejects empty, absolute, prefixed, and parent-traversing payload paths.
#[test]
fn relative_path_validation_rejects_unsafe_paths() {
    assert!(validate_relative_path(Path::new("package/src/Main.terl")).is_ok());
    for path in ["", "/etc/passwd", "package/../secret"] {
        assert!(validate_relative_path(Path::new(path)).is_err(), "{path}");
    }
}

/// Rejects links from the payload accepted by the package fetcher.
#[test]
fn payload_hashes_reject_links() {
    let root = env::temp_dir().join(format!(
        "terlan-opencv-artifact-hash-{}",
        std::process::id()
    ));
    remove_dir_if_present(&root).expect("remove stale fixture");
    fs::create_dir_all(&root).expect("create fixture");
    write_file(&root.join("payload"), b"opencv").expect("write payload");
    std::os::unix::fs::symlink("payload", root.join("link")).expect("create link");
    assert!(hash_path(&root.join("payload")).is_ok());
    assert!(hash_path(&root.join("link")).is_err());
    assert!(collect_payload_paths(&root).is_err());
    remove_dir_if_present(&root).expect("remove fixture");
}

/// Detects payload tampering after checksum generation.
#[test]
fn checksum_verification_rejects_modified_payload() {
    let root = env::temp_dir().join(format!(
        "terlan-opencv-artifact-check-{}",
        std::process::id()
    ));
    remove_dir_if_present(&root).expect("remove stale fixture");
    fs::create_dir_all(&root).expect("create fixture");
    write_file(&root.join("artifact.json"), b"{}\n").expect("write manifest");
    write_payload_checksums(&root).expect("write checksums");
    verify_payload_checksums(&root).expect("verify pristine payload");
    write_file(&root.join("artifact.json"), b"tampered\n").expect("tamper payload");
    assert!(verify_payload_checksums(&root).is_err());
    remove_dir_if_present(&root).expect("remove fixture");
}

/// Accepts only stable numeric tags matching the package version exactly.
#[test]
fn release_tags_are_exact_and_stable() {
    assert!(release::validate_release_tag("v0.1.0", "0.1.0").is_ok());
    for (tag, version) in [
        ("0.1.0", "0.1.0"),
        ("v0.1.1", "0.1.0"),
        ("v0.1.0-dev", "0.1.0-dev"),
        ("v1.0", "1.0"),
        ("v1.x.0", "1.x.0"),
    ] {
        assert!(
            release::validate_release_tag(tag, version).is_err(),
            "{tag} {version}"
        );
    }
}

/// Rejects abbreviated, uppercase, malformed, and unsupported release inputs.
#[test]
fn release_revision_and_target_are_strict() {
    assert!(release::validate_release_revision("0123456789abcdef0123456789abcdef01234567").is_ok());
    for revision in [
        "0123456",
        "0123456789ABCDEF0123456789ABCDEF01234567",
        "z123456789abcdef0123456789abcdef01234567",
    ] {
        assert!(
            release::validate_release_revision(revision).is_err(),
            "{revision}"
        );
    }
    assert!(release::validate_release_target("x86_64-unknown-linux-gnu").is_ok());
    assert!(release::validate_release_target("aarch64-unknown-linux-gnu").is_err());
}

/// Verifies the external checksum filename, digest shape, and archive bytes.
#[test]
fn external_archive_checksum_is_mandatory_and_exact() {
    let root = env::temp_dir().join(format!(
        "terlan-opencv-external-checksum-{}",
        std::process::id()
    ));
    remove_dir_if_present(&root).expect("remove stale fixture");
    fs::create_dir_all(&root).expect("create fixture");
    let archive = root.join("artifact.tar.zst");
    write_file(&archive, b"archive").expect("write archive");
    assert!(verify_archive_checksum(&archive).is_err());

    write_archive_checksum(&archive).expect("write archive checksum");
    verify_archive_checksum(&archive).expect("verify archive checksum");
    write_file(&archive, b"tampered").expect("tamper archive");
    assert!(verify_archive_checksum(&archive).is_err());

    write_file(&archive, b"archive").expect("restore archive");
    write_file(
        &PathBuf::from(format!("{}.sha256", archive.display())),
        b"00  wrong.tar.zst\n",
    )
    .expect("write malformed checksum");
    assert!(verify_archive_checksum(&archive).is_err());
    remove_dir_if_present(&root).expect("remove fixture");
}

/// Keeps release manifest fixtures serializable for archive validation tests.
#[test]
fn release_manifest_fixture_round_trips() {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let manifest = release_manifest("0.1.0", revision, "x86_64-unknown-linux-gnu");
    let encoded = serde_json::to_vec(&manifest).expect("encode manifest");
    let decoded: ArtifactManifest = serde_json::from_slice(&encoded).expect("decode manifest");
    assert_eq!(decoded, manifest);
}

/// Resolves reviewed compiler and OpenCV pins from a clean release source.
#[test]
fn release_plan_binds_clean_source_and_dependency_revisions() {
    let root = env::temp_dir().join(format!("terlan-opencv-release-plan-{}", std::process::id()));
    remove_dir_if_present(&root).expect("remove stale fixture");
    fs::create_dir_all(root.join("upstream")).expect("create fixture");
    write_file(
        &root.join("terlan.toml"),
        b"[package]\nname = \"terlan-opencv\"\nversion = \"0.1.0\"\nnamespace = \"opencv\"\n",
    )
    .expect("write package manifest");
    write_file(
        &root.join("upstream/terlan.lock.toml"),
        b"repository = \"https://example.invalid/terlan.git\"\ncommit = \"0123456789abcdef0123456789abcdef01234567\"\n",
    )
    .expect("write compiler lock");
    write_file(
        &root.join("upstream/opencv.lock.toml"),
        b"repository = \"https://example.invalid/opencv.git\"\ntag = \"4.13.0\"\ncommit = \"1111111111111111111111111111111111111111\"\n",
    )
    .expect("write OpenCV lock");
    fixture_git(&root, &["init", "--quiet"]);
    fixture_git(&root, &["add", "."]);
    fixture_git(
        &root,
        &[
            "-c",
            "user.name=Terlan Test",
            "-c",
            "user.email=tests@terlan.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    );
    let revision = fixture_git(&root, &["rev-parse", "HEAD"]);

    let plan = build_release_plan(&root, "v0.1.0", &revision, "x86_64-unknown-linux-gnu")
        .expect("build release plan");
    let plan = serde_json::to_value(plan).expect("encode release plan");
    assert_eq!(plan["revision"], revision);
    assert_eq!(
        plan["archive"],
        "terlan-opencv-0.1.0-x86_64-unknown-linux-gnu.tar.zst"
    );
    assert_eq!(
        plan["compiler_revision"],
        "0123456789abcdef0123456789abcdef01234567"
    );

    write_file(&root.join("untracked"), b"dirty\n").expect("dirty fixture");
    assert!(build_release_plan(&root, "v0.1.0", &revision, "x86_64-unknown-linux-gnu").is_err());
    remove_dir_if_present(&root).expect("remove fixture");
}

/// Validates a complete release archive and rejects an unrelated source SHA.
#[test]
fn release_archive_is_bound_to_tag_revision_target_and_name() {
    let root = env::temp_dir().join(format!(
        "terlan-opencv-release-archive-{}",
        std::process::id()
    ));
    remove_dir_if_present(&root).expect("remove stale fixture");
    fs::create_dir_all(&root).expect("create fixture");
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let target = "x86_64-unknown-linux-gnu";
    let name = format!("terlan-opencv-0.1.0-{target}");
    let staging = root.join(&name);
    fs::create_dir_all(staging.join("licenses")).expect("create staging");
    let manifest = release_manifest("0.1.0", revision, target);
    write_file(
        &staging.join("artifact.json"),
        &serde_json::to_vec_pretty(&manifest).expect("encode manifest"),
    )
    .expect("write manifest");
    write_file(&staging.join("licenses/LICENSE"), b"fixture\n").expect("write license");
    write_payload_checksums(&staging).expect("write payload checksums");
    let archive = root.join(format!("{name}.tar.zst"));
    create_archive(&root, &name, &archive).expect("create archive");
    write_archive_checksum(&archive).expect("write archive checksum");

    verify_release_archive(&archive, &root.join("extract"), "v0.1.0", revision, target)
        .expect("verify release archive");
    assert!(verify_release_archive(
        &archive,
        &root.join("wrong-revision"),
        "v0.1.0",
        "2222222222222222222222222222222222222222",
        target,
    )
    .is_err());
    remove_dir_if_present(&root).expect("remove fixture");
}
