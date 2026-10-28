# GitHub Workflows

The package CI separates repository-owned contracts from native and promotion
execution that require the OpenCV source and runtime.

## Package CI

`ci.yml` runs on pull requests and `main` pushes. It validates:

- C++ adapter validation helpers;
- deterministic ONNX fixtures;
- selected OpenCV surface and codec policy;
- committed normalized metadata;
- native install-layout expectations;
- the unlinked runtime guard;
- promotion revision tooling;
- every registered package-owned Terlan script.

This job builds the reviewed compiler revision from `upstream/terlan.lock.toml`
to check and run package-owned scripts. It does not download OpenCV or execute
the native package. Those operations belong to promotion and release jobs with
reviewed immutable inputs.

## Promotion

After repository history exists, `make opencv-fresh-workspace-check` clones
the exact local compiler, package, and pinned OpenCV commits. It rebuilds the
compiler-owned executables and executes the external-consumer package gate.
The package-owned `scripts/FreshWorkspaceCheck.terls` runner writes verified
evidence only after the consumer succeeds and removes its temporary workspace
on success or failure.

CI must not substitute a moving compiler branch for this contract.

## Package Release

`release.yml` builds the Linux x86-64 package from a stable package tag. The
workflow reads the exact compiler revision from `upstream/terlan.lock.toml` and
the exact OpenCV revision from `upstream/opencv.lock.toml`; neither dependency
may resolve through a moving branch or tag.

The build job has read-only repository access. It validates the stable package
version and tag, builds and deterministically replays the artifact, verifies
the external and internal checksums, executes the extracted-package consumer,
and records release-plan and promotion evidence. A manual workflow dispatch
runs this validation path without publishing.

Tag pushes proceed to a separate publication job. It refuses to replace an
existing release, attests the archive, creates a draft with the
archive, checksum, release plan, and promotion evidence already attached, and
then publishes the complete draft. GitHub release immutability must be enabled
for the repository before the first package release; without that repository
setting, the publication job is not an approved release path.

Before creating a release tag:

1. Commit all compiler and package changes.
2. Update `upstream/terlan.lock.toml` to the compiler commit that passes the
   artifact consumer gate.
3. Replace the package development version with the stable release version.
4. Run `make opencv-package-artifact-check` from a clean package checkout.
5. Dispatch `OpenCV Package Release` manually with the intended tag and retain
   the successful validation run.
6. Create and push that exact tag only after release immutability is enabled.
