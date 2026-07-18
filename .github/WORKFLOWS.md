# GitHub Workflows

The package CI separates repository-owned offline contracts from native and
promotion execution that require exact external revisions.

## Package CI

`ci.yml` runs on pull requests and `main` pushes. It validates:

- C++ adapter validation helpers;
- deterministic ONNX fixtures;
- selected OpenCV surface and codec policy;
- committed normalized metadata;
- native install-layout expectations;
- the unlinked runtime guard;
- promotion revision tooling and runner syntax.

This job does not download OpenCV, compile the Terlan compiler, or execute the
native package. Those operations would be misleading until the package and
compiler have committed revisions that can be recorded as promotion inputs.

## Promotion

After repository history exists, `make opencv-fresh-workspace-check` clones
the exact local compiler, package, and pinned OpenCV commits. It rebuilds the
compiler-owned executables and executes the external-consumer package gate.
The command writes verified evidence only after the consumer succeeds.

The fresh-workspace target should become a required CI job only after the
compiler and package revisions are pinned by release automation. CI must not
substitute a moving compiler branch for that contract.
