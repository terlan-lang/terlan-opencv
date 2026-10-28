# OpenCV Promotion Check

This internal package validates the immutable inputs used to promote
`terlan-opencv`. It does not build OpenCV or execute Terlan programs.

## Responsibilities

- require committed compiler, package, and OpenCV repositories;
- reject tracked changes and untracked files before expensive native work;
- require the exact package-pinned OpenCV commit;
- write deterministic JSON evidence containing all three revisions;
- distinguish revision inventory from a completed fresh-workspace run.

`make opencv-promotion-tool-check` runs the unit and documentation gates.
`make opencv-promotion-inventory-check` applies the checker to the current
repositories. `make opencv-fresh-workspace-check` owns cloning, rebuilding,
consumer execution, and the final verified evidence flag through the
package-owned Terlan runner at `scripts/FreshWorkspaceCheck.terls`.

The `fresh_workspace_verified` field is an orchestration result. This crate
records it only when requested by the fresh-workspace runner; consumers must
not infer successful execution from clean revisions alone.
