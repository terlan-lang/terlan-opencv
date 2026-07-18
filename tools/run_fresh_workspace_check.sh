#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 COMPILER_REPO PACKAGE_REPO OPENCV_REPO EXPECTED_OPENCV_COMMIT REPORT" >&2
    exit 2
fi

compiler_repo=$(realpath "$1")
package_repo=$(realpath "$2")
opencv_repo=$(realpath "$3")
expected_opencv_commit=$4
report=$(realpath -m "$5")
checker_manifest="$package_repo/tools/promotion_check/Cargo.toml"
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT

# Reject mutable inputs before cloning or compiling native code.
cargo run --quiet --manifest-path "$checker_manifest" -- \
    --compiler "$compiler_repo" \
    --package "$package_repo" \
    --opencv "$opencv_repo" \
    --expected-opencv-commit "$expected_opencv_commit" \
    --report "$workspace/inventory.json"

git clone --quiet --no-hardlinks "$compiler_repo" "$workspace/terlan"
git clone --quiet --no-hardlinks "$package_repo" "$workspace/terlan-opencv"
git clone --quiet --no-hardlinks "$opencv_repo" "$workspace/opencv"

# Rebuild every compiler-owned executable consumed by the package gate.
cargo build --quiet --manifest-path "$workspace/terlan/Cargo.toml" \
    -p terlan --bin terlc --bin terlan-vm
cmake -S "$workspace/terlan/tools/cpp_metadata_extractor" \
    -B "$workspace/terlan/target/cpp-metadata-extractor"
cmake --build "$workspace/terlan/target/cpp-metadata-extractor" --parallel

make -C "$workspace/terlan-opencv" terlan-opencv-package-check \
    OPENCV_SOURCE="$workspace/opencv" \
    TERLC="$workspace/terlan/target/debug/terlc" \
    CPP_METADATA_EXTRACTOR="$workspace/terlan/target/cpp-metadata-extractor/terlan-cpp-metadata-extractor"

# Publish verified evidence only after the external consumer succeeds.
cargo run --quiet --manifest-path "$checker_manifest" -- \
    --compiler "$compiler_repo" \
    --package "$package_repo" \
    --opencv "$opencv_repo" \
    --expected-opencv-commit "$expected_opencv_commit" \
    --report "$report" \
    --verified

echo "opencv-fresh-workspace-check: consumer reproduced from committed revisions"
