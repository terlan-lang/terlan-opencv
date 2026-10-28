OPENCV_SOURCE ?= upstream/opencv
OPENCV_BUILD ?= upstream/build
OPENCV_INSTALL ?= upstream/install
OPENCV_PINNED_COMMIT := fe38fc608f6acb8b68953438a62305d8318f4fcd
CXX ?= c++
ASAN_STATIC_RUNTIME ?= $(shell $(CXX) -print-file-name=libasan.a)
UBSAN_STATIC_RUNTIME ?= $(shell $(CXX) -print-file-name=libubsan.a)
SANITIZER_OPTIONS ?= detect_leaks=1
TERLC ?= ../terlan/target/debug/terlc
SURFACE_CHECK_MANIFEST := tools/surface_check/Cargo.toml
NATIVE_CHECK_MANIFEST := tools/native_adapter_check/Cargo.toml
NATIVE_DIRECT_MANIFEST := tools/native_direct_check/Cargo.toml
NATIVE_INSTALL_CHECK_MANIFEST := tools/native_install_check/Cargo.toml
CODEC_POLICY_CHECK_MANIFEST := tools/codec_policy_check/Cargo.toml
METADATA_CHECK_MANIFEST := tools/metadata_check/Cargo.toml
PROMOTION_CHECK_MANIFEST := tools/promotion_check/Cargo.toml
PACKAGE_ARTIFACT_MANIFEST := tools/package_artifact/Cargo.toml
API_PARITY_MANIFEST := tools/api_parity/Cargo.toml
BEHAVIORAL_PARITY_MANIFEST := tools/behavioral_parity/Cargo.toml
RUNTIME_GUARD_MANIFEST := native/runtime_guard/Cargo.toml
CPP_METADATA_EXTRACTOR ?= ../terlan/target/cpp-metadata-extractor/terlan-cpp-metadata-extractor
NATIVE_MANIFEST := bindings/opencv.native-binding.json
GENERATED_NATIVE := generated/native/rust
NATIVE_HELPER := $(GENERATED_NATIVE)/target/debug/native-boundary-helper
RUNTIME_GUARD := native/runtime_guard/target/debug/terlan-opencv-runtime-guard
PROMOTION_REPORT ?= _build/promotion/opencv-promotion.json
SCRIPT_BUILD := _build/scripts
TERLAN_SCRIPT = "$(TERLC)" --out-dir "$(abspath $(SCRIPT_BUILD))" run script
PACKAGE_ARTIFACT_TARGET ?= x86_64-unknown-linux-gnu
PACKAGE_ARTIFACT_DIR ?= _build/artifacts
PACKAGE_ARTIFACT_REPLAY_DIR ?= _build/artifacts-replay
RUST_REMAP_FLAGS := --remap-path-prefix=$(CURDIR)=terlan-opencv --remap-path-prefix=$(HOME)=home
OPENCV_REMAP_FLAGS := -ffile-prefix-map=$(abspath $(OPENCV_SOURCE))=opencv -ffile-prefix-map=$(abspath $(OPENCV_BUILD))=build
BRIDGE_REMAP_FLAGS := -ffile-prefix-map=$(CURDIR)=terlan-opencv $(OPENCV_REMAP_FLAGS)

.PHONY: opencv-validation-check opencv-onnx-fixture-check opencv-surface-check opencv-api-parity-check opencv-api-parity-candidates opencv-api-parity-reconcile opencv-api-parity-live-check opencv-api-parity-strict-check opencv-behavioral-parity-check opencv-behavioral-parity-refresh opencv-behavioral-parity-strict-check opencv-samples-check opencv-samples-runtime-check opencv-codec-policy-check opencv-metadata-check opencv-metadata-live-check opencv-aot-adapter-check opencv-source-check opencv-scripts-check \
	opencv-install-layout-check opencv-install-check \
	opencv-configure opencv-build opencv-generate opencv-native-adapter-check \
	opencv-native-sanitizer-check opencv-runtime-guard-check \
	opencv-promotion-tool-check opencv-promotion-inventory-check \
	opencv-fresh-workspace-check \
	opencv-package-artifact-tool-check opencv-package-artifact-build \
	opencv-package-artifact-check \
	opencv-runtime-unavailable-check terlan-opencv-package-preflight-check \
	terlan-opencv-package-check opencv-codec-check opencv-geometry-check \
	opencv-threshold-check opencv-morphology-check opencv-edge-check \
	opencv-normalization-check opencv-affine-check opencv-perspective-check \
	opencv-calib3d-check opencv-ml-check \
	opencv-dnn-check \
	opencv-histogram-check opencv-components-check opencv-contours-check \
	opencv-features-check opencv-arithmetic-check opencv-mask-check \
	opencv-range-check opencv-mask-logic-check opencv-mask-points-check \
	opencv-statistics-check opencv-filter-check terlan-grouped-binding-check terlan-function-reference-check clean

opencv-scripts-check:
	$(TERLAN_SCRIPT) check-sources -- "$(abspath $(TERLC))" "$(CURDIR)" scripts

terlan-grouped-binding-check:
	$(TERLAN_SCRIPT) check-sources -- "$(abspath $(TERLC))" "$(CURDIR)" TL0009

terlan-function-reference-check:
	$(TERLAN_SCRIPT) check-sources -- "$(abspath $(TERLC))" "$(CURDIR)" TL0010

opencv-validation-check:
	$(TERLAN_SCRIPT) check-cpp-validation -- "$(CXX)" "$(CURDIR)"

# `protoc --encode` writes binary stdout; keep this recipe until the Terlan
# process API can capture or redirect child output as bytes.
opencv-onnx-fixture-check:
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		for fixture in identity_rgb identity_grayscale ssd_reshape xyxy_reshape \
			objectness_reshape transposed_class_score_reshape transposed_objectness_reshape; do \
			protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
				tests/fixtures/minimal_onnx.proto \
				< "tests/fixtures/$$fixture.textproto" > "$$generated"; \
			cmp -s "tests/fixtures/$$fixture.onnx" "$$generated" || { \
				echo "$$fixture ONNX fixture is stale" >&2; exit 1; \
			}; \
		done

opencv-surface-check:
	cargo test --quiet --manifest-path "$(SURFACE_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(SURFACE_CHECK_MANIFEST)"

opencv-api-parity-check:
	cargo test --quiet --manifest-path "$(API_PARITY_MANIFEST)"
	cargo run --quiet --manifest-path "$(API_PARITY_MANIFEST)"

opencv-api-parity-candidates:
	cargo run --quiet --manifest-path "$(API_PARITY_MANIFEST)" -- --candidates

opencv-api-parity-reconcile:
	cargo run --quiet --manifest-path "$(API_PARITY_MANIFEST)" -- --refresh-reconciliation

opencv-api-parity-live-check: opencv-source-check
	@test -x "$(CPP_METADATA_EXTRACTOR)" || { echo "missing maintained C++ metadata extractor: $(CPP_METADATA_EXTRACTOR)" >&2; exit 1; }
	cargo run --quiet --manifest-path "$(API_PARITY_MANIFEST)" -- \
		--reproduce "$(abspath $(CPP_METADATA_EXTRACTOR))"

opencv-api-parity-strict-check: opencv-api-parity-check
	cargo run --quiet --manifest-path "$(API_PARITY_MANIFEST)" -- --strict

opencv-behavioral-parity-check:
	cargo test --quiet --manifest-path "$(BEHAVIORAL_PARITY_MANIFEST)"
	cargo run --quiet --manifest-path "$(BEHAVIORAL_PARITY_MANIFEST)"

opencv-behavioral-parity-refresh:
	cargo run --quiet --manifest-path "$(BEHAVIORAL_PARITY_MANIFEST)" -- --refresh

opencv-behavioral-parity-strict-check: opencv-behavioral-parity-check
	cargo run --quiet --manifest-path "$(BEHAVIORAL_PARITY_MANIFEST)" -- --strict

opencv-codec-policy-check:
	cargo test --quiet --manifest-path "$(CODEC_POLICY_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(CODEC_POLICY_CHECK_MANIFEST)" -- "$(CURDIR)"

opencv-metadata-check: opencv-generate
	cargo test --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)"

opencv-metadata-live-check: opencv-build opencv-metadata-check
	@test -x "$(CPP_METADATA_EXTRACTOR)" || { echo "missing maintained C++ metadata extractor: $(CPP_METADATA_EXTRACTOR)" >&2; exit 1; }
	cargo run --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)" -- "$(abspath $(CPP_METADATA_EXTRACTOR))"

opencv-source-check:
	@test -f "$(OPENCV_SOURCE)/CMakeLists.txt" || { echo "missing OpenCV checkout: $(OPENCV_SOURCE)" >&2; exit 1; }
	@test "$$(git -C "$(OPENCV_SOURCE)" rev-parse HEAD)" = "$(OPENCV_PINNED_COMMIT)" || { echo "OpenCV checkout is not at pinned commit $(OPENCV_PINNED_COMMIT)" >&2; exit 1; }

opencv-configure: opencv-surface-check opencv-behavioral-parity-check opencv-source-check
	cmake -S "$(OPENCV_SOURCE)" -B "$(OPENCV_BUILD)" -C cmake/opencv-cpu.cmake \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_FLAGS="$(OPENCV_REMAP_FLAGS)" \
		-DCMAKE_CXX_FLAGS="$(OPENCV_REMAP_FLAGS)" \
		-DCMAKE_INSTALL_PREFIX="$(abspath $(OPENCV_INSTALL))"

opencv-build: opencv-configure
	cmake --build "$(OPENCV_BUILD)" --parallel
	cmake --install "$(OPENCV_BUILD)"

opencv-install-layout-check:
	cargo test --quiet --manifest-path "$(NATIVE_INSTALL_CHECK_MANIFEST)"

opencv-install-check: opencv-build opencv-install-layout-check opencv-codec-policy-check
	cargo run --quiet --manifest-path "$(NATIVE_INSTALL_CHECK_MANIFEST)" -- \
		"$(abspath $(OPENCV_INSTALL))" "$(abspath tests/cpp/opencv_link_probe.cc)" "$(CXX)"

opencv-generate:
	$(TERLAN_SCRIPT) generate-package -- "$(abspath $(TERLC))" "$(CURDIR)"

opencv-aot-adapter-check: opencv-generate
	cargo test --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)" -- --generated-aot

opencv-samples-check: opencv-generate
	"$(TERLC)" build generated/samples --target terlan-vm

opencv-samples-runtime-check: opencv-native-adapter-check opencv-runtime-guard-check
	"$(TERLC)" build generated/samples --target terlan-vm
	$(TERLAN_SCRIPT) runtime-checks -- samples \
		"$(abspath ../terlan/target/debug/terlan-vm)" "$(CURDIR)" \
		"$(abspath $(OPENCV_INSTALL))" "$(abspath $(RUNTIME_GUARD))" "$(abspath $(NATIVE_HELPER))"

opencv-promotion-tool-check:
	cargo fmt --manifest-path "$(PROMOTION_CHECK_MANIFEST)" -- --check
	cargo clippy --quiet --manifest-path "$(PROMOTION_CHECK_MANIFEST)" --all-targets -- -D warnings
	cargo test --quiet --manifest-path "$(PROMOTION_CHECK_MANIFEST)"
	RUSTDOCFLAGS="-D warnings" cargo doc --quiet --no-deps --manifest-path "$(PROMOTION_CHECK_MANIFEST)"

opencv-promotion-inventory-check: opencv-promotion-tool-check
	cargo run --quiet --manifest-path "$(PROMOTION_CHECK_MANIFEST)" -- \
		--compiler "$(abspath ../terlan)" \
		--package "$(CURDIR)" \
		--opencv "$(abspath $(OPENCV_SOURCE))" \
		--expected-opencv-commit "$(OPENCV_PINNED_COMMIT)" \
		--report "$(abspath $(PROMOTION_REPORT))"

opencv-fresh-workspace-check: opencv-promotion-tool-check
	$(TERLAN_SCRIPT) fresh-workspace-check -- \
		"$(abspath ../terlan)" "$(CURDIR)" "$(abspath $(OPENCV_SOURCE))" \
		"$(OPENCV_PINNED_COMMIT)" "$(abspath $(PROMOTION_REPORT))"

opencv-package-artifact-tool-check:
	cargo fmt --manifest-path "$(PACKAGE_ARTIFACT_MANIFEST)" -- --check
	cargo clippy --locked --quiet --manifest-path "$(PACKAGE_ARTIFACT_MANIFEST)" --all-targets -- -D warnings
	cargo test --locked --quiet --manifest-path "$(PACKAGE_ARTIFACT_MANIFEST)"
	RUSTDOCFLAGS="--document-private-items -D missing_docs -D warnings" \
		cargo doc --locked --quiet --no-deps --manifest-path "$(PACKAGE_ARTIFACT_MANIFEST)"

opencv-package-artifact-build: opencv-install-check opencv-generate opencv-runtime-guard-check opencv-package-artifact-tool-check
	$(TERLAN_SCRIPT) workspace -- link-native "$(abspath $(OPENCV_INSTALL))" "$(abspath $(GENERATED_NATIVE)/opencv)"
	$(TERLAN_SCRIPT) package-artifact -- build "$(abspath $(TERLC))" "$(CURDIR)" \
		"$(abspath $(PACKAGE_ARTIFACT_DIR))" "$(abspath $(PACKAGE_ARTIFACT_REPLAY_DIR))" \
		"$(PACKAGE_ARTIFACT_TARGET)" "$(RUST_REMAP_FLAGS)" "$(BRIDGE_REMAP_FLAGS)"

opencv-package-artifact-check: opencv-package-artifact-build
	$(TERLAN_SCRIPT) package-artifact -- check "$(abspath $(TERLC))" "$(CURDIR)" \
		"$(abspath $(PACKAGE_ARTIFACT_DIR))" "$(abspath $(PACKAGE_ARTIFACT_REPLAY_DIR))" \
		"$(PACKAGE_ARTIFACT_TARGET)" "$(RUST_REMAP_FLAGS)" "$(BRIDGE_REMAP_FLAGS)"

opencv-native-adapter-check: opencv-validation-check opencv-onnx-fixture-check opencv-install-check opencv-metadata-live-check opencv-aot-adapter-check
	@test -f "$(OPENCV_INSTALL)/include/opencv4/opencv2/core.hpp" || { echo "OpenCV install is incomplete" >&2; exit 1; }
	$(TERLAN_SCRIPT) workspace -- link-native "$(abspath $(OPENCV_INSTALL))" "$(abspath $(GENERATED_NATIVE)/opencv)"
	cargo test --quiet --manifest-path "$(NATIVE_CHECK_MANIFEST)"
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		cargo test --quiet --manifest-path "$(NATIVE_DIRECT_MANIFEST)" -- --test-threads=1
	cargo build --offline --quiet --manifest-path "$(GENERATED_NATIVE)/Cargo.toml" --bin native-boundary-helper
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		cargo run --quiet --manifest-path "$(NATIVE_CHECK_MANIFEST)" -- "$(abspath $(NATIVE_HELPER))"

opencv-native-sanitizer-check: opencv-install-check opencv-metadata-live-check opencv-generate
	@test -f "$(ASAN_STATIC_RUNTIME)" || { echo "missing static AddressSanitizer runtime: $(ASAN_STATIC_RUNTIME)" >&2; exit 1; }
	@test -f "$(UBSAN_STATIC_RUNTIME)" || { echo "missing static UndefinedBehaviorSanitizer runtime: $(UBSAN_STATIC_RUNTIME)" >&2; exit 1; }
	$(TERLAN_SCRIPT) workspace -- link-native "$(abspath $(OPENCV_INSTALL))" "$(abspath $(GENERATED_NATIVE)/opencv)"
	ASAN_OPTIONS="$(SANITIZER_OPTIONS)" \
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
	CXXFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
	RUSTFLAGS="-Clink-arg=$(ASAN_STATIC_RUNTIME) -Clink-arg=$(UBSAN_STATIC_RUNTIME)" \
		cargo test --quiet --manifest-path "$(NATIVE_DIRECT_MANIFEST)" --tests -- --test-threads=1
	CXXFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
	RUSTFLAGS="-Clink-arg=$(ASAN_STATIC_RUNTIME) -Clink-arg=$(UBSAN_STATIC_RUNTIME)" \
		cargo build --offline --quiet --manifest-path "$(GENERATED_NATIVE)/Cargo.toml" --bin native-boundary-helper
	ASAN_OPTIONS="$(SANITIZER_OPTIONS)" \
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		cargo run --quiet --manifest-path "$(NATIVE_CHECK_MANIFEST)" -- "$(abspath $(NATIVE_HELPER))"

terlan-opencv-package-preflight-check: opencv-generate
	$(TERLAN_SCRIPT) workspace -- link-native "$(abspath $(OPENCV_INSTALL))" "$(abspath $(GENERATED_NATIVE)/opencv)"
	$(TERLAN_SCRIPT) runtime-checks -- preflight \
		"$(abspath $(TERLC))" "$(CURDIR)" "$(abspath $(OPENCV_INSTALL))" \
		"$(abspath $(RUNTIME_GUARD))" "$(abspath $(NATIVE_HELPER))"

opencv-runtime-guard-check:
	cargo test --quiet --manifest-path "$(RUNTIME_GUARD_MANIFEST)"
	cargo build --quiet --manifest-path "$(RUNTIME_GUARD_MANIFEST)"

opencv-runtime-unavailable-check: terlan-opencv-package-preflight-check opencv-runtime-guard-check
	$(TERLAN_SCRIPT) runtime-checks -- unavailable \
		"$(abspath $(TERLC))" "$(CURDIR)" "$(abspath $(OPENCV_INSTALL))" \
		"$(abspath $(RUNTIME_GUARD))" "$(abspath upstream/install/missing-native-worker)"

terlan-opencv-package-check: opencv-native-adapter-check opencv-runtime-unavailable-check opencv-scripts-check terlan-grouped-binding-check terlan-function-reference-check
	$(TERLAN_SCRIPT) runtime-checks -- package \
		"$(abspath $(TERLC))" "$(CURDIR)" "$(abspath $(OPENCV_INSTALL))" \
		"$(abspath $(RUNTIME_GUARD))" "$(abspath $(NATIVE_HELPER))"

opencv-codec-check: terlan-opencv-package-check

opencv-geometry-check: terlan-opencv-package-check

opencv-threshold-check: terlan-opencv-package-check

opencv-morphology-check: terlan-opencv-package-check

opencv-edge-check: terlan-opencv-package-check

opencv-normalization-check: terlan-opencv-package-check

opencv-affine-check: terlan-opencv-package-check

opencv-perspective-check: terlan-opencv-package-check

opencv-calib3d-check: terlan-opencv-package-check

opencv-ml-check: terlan-opencv-package-check

opencv-dnn-check: opencv-onnx-fixture-check terlan-opencv-package-check

opencv-histogram-check: terlan-opencv-package-check

opencv-components-check: terlan-opencv-package-check

opencv-contours-check: terlan-opencv-package-check

opencv-features-check: terlan-opencv-package-check

opencv-arithmetic-check: terlan-opencv-package-check

opencv-mask-check: terlan-opencv-package-check

opencv-range-check: terlan-opencv-package-check

opencv-mask-logic-check: terlan-opencv-package-check

opencv-mask-points-check: terlan-opencv-package-check

opencv-statistics-check: terlan-opencv-package-check

opencv-filter-check: terlan-opencv-package-check

clean:
	$(TERLAN_SCRIPT) workspace -- clean "$(CURDIR)"
