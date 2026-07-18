OPENCV_SOURCE ?= upstream/opencv
OPENCV_BUILD ?= upstream/build
OPENCV_INSTALL ?= upstream/install
OPENCV_PINNED_COMMIT := fe38fc608f6acb8b68953438a62305d8318f4fcd
CXX ?= c++
ASAN_STATIC_RUNTIME ?= $(shell $(CXX) -print-file-name=libasan.a)
UBSAN_STATIC_RUNTIME ?= $(shell $(CXX) -print-file-name=libubsan.a)
TERLC ?= ../terlan/target/debug/terlc
SURFACE_CHECK_MANIFEST := tools/surface_check/Cargo.toml
NATIVE_CHECK_MANIFEST := tools/native_adapter_check/Cargo.toml
NATIVE_DIRECT_MANIFEST := tools/native_direct_check/Cargo.toml
NATIVE_INSTALL_CHECK_MANIFEST := tools/native_install_check/Cargo.toml
CODEC_POLICY_CHECK_MANIFEST := tools/codec_policy_check/Cargo.toml
METADATA_CHECK_MANIFEST := tools/metadata_check/Cargo.toml
PROMOTION_CHECK_MANIFEST := tools/promotion_check/Cargo.toml
RUNTIME_GUARD_MANIFEST := native/runtime_guard/Cargo.toml
CPP_METADATA_EXTRACTOR ?= ../terlan/target/cpp-metadata-extractor/terlan-cpp-metadata-extractor
NATIVE_MANIFEST := bindings/opencv.native-binding.json
GENERATED_NATIVE := generated/native/rust
NATIVE_HELPER := $(GENERATED_NATIVE)/target/debug/native-boundary-helper
RUNTIME_GUARD := native/runtime_guard/target/debug/terlan-opencv-runtime-guard
PROMOTION_REPORT ?= _build/promotion/opencv-promotion.json

.PHONY: opencv-validation-check opencv-onnx-fixture-check opencv-surface-check opencv-samples-check opencv-samples-runtime-check opencv-codec-policy-check opencv-metadata-check opencv-metadata-live-check opencv-source-check \
	opencv-install-layout-check opencv-install-check \
	opencv-configure opencv-build opencv-generate opencv-native-adapter-check \
	opencv-native-sanitizer-check opencv-runtime-guard-check \
	opencv-promotion-tool-check opencv-promotion-inventory-check \
	opencv-fresh-workspace-check \
	opencv-runtime-unavailable-check terlan-opencv-package-preflight-check \
	terlan-opencv-package-check opencv-codec-check opencv-geometry-check \
	opencv-threshold-check opencv-morphology-check opencv-edge-check \
	opencv-normalization-check opencv-affine-check opencv-perspective-check \
	opencv-calib3d-check opencv-ml-check \
	opencv-dnn-check \
	opencv-histogram-check opencv-components-check opencv-contours-check \
	opencv-features-check opencv-arithmetic-check opencv-mask-check \
	opencv-range-check opencv-mask-logic-check opencv-mask-points-check \
	opencv-statistics-check opencv-filter-check clean

opencv-validation-check:
	@binary=$$(mktemp); trap 'rm -f "$$binary"' EXIT; \
		"$(CXX)" -std=c++14 -Wall -Wextra -Wpedantic -Werror -pthread \
		-Ibindings/cpp tests/cpp/opencv_validation_test.cc -o "$$binary"; \
		"$$binary"
	@binary=$$(mktemp); trap 'rm -f "$$binary"' EXIT; \
		"$(CXX)" -std=c++14 -Wall -Wextra -Wpedantic -Werror \
		-Ibindings/cpp tests/cpp/opencv_dnn_validation_test.cc -o "$$binary"; \
		"$$binary"
	@binary=$$(mktemp); trap 'rm -f "$$binary"' EXIT; \
		"$(CXX)" -std=c++14 -Wall -Wextra -Wpedantic -Werror \
		-Ibindings/cpp tests/cpp/opencv_dnn_detection_validation_test.cc -o "$$binary"; \
		"$$binary"

opencv-onnx-fixture-check:
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/identity_rgb.textproto > "$$generated"; \
		cmp -s tests/fixtures/identity_rgb.onnx "$$generated" || { \
			echo "RGB Identity ONNX fixture is stale" >&2; exit 1; \
		}
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/identity_grayscale.textproto > "$$generated"; \
		cmp -s tests/fixtures/identity_grayscale.onnx "$$generated" || { \
			echo "grayscale Identity ONNX fixture is stale" >&2; exit 1; \
		}
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/ssd_reshape.textproto > "$$generated"; \
		cmp -s tests/fixtures/ssd_reshape.onnx "$$generated" || { \
			echo "SSD ONNX fixture is stale" >&2; exit 1; \
		}
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/xyxy_reshape.textproto > "$$generated"; \
		cmp -s tests/fixtures/xyxy_reshape.onnx "$$generated" || { \
			echo "XYXY ONNX fixture is stale" >&2; exit 1; \
		}
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/objectness_reshape.textproto > "$$generated"; \
		cmp -s tests/fixtures/objectness_reshape.onnx "$$generated" || { \
			echo "objectness ONNX fixture is stale" >&2; exit 1; \
		}
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/transposed_class_score_reshape.textproto > "$$generated"; \
		cmp -s tests/fixtures/transposed_class_score_reshape.onnx "$$generated" || { \
			echo "transposed class-score ONNX fixture is stale" >&2; exit 1; \
		}
	@generated=$$(mktemp); trap 'rm -f "$$generated"' EXIT; \
		protoc --proto_path=tests/fixtures --encode=onnx.ModelProto \
			tests/fixtures/minimal_onnx.proto \
			< tests/fixtures/transposed_objectness_reshape.textproto > "$$generated"; \
		cmp -s tests/fixtures/transposed_objectness_reshape.onnx "$$generated" || { \
			echo "transposed objectness ONNX fixture is stale" >&2; exit 1; \
		}

opencv-surface-check:
	cargo test --quiet --manifest-path "$(SURFACE_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(SURFACE_CHECK_MANIFEST)"

opencv-codec-policy-check:
	cargo test --quiet --manifest-path "$(CODEC_POLICY_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(CODEC_POLICY_CHECK_MANIFEST)" -- "$(CURDIR)"

opencv-metadata-check:
	cargo test --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)"
	cargo run --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)"

opencv-metadata-live-check: opencv-build opencv-metadata-check
	@test -x "$(CPP_METADATA_EXTRACTOR)" || { echo "missing maintained C++ metadata extractor: $(CPP_METADATA_EXTRACTOR)" >&2; exit 1; }
	cargo run --quiet --manifest-path "$(METADATA_CHECK_MANIFEST)" -- "$(abspath $(CPP_METADATA_EXTRACTOR))"

opencv-source-check:
	@test -f "$(OPENCV_SOURCE)/CMakeLists.txt" || { echo "missing OpenCV checkout: $(OPENCV_SOURCE)" >&2; exit 1; }
	@test "$$(git -C "$(OPENCV_SOURCE)" rev-parse HEAD)" = "$(OPENCV_PINNED_COMMIT)" || { echo "OpenCV checkout is not at pinned commit $(OPENCV_PINNED_COMMIT)" >&2; exit 1; }

opencv-configure: opencv-surface-check opencv-source-check
	cmake -S "$(OPENCV_SOURCE)" -B "$(OPENCV_BUILD)" -C cmake/opencv-cpu.cmake \
		-DCMAKE_BUILD_TYPE=Release \
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
	@test -x "$(TERLC)" || { echo "missing Terlan compiler: $(TERLC)" >&2; exit 1; }
	rm -rf generated
	"$(TERLC)" bind cpp --manifest "$(NATIVE_MANIFEST)" --out generated
	cp tests/opencv/DnnDetectionChecks.terl generated/src/opencv/DnnDetectionChecks.terl
	mkdir -p generated/tests/opencv
	cp tests/opencv/*Test.terl generated/tests/opencv/
	mkdir -p generated/samples/src/samples
	cp samples/terlan.toml.template generated/samples/terlan.toml
	find samples -mindepth 1 -maxdepth 1 -type d -exec cp -R {} generated/samples/src/samples/ \;

opencv-samples-check: opencv-generate
	"$(TERLC)" build generated/samples --target terlan-vm

opencv-samples-runtime-check: opencv-native-adapter-check opencv-runtime-guard-check
	"$(TERLC)" build generated/samples --target terlan-vm
	@set -eu; \
	while IFS="$$(printf '\t')" read -r artifact expected; do \
		output=$$(LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
			TERLAN_NATIVE_BOUNDARY_HELPER_PATH="$(abspath $(RUNTIME_GUARD))" \
			TERLAN_OPENCV_NATIVE_WORKER_PATH="$(abspath $(NATIVE_HELPER))" \
			"$(abspath ../terlan/target/debug/terlan-vm)" run "$(abspath _build/vm)/$$artifact"); \
		test "$$output" = "$$expected" || { \
			echo "sample $$artifact produced '$$output'; expected '$$expected'" >&2; \
			exit 1; \
		}; \
		echo "sample $$artifact: $$output"; \
	done < samples/expected-output.tsv

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
	tools/run_fresh_workspace_check.sh \
		"$(abspath ../terlan)" \
		"$(CURDIR)" \
		"$(abspath $(OPENCV_SOURCE))" \
		"$(OPENCV_PINNED_COMMIT)" \
		"$(abspath $(PROMOTION_REPORT))"

opencv-native-adapter-check: opencv-validation-check opencv-onnx-fixture-check opencv-install-check opencv-metadata-live-check opencv-generate
	@test -f "$(OPENCV_INSTALL)/include/opencv4/opencv2/core.hpp" || { echo "OpenCV install is incomplete" >&2; exit 1; }
	ln -s "$(abspath $(OPENCV_INSTALL))" "$(GENERATED_NATIVE)/opencv"
	cargo test --quiet --manifest-path "$(NATIVE_CHECK_MANIFEST)"
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		cargo test --quiet --manifest-path "$(NATIVE_DIRECT_MANIFEST)" -- --test-threads=1
	cargo build --quiet --manifest-path "$(GENERATED_NATIVE)/Cargo.toml" --bin native-boundary-helper
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		cargo run --quiet --manifest-path "$(NATIVE_CHECK_MANIFEST)" -- "$(abspath $(NATIVE_HELPER))"

opencv-native-sanitizer-check: opencv-install-check opencv-metadata-live-check opencv-generate
	@test -f "$(ASAN_STATIC_RUNTIME)" || { echo "missing static AddressSanitizer runtime: $(ASAN_STATIC_RUNTIME)" >&2; exit 1; }
	@test -f "$(UBSAN_STATIC_RUNTIME)" || { echo "missing static UndefinedBehaviorSanitizer runtime: $(UBSAN_STATIC_RUNTIME)" >&2; exit 1; }
	ln -s "$(abspath $(OPENCV_INSTALL))" "$(GENERATED_NATIVE)/opencv"
	ASAN_OPTIONS=detect_leaks=1 \
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
	CXXFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
	RUSTFLAGS="-Clink-arg=$(ASAN_STATIC_RUNTIME) -Clink-arg=$(UBSAN_STATIC_RUNTIME)" \
		cargo test --quiet --manifest-path "$(NATIVE_DIRECT_MANIFEST)" --tests -- --test-threads=1
	CXXFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
	RUSTFLAGS="-Clink-arg=$(ASAN_STATIC_RUNTIME) -Clink-arg=$(UBSAN_STATIC_RUNTIME)" \
		cargo build --quiet --manifest-path "$(GENERATED_NATIVE)/Cargo.toml" --bin native-boundary-helper
	ASAN_OPTIONS=detect_leaks=1 \
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
		cargo run --quiet --manifest-path "$(NATIVE_CHECK_MANIFEST)" -- "$(abspath $(NATIVE_HELPER))"

terlan-opencv-package-preflight-check: opencv-generate
	@output=$$(mktemp); \
	if env -u TERLAN_NATIVE_BOUNDARY_HELPER_PATH -u TERLAN_SAFE_NATIVE_PATH "$(TERLC)" test generated/tests/opencv/ImgprocTest.terl --target terlan-vm >"$$output" 2>&1; then \
		cat "$$output"; rm -f "$$output"; echo "package preflight unexpectedly executed without a native helper" >&2; exit 1; \
	fi; \
	if ! grep -Fq 'error[native_helper_unavailable]: TERLAN_NATIVE_BOUNDARY_HELPER_PATH is not set for compiler-native package operation' "$$output"; then \
		cat "$$output"; rm -f "$$output"; echo "package-local test did not reach the native helper boundary" >&2; exit 1; \
	fi; \
	cat "$$output"; rm -f "$$output"
	@output=$$(mktemp); \
	if env -u TERLAN_NATIVE_BOUNDARY_HELPER_PATH -u TERLAN_SAFE_NATIVE_PATH "$(TERLC)" test tests/consumer/tests/opencv_consumer/OpenCvConsumerTest.terl --target terlan-vm >"$$output" 2>&1; then \
		cat "$$output"; rm -f "$$output"; echo "consumer preflight unexpectedly executed without a native helper" >&2; exit 1; \
	fi; \
	if ! grep -Fq 'error[native_helper_unavailable]: TERLAN_NATIVE_BOUNDARY_HELPER_PATH is not set for compiler-native package operation' "$$output"; then \
		cat "$$output"; rm -f "$$output"; echo "external consumer did not reach the native helper boundary" >&2; exit 1; \
	fi; \
	cat "$$output"; rm -f "$$output"

opencv-runtime-guard-check:
	cargo test --quiet --manifest-path "$(RUNTIME_GUARD_MANIFEST)"
	cargo build --quiet --manifest-path "$(RUNTIME_GUARD_MANIFEST)"

opencv-runtime-unavailable-check: terlan-opencv-package-preflight-check opencv-runtime-guard-check
	@output=$$(mktemp); \
	if TERLAN_NATIVE_BOUNDARY_HELPER_PATH="$(abspath $(RUNTIME_GUARD))" TERLAN_OPENCV_NATIVE_WORKER_PATH="$(abspath upstream/install/missing-native-worker)" "$(TERLC)" test generated/tests/opencv/ImgprocTest.terl --target terlan-vm >"$$output" 2>&1; then \
		cat "$$output"; rm -f "$$output"; echo "package unavailable check unexpectedly passed" >&2; exit 1; \
	fi; \
	if ! grep -Fq 'error[opencv.unavailable]: OpenCV native runtime is unavailable' "$$output"; then \
		cat "$$output"; rm -f "$$output"; echo "package-local test did not receive opencv.unavailable" >&2; exit 1; \
	fi; \
	cat "$$output"; rm -f "$$output"
	@output=$$(mktemp); \
	if TERLAN_NATIVE_BOUNDARY_HELPER_PATH="$(abspath $(RUNTIME_GUARD))" TERLAN_OPENCV_NATIVE_WORKER_PATH="$(abspath upstream/install/missing-native-worker)" "$(TERLC)" test tests/consumer/tests/opencv_consumer/OpenCvConsumerTest.terl --target terlan-vm >"$$output" 2>&1; then \
		cat "$$output"; rm -f "$$output"; echo "consumer unavailable check unexpectedly passed" >&2; exit 1; \
	fi; \
	if ! grep -Fq 'error[opencv.unavailable]: OpenCV native runtime is unavailable' "$$output"; then \
		cat "$$output"; rm -f "$$output"; echo "external consumer did not receive opencv.unavailable" >&2; exit 1; \
	fi; \
	cat "$$output"; rm -f "$$output"

terlan-opencv-package-check: opencv-native-adapter-check opencv-runtime-unavailable-check
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
	TERLAN_NATIVE_BOUNDARY_HELPER_PATH="$(abspath $(RUNTIME_GUARD))" \
	TERLAN_OPENCV_NATIVE_WORKER_PATH="$(abspath $(NATIVE_HELPER))" \
		"$(TERLC)" test generated/tests --target terlan-vm
	LD_LIBRARY_PATH="$(abspath $(OPENCV_INSTALL))/lib$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" \
	TERLAN_NATIVE_BOUNDARY_HELPER_PATH="$(abspath $(RUNTIME_GUARD))" \
	TERLAN_OPENCV_NATIVE_WORKER_PATH="$(abspath $(NATIVE_HELPER))" \
		"$(TERLC)" test tests/consumer/tests --target terlan-vm

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
	rm -rf generated upstream/build upstream/install
