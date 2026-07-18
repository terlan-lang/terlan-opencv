# Core Tutorial Inventory

This inventory maps every C++ tutorial under OpenCV's `tutorial_code/core`
tree to the reviewed Terlan package surface. GUI, timing, pointer access, and
filesystem mechanics are adapted only when the underlying operation remains
observable through owned values.

| OpenCV tutorial | Terlan coverage | Status |
| --- | --- | --- |
| The basic image container | `MatBasics.terl` | Runnable |
| How to scan images | `ImageScanning.terl` using a copied LUT | Runnable |
| Adding two images | `ImageBlending.terl` | Runnable |
| Mask operations on matrices | `MaskFiltering.terl` | Runnable |
| Matrix operations | `MatBasics.terl` and `MatrixOperations.terl` | Runnable |
| File input and output | `../imgcodecs/EncodeDecode.terl` with copied bytes | Adapted |
| Discrete Fourier transform | No reviewed DFT package operation | Blocked |
| `parallel_for_` and parallel backends | Compiler/runtime scheduling concern, not a public OpenCV operation | Not applicable |
| Universal intrinsics | Native adapter implementation detail | Not applicable |

The inventory is complete for the pinned upstream core tutorial tree. New
upstream core tutorials must be classified here before a sample is added.
