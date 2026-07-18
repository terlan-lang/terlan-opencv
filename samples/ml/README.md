# Machine Learning Tutorial Inventory

This inventory maps the pinned OpenCV classical machine-learning tutorials to
the package's copied row-major training and prediction contracts.

| Tutorial family | Terlan coverage | Status |
| --- | --- | --- |
| K-nearest-neighbor classification | `KnnClassification.terl` | Runnable |
| Linear support-vector classification | `LinearSvmClassification.terl` | Runnable |
| Nonlinear support-vector machines | Package intentionally fixes a linear kernel | Blocked |
| Principal component analysis | No reviewed PCA model or projection contract | Blocked |
| Other bounded classifiers and clustering | Reviewed package APIs and package tests; dedicated tutorials pending | Supported |
| Model persistence and arbitrary training configuration | No reviewed serialization or open-ended parameter contract | Blocked |

Additional classifier tutorials must use deterministic copied datasets and
release every model owner explicitly.
