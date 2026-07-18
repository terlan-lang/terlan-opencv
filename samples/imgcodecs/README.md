# Image Codecs Tutorial Inventory

This inventory maps the pinned OpenCV image-codec examples to the reviewed
Terlan package surface. Runtime samples operate only on copied in-memory bytes;
paths, URLs, and filesystem discovery remain explicit host capabilities.

| Tutorial family | Terlan coverage | Status |
| --- | --- | --- |
| In-memory image encoding and decoding | `EncodeDecode.terl` | Runnable |
| Image file reading and writing | Ambient filesystem access is outside the package contract | Blocked |
| Multi-page image containers | No reviewed bounded page-list contract | Blocked |
| Image metadata and orientation | No reviewed copied metadata type or orientation policy | Blocked |
| Animated image codecs | No reviewed frame timing, count, or allocation contract | Blocked |

New codec tutorials must use copied bytes, an explicitly reviewed format, hard
allocation limits, exact decoded layout checks, and deterministic owner
cleanup.

