# Two-Dimensional Features Tutorial Inventory

This inventory maps the pinned OpenCV feature tutorials to bounded, copied
Terlan package results. GUI rendering and ambient image files are replaced by
deterministic in-memory matrices.

| Tutorial family | Terlan coverage | Status |
| --- | --- | --- |
| ORB detection and description | `OrbFeatures.terl` | Runnable |
| AKAZE detection and matching | `AkazeMatching.terl` | Runnable |
| SIFT detection and matching | `SiftMatching.terl` | Runnable |
| BRISK detection and matching | `BriskMatching.terl` | Runnable |
| KAZE detection and matching | `KazeMatching.terl` | Runnable |
| MSER regions | `MserRegions.terl` | Runnable |
| Simple blobs | `SimpleBlobDetection.terl` | Runnable |
| SURF tutorials | SURF is not part of the reviewed package surface | Blocked |
| Match-derived robust homography | No reviewed RANSAC match-to-homography contract | Blocked |
| Planar tracking and panorama stitching | Tracking, stitching, and GUI capabilities are not reviewed | Blocked |
| Homography decomposition and camera pose | No reviewed decomposition result type | Blocked |

Detector-specific tutorials must validate copied keypoints, descriptor shape,
bounded matches, and owner cleanup rather than merely invoking an algorithm.
