# Camera Calibration And 3D Tutorial Inventory

This inventory maps the pinned OpenCV calibration tutorials to the reviewed
Terlan package surface. Samples use copied points and deterministic in-memory
images instead of cameras, GUI loops, and calibration files.

| Tutorial family | Terlan coverage | Status |
| --- | --- | --- |
| Camera projection | `ProjectPoints.terl` | Runnable |
| Camera-matrix image correction | `UndistortImage.terl` | Runnable |
| Perspective-n-point pose recovery | `PoseEstimation.terl` | Runnable |
| Multi-view intrinsic calibration | No reviewed `calibrateCamera` contract | Blocked |
| Stereo calibration, rectification, and depth | No reviewed stereo owner model | Blocked |
| Real-time pose registration and tracking | Camera, filesystem, and tracking capabilities are not reviewed | Blocked |
| Homography decomposition | No reviewed decomposition result type | Blocked |

Unsupported rows require explicit copied-result, ownership, and capability
contracts before tutorial code is added.
