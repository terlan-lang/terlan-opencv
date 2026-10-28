# OpenCV runtime guard

The runtime guard is the package-owned executable exposed to Terlan VM native
dispatch. It has no OpenCV linkage. It starts the generated native worker from
`TERLAN_OPENCV_NATIVE_WORKER_PATH`, or from `terlan-opencv-worker` beside the
guard in an installed artifact, and proxies correlated requests and replies.

If the generated worker is absent, cannot start, or exits before replying, the
guard returns the stable `opencv.unavailable` package error. Dynamic-loader
failures therefore remain contained outside the VM and cannot become raw
linker crashes or compiler-specific OpenCV policy.
