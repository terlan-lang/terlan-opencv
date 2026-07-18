# Image Processing Tutorial Inventory

This inventory groups the OpenCV 4.12 image-processing tutorials by distinct
operation. Runnable ports preserve useful algorithm-step comments while using
owned in-memory inputs instead of windows, trackbars, cameras, or files.

| Tutorial family | Terlan sample | Status |
| --- | --- | --- |
| Smoothing and filtering | `Smoothing.terl`, `../core/MaskFiltering.terl` | Runnable |
| Erosion, dilation, and compound morphology | `Morphology.terl` | Runnable |
| Fixed, adaptive, Otsu, and triangle thresholding | `Thresholding.terl` | Runnable |
| Basic drawing | `Drawing.terl` | Runnable |
| Geometric affine and perspective transforms | `GeometricTransforms.terl` | Runnable |
| Histograms and equalization | `Histograms.terl` | Runnable |
| Canny, Sobel, and Scharr edges | `EdgeDetection.terl` | Runnable |
| External contours and shape geometry | `Contours.terl` | Runnable |
| Shi-Tomasi and Harris corners | `CornerDetection.terl` | Runnable |
| Gaussian image pyramids | `Pyramids.terl` | Runnable |
| Constant, replicated, and reflected borders | `Borders.terl` | Runnable |
| Pixel range masks | `RangeMask.terl` | Runnable |
| Bounded Hough detection | `HoughDetection.terl` | Runnable |
| Flood fill and connected components | `FloodFillComponents.terl` | Runnable |
| Histogram template matching | `TemplateMatching.terl` | Runnable |
| Arbitrary remap, backprojection, and histogram comparison | No reviewed package operation | Blocked |
| Hit-or-miss and arbitrary structuring elements | Current morphology contract uses bounded rectangular kernels | Blocked |
| Frequency-domain deblurring and periodic-noise removal | No reviewed DFT/deconvolution operation | Blocked |
| GUI display and trackbars | Outside the current package capability contract | Blocked |

Unsupported rows require a package capability review before tutorial code is
added.
