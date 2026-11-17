# Accuracy and speed

## Agreement with the original

Every backend is checked against the original FlexGlue (release v0.1, PyTorch) on 6 pairs: FlexGlue's and
ALIKED's sample images, with 2048 RaCo keypoints per image, at 512 px with 9 layers and at 384 px with 5 layers.
In FP32:

| Backend | Matches | Predicted points | Confidences |
|---|---|---|---|
| CPU | identical on all 12 runs | within 4.8e-4 px | within 1.6e-5 |
| Metal | identical on all 12 runs | within 1.9e-4 px | within 5.6e-6 |
| CUDA | identical on all 12 runs | within 4.1e-4 px | within 1.5e-5 |
| WebGPU (native) | identical on all 12 runs | within 2.4e-4 px | within 1.5e-5 |
| WebGPU (Chrome) | the native CPU backend's, on all 12 runs | within 3.1e-4 px | |

The covisibility score agrees to 4 digits. Every intermediate tensor of the network is also compared with a
reference forward pass, written from the weights file alone, that matches the original.

With early stop (`min_covisibility` 0.8, decay 0, 0.2 and 0.5, which stop the pairs after layer 2, 4 or 8, or not
at all), every backend in both precisions stops each pair at the original's layer.

### FP16

FP16 stores feature maps and weights as 16-bit floats and does its arithmetic on the GPU's matrix hardware.
Against the original, on the same 12 runs:

| Backend | Runs with identical matches | Otherwise | Confident points move by (median) |
|---|---|---|---|
| Metal | 11 of 12 | 38 of 39 matches the same | 0.005 to 0.034 px |
| CUDA | 9 of 12 | 1 match differs | |
| WebGPU (native) | 11 of 12 | 65 of 66 the same | 0.005 to 0.034 px |
| WebGPU (Chrome) | 7 of 12 | 1 or 2 matches differ; 5 of 70 on the hardest pair | |

On Metal and native WebGPU the covisibility score moves by at most 3.2e-3.

### Settings and accuracy

The original network on 8 synthetic pairs (4 photos, each warped by a mild and a strong homography with a change
of brightness, so the truth is exact), 2048 RaCo keypoints per image. Errors in pixels of the original images,
about 1000 px wide, over every keypoint whose true position is inside the other image:

| Size | Layers | Median error | Within 1 px | Within 3 px | Matches | Correct (within 3 px) |
|---|---|---|---|---|---|---|
| 640 | 9 | 0.18 px | 99.2 % | 99.9 % | 1488 | 99.7 % |
| 512 | 9 | 0.20 px | 98.9 % | 99.8 % | 1475 | 99.5 % |
| 512 | 7 | 0.22 px | 98.4 % | 99.8 % | 1501 | 99.5 % |
| 512 | 5 | 0.33 px | 95.3 % | 99.8 % | 1568 | 99.5 % |
| 512 | 3 | 0.72 px | 68.8 % | 98.6 % | 1558 | 99.2 % |
| 384 | 9 | 0.40 px | 93.3 % | 99.5 % | 1490 | 99.2 % |
| 256 | 9 | 2.06 px | 15.5 % | 76.1 % | 1446 | 95.1 % |

These pairs are easier than real ones: read the table as trends. At 256 px a real pair with a large change of
scale drops from 70 matches to 10, which is why 384 px is the smallest size to use.

## Speed

Milliseconds per pair, from two described images and their keypoints to matches in host memory, 2048 keypoints
per image; best of 8 runs. "Describe" is per image, from 8-bit pixels in host memory.

On an M5 Pro (macOS 26), against the original under PyTorch (MPS) on the same GPU:

| Size, layers | Metal FP32 | PyTorch FP32 | Metal FP16 | PyTorch FP16 | WebGPU FP32 | WebGPU FP16 | CPU |
|---|---|---|---|---|---|---|---|
| 512, 9 | 314 | 376 | 105 | 157 | 478 | 313 | 7200 |
| 512, 5 | 180 | 215 | 62 | 92 | 271 | 178 | 4300 |
| 384, 9 | 140 | 167 | 48 | 82 | 209 | 139 | 2500 |
| 384, 5 | 82 | | 29 | | 121 | 81 | 1400 |
| Describe, 512 | 31 | 39 | 13 | 39 | 46 | 30 | 410 |
| Describe, 384 | 19 | 23 | 8 | 23 | 28 | 18 | 270 |

On an RTX 2070 Max-Q laptop (CUDA 13.2, PyTorch 2.15):

| Size, layers | CUDA FP32 | PyTorch FP32 | CUDA FP16 | PyTorch FP16 |
|---|---|---|---|---|
| 512, 9 | 417 | 629 | 141 | 207 |
| 512, 5 | 245 | 360 | 83 | 121 |
| 384, 9 | 190 | 249 | 54 | 97 |
| Describe, 512 | 54 | 54 | 18 | 54 |
| Describe, 384 | 40 | 31 | 14 | 31 |

WebGPU is native Dawn; in Chrome on the same M5 Pro a pair takes 500 ms in FP32 and 335 ms in FP16 at 512 px and
9 layers, about 140 and 95 ms at 384 px and 5 layers. WebGPU has no matrix instructions yet, which is most of its gap
to Metal. The CPU backend runs on all cores of the M5 Pro.

With early stop, a pair stopped after layer 2 costs about a quarter of a full one: 30 ms on Metal in FP16 and
77 ms on WebGPU in FP16, against 102 and 308 ms.

For comparison, FlexGlue's authors report 29 ms per pair on an RTX 4090.
