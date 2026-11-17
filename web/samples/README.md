# Demo sample images

Twelve images for `demo.html`, in capture order within each scene: four TUM frames, four KITTI frames, two
views of one place in Berlin, and two LaMAR images of different places that look alike (FlexGlue's hard
negative: its covisibility for them is near zero).

| Files | Source | Licence |
|---|---|---|
| `tum-1` to `tum-4` | TUM RGB-D benchmark, frames 0, 4, 8 and 12 of the sample sequence in the ALIKED repository (0.4 s apart). Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard and Daniel Cremers, "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012. <https://cvg.cit.tum.de/data/datasets/rgbd-dataset> | CC BY 4.0 |
| `kitti-1` to `kitti-4` | KITTI odometry benchmark, frames 100, 103, 106 and 109 of the sample sequence in the ALIKED repository. Andreas Geiger, Philip Lenz and Raquel Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012. <https://www.cvlibs.net/datasets/kitti/> | CC BY-NC-SA 3.0 (non-commercial) |
| `berlin-1`, `berlin-2` | `assets/berlin1.jpg`, `berlin2.jpg` of the FlexGlue repository | Not stated (the repository's code is MIT) |
| `lamar-1`, `lamar-2` | `assets/lamar_cab_1446.jpg`, `lamar_cab_1450.jpg` of the FlexGlue repository; from the LaMAR dataset (CAB building) | Not stated in the repository |

TUM and KITTI frames were re-encoded from PNG as JPEG (quality 88); the FlexGlue images are copied as they are.
Before the demo is published, the Berlin and LaMAR images need a stated licence or replacement.
