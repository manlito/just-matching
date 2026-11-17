# Demo sample images

Twelve images for `demo.html`, in capture order within each scene: four TUM frames, four KITTI frames, two
KITTI views of one street corner passed twice (a revisit, minutes apart and 22° apart in heading), and two KITTI
frames of different streets that look alike (a hard negative: FlexGlue's covisibility for them is 0.06).

| Files | Source | Licence |
|---|---|---|
| `tum-1` to `tum-4` | TUM RGB-D benchmark, frames 0, 4, 8 and 12 of the sample sequence in the ALIKED repository (0.4 s apart). Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard and Daniel Cremers, "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012. <https://cvg.cit.tum.de/data/datasets/rgbd-dataset> | CC BY 4.0 |
| `kitti-1` to `kitti-4` | KITTI odometry benchmark, sequence 00, frames 100, 103, 106 and 109, from the sample sequence in the ALIKED repository. Andreas Geiger, Philip Lenz and Raquel Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012. <https://www.cvlibs.net/datasets/kitti/> | CC BY-NC-SA 3.0 (non-commercial) |
| `kitti-revisit-1`, `kitti-revisit-2` | KITTI raw data, drive `2011_10_03_drive_0027` (odometry sequence 00), left colour camera (`image_02`), frames 745 and 3679: the same corner, 3 m apart. Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun, "Vision meets Robotics: The KITTI Dataset", IJRR 2013. <https://www.cvlibs.net/datasets/kitti/raw_data.php> | CC BY-NC-SA 3.0 (non-commercial) |
| `kitti-alike-1`, `kitti-alike-2` | The same drive and camera, frames 3100 and 1300: two streets 270 m apart, at least 147 m from every other sample. Same citation. | CC BY-NC-SA 3.0 (non-commercial) |

All images were re-encoded from PNG as JPEG (quality 88), otherwise unchanged. The KITTI images, and these JPEG
copies of them, are under CC BY-NC-SA 3.0 <https://creativecommons.org/licenses/by-nc-sa/3.0/>: they may be used
for non-commercial purposes only, with credit to the authors above, and shared under the same licence.

Sequence 00 ends where it starts, so frames near its end see the place of `kitti-1` to `kitti-4` (frame 4500 is
34 m from frame 100 and scores 0.9995 with them): a look-alike must be checked against every other sample, not
only against its partner.
