"""Test data for the browser test page (web/test.html): the 6 test pairs of tools/compare_all.sh as raw RGBA,
their keypoints, and the native CPU backend's results on them at 512 px / 9 layers and 384 px / 5 layers, without
and with early stop (min_covisibility 0.8; decay 0.2 at 512 px, 0 at 384 px: pairs stop after layer 2, 4 or 8),
and the native shortlist of the 10 images at 512 px: the level-1 matrix, the level-2 score of every pair
(`jmt shortlist --k 9 --rerank`) and the pairs kept with k = 2.

    python tools/web_testdata.py --out build/wasm/web/testdata [--jmt build/release/jmt]

Each image is decoded once here (OpenCV, as tools/compare.py does), written as a lossless PNG for the native
`jmt match` and as raw RGBA (u32 width, u32 height, then the pixels) for the page, so both see exactly the
same pixels whatever the browser's decoder does. Keypoints come from just-keypoints (tools/common.py, cached in
build/keypoints). Writes manifest.json (pairs, settings, keypoints and reference results) next to them.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import jmtfile  # noqa: E402

A = os.path.join(common.ROOT, "third_party", "FlexGlue", "assets")
S = os.path.join(common.ROOT, "build", "synth")
PAIRS = [(f"{A}/berlin1.jpg", f"{A}/berlin2.jpg"), (f"{A}/lamar_cab_1446.jpg", f"{A}/lamar_cab_1450.jpg"),
         (f"{A}/berlin1.jpg", f"{A}/lamar_cab_1446.jpg"), (f"{S}/berlin1_h1.png", f"{S}/berlin1_h2.png"),
         (f"{S}/lamar_cab_1446_h1.png", f"{S}/lamar_cab_1446_h2.png"), (f"{S}/berlin2_h1.png", f"{S}/lamar_cab_1450_h2.png")]
SHORTLIST_K = 2
# size, layers, min_covisibility, early_stop_decay (< 0: off)
SETTINGS = [(512, 9, 0.0, -1.0), (384, 5, 0.0, -1.0), (512, 9, 0.8, 0.2), (384, 5, 0.8, 0.0)]


def rounded(a, digits=6):
    return [round(float(v), digits) for v in np.asarray(a).ravel()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (served next to test.html)")
    ap.add_argument("--jmt", default=os.path.join(common.ROOT, "build", "release", "jmt"), help="native jmt command line tool")
    ap.add_argument("--backend", default="cpu", help="backend of the native reference (default cpu)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp()
    images, pairs = {}, []
    for a, b in PAIRS:
        names = []
        for path in (a, b):
            name = os.path.splitext(os.path.basename(path))[0]
            if name not in images:
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                rgb = np.ascontiguousarray(bgr[..., ::-1])
                cv2.imwrite(os.path.join(tmp, name + ".png"), bgr)
                h, w = rgb.shape[:2]
                rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
                with open(os.path.join(args.out, name + ".rgba"), "wb") as f:
                    f.write(struct.pack("<II", w, h))
                    f.write(rgba.tobytes())
                xy = (common.keypoints(path) - 0.5).numpy()  # just-keypoints' convention
                jmtfile.write(os.path.join(tmp, name + ".jkpt"), {"kind": "keypoints"}, {"keypoints": xy})
                images[name] = {"rgba": name + ".rgba", "width": w, "height": h, "keypoints": rounded(xy)}
            names.append(name)
        refs = []
        for size, layers, min_covis, decay in SETTINGS:
            out = os.path.join(tmp, "out.jmtt")
            cmd = [args.jmt, "match", os.path.join(tmp, names[0] + ".png"), os.path.join(tmp, names[1] + ".png"),
                   "--kpts-a", os.path.join(tmp, names[0] + ".jkpt"), "--kpts-b", os.path.join(tmp, names[1] + ".jkpt"),
                   "--backend", args.backend, "--size", str(size), "--layers", str(layers), "--out", out]
            if decay >= 0:
                cmd += ["--min-covisibility", str(min_covis), "--early-stop", str(decay)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            _, t = jmtfile.read(out)
            refs.append({"size": size, "layers": layers, "min_covisibility": min_covis, "early_stop_decay": decay,
                         "matches": [int(v) for v in t["matches"].ravel()],
                         "scores": rounded(t["scores"]), "a_in_b": rounded(t["a_in_b"], 4), "b_in_a": rounded(t["b_in_a"], 4),
                         "conf_a": rounded(t["conf_a"]), "conf_b": rounded(t["conf_b"]),
                         "covisibility": float(t["covisibility"][0]), "stop_layer": int(t["stop_layer"][0])})
            stop = f", early stop {decay}: {int(t['stop_layer'][0])} layers" if decay >= 0 else ""
            print(f"{names[0]} - {names[1]}, {size} px, {layers} layers{stop}: {len(t['scores'])} matches", flush=True)
        pairs.append({"a": names[0], "b": names[1], "refs": refs})
    names = list(images)
    shortlist = {"images": names, "size": 512, "k": SHORTLIST_K}
    for key, extra in (("all", ["--k", str(len(names) - 1), "--rerank"]), ("kept", ["--k", str(SHORTLIST_K)])):
        out = os.path.join(tmp, "shortlist.jmtt")
        cmd = [args.jmt, "shortlist", *[os.path.join(tmp, n + ".png") for n in names], "--backend", args.backend,
               "--size", "512", "--out", out, *extra]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        _, t = jmtfile.read(out)
        shortlist[key] = {"pairs": [int(v) for v in t["pairs"].ravel()], "scores": rounded(t["scores"])}
        shortlist["level1"] = rounded(t["level1"])
    print(f"shortlist: {len(shortlist['kept']['scores'])} of {len(shortlist['all']['scores'])} pairs kept with k = {SHORTLIST_K}")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"reference": f"native {args.backend} backend", "images": images, "pairs": pairs,
                   "shortlist": shortlist}, f)
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
