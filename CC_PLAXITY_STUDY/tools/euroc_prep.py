"""EuRoC cam0 -> raw 8-bit grayscale frames for vo_ros_tm. Every 3rd frame. Calibration = MH_04_difficult (machine hall),
validation = V1_02_medium (Vicon room 1): environment-disjoint. Output <dst>/%04d.gray (752x480), idx_calib.txt, idx_val.txt."""
import io, sys, os, zipfile, hashlib
import numpy as np
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
i = 0; idx = {}
for split, z in [("calib", "MH_04_difficult.zip"), ("val", "V1_02_medium.zip")]:
    Z = zipfile.ZipFile(os.path.join(src, z))
    names = sorted(n for n in Z.namelist() if "/cam0/data/" in n and n.endswith(".png"))[::3]
    idx[split] = []
    for n in names:
        im = np.asarray(Image.open(io.BytesIO(Z.read(n))).convert("L"), dtype=np.uint8)
        assert im.shape == (480, 752), im.shape
        im.tofile(os.path.join(dst, f"{i:04d}.gray")); idx[split].append(i); i += 1
    open(os.path.join(dst, f"idx_{split}.txt"), "w").write("\n".join(map(str, idx[split])) + "\n")
    print(split, z, len(names), "frames")
print("total", i)
