"""Convert nuScenes LIDAR_TOP .pcd.bin (float32 x,y,z,intensity,ring) to compact float32 x,y,z files.
Keeps rings with ring % 4 == 0 (an 8-beam sensor, so pure Python keeps up on a Cortex-A53). Scene order is kept."""
import glob, os, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
fs = sorted(glob.glob(os.path.join(src, "*.pcd.bin")))
n = []
for i, fn in enumerate(fs):
    p = np.fromfile(fn, dtype=np.float32).reshape(-1, 5)
    q = p[(p[:, 4].astype(int) % 4) == 0][:, :3].astype(np.float32)
    q.tofile(os.path.join(dst, f"{i:04d}.xyz")); n.append(len(q))
print(len(fs), "scans; points/scan kept: min %d median %d max %d" % (min(n), int(np.median(n)), max(n)))
