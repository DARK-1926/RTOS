"""Full-resolution scans for the compiled C pipeline (G8): all 32 rings, float32 x,y,z.
Recreates gen/lidar_xyz_full exactly (verified bit-identical on scan 0000).
Usage: python lidar_prep_full.py <nuscenes-mini>/samples/LIDAR_TOP lidar_xyz_full"""
import glob, os, sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for i, fn in enumerate(sorted(glob.glob(os.path.join(src, "*.pcd.bin")))):
    np.fromfile(fn, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float32).tofile(os.path.join(dst, f"{i:04d}.xyz"))
print(i + 1, "scans written")
