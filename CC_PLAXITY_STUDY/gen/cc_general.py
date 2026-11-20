#!/usr/bin/env python3
"""Generalization runs for CC-plaxity (see PROTOCOL_GENERALIZATION.md). Python standard library only.

Same trace schema as unoq_cc_confirm.py, so the frozen analysis (cc_plaxity.run_unoq) applies unchanged.

  --workload synthetic : identical node work/jitter to unoq_cc_confirm.py (N1->N2->(N3||N4)->N5)
  --workload lidar     : real nuScenes LIDAR_TOP scans (pre-converted by lidar_prep.py), pure-Python stages:
                         N1 decode+crop, N2 voxel downsample, N3 ground segmentation || N4 BEV height map
                         (on two worker processes), N5 sector free-space fusion. No artificial jitter:
                         all variation comes from scene content and contention.
  --hog mem            : memory-streaming hogs (as in unoq_cc_confirm.py)
  --hog cpu            : ALU-only hogs (no memory traffic beyond L1): contention through CPU time-slicing
  --hog_mult m         : K hidden level -> K*m hog processes (scales contention to platforms with more cores)
Hidden K follows a random telegraph schedule (dwell ~ Exp(mean 4 s)); K in {0,1,2} in calib, {0,1,2,3} in val.
"""
import argparse, array, csv, glob, math, os, random, time
from multiprocessing import Process, Pool, Event

BUF = bytearray(4 * 1024 * 1024)
SCALE = 1.0


def work(units):
    n = int(units * SCALE)
    s, b, L = 0, BUF, len(BUF)
    for i in range(n):
        j = (i * 4099) % L
        s += b[j]
        b[j] = (s & 0xFF)
    return s


def calibrate():
    global SCALE
    SCALE = 20000.0
    t = time.perf_counter(); work(20); dt = (time.perf_counter() - t) * 1000
    SCALE = SCALE * 20 / max(dt, 1e-3)


def hog_mem(stop):
    b = bytearray(16 * 1024 * 1024); L = len(b); i = 0
    while not stop.is_set():
        for _ in range(200000):
            b[i] = (b[i] + 1) & 0xFF; i = (i + 4097) % L


def hog_cpu(stop):
    x = 1
    while not stop.is_set():
        for _ in range(200000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF


def branch(args):
    units, = args
    t = time.perf_counter(); work(units); return (time.perf_counter() - t) * 1000


# ---------------- real LiDAR stages ----------------
def lidar_n1(path):
    a = array.array("f");
    with open(path, "rb") as f:
        a.frombytes(f.read())
    pts = []
    for i in range(0, len(a), 3):
        x, y, z = a[i], a[i + 1], a[i + 2]
        if -3.0 < z < 4.0 and x * x + y * y < 2500.0:
            pts.append((x, y, z))
    return pts


def lidar_n2(pts, v=0.3):
    acc = {}
    for x, y, z in pts:
        k = (int(x // v), int(y // v), int(z // v))
        s = acc.get(k)
        if s is None:
            acc[k] = [x, y, z, 1]
        else:
            s[0] += x; s[1] += y; s[2] += z; s[3] += 1
    return [(s[0] / s[3], s[1] / s[3], s[2] / s[3]) for s in acc.values()]


def lidar_n3(args):                      # ground segmentation (worker)
    vox, = args
    t = time.perf_counter()
    zmin = {}
    for x, y, z in vox:
        k = (int(x // 1.0), int(y // 1.0))
        if z < zmin.get(k, 1e9):
            zmin[k] = z
    obst = {}
    for x, y, z in vox:
        k = (int(x // 1.0), int(y // 1.0))
        if z - zmin[k] > 0.3:
            obst[k] = obst.get(k, 0) + 1
    cells = [k for k, c in obst.items() if c >= 2]
    return (time.perf_counter() - t) * 1000, cells


def lidar_n4(args):                      # BEV max-height map + per-sector nearest return (worker)
    vox, = args
    t = time.perf_counter()
    hmax, near = {}, [99.0] * 72
    for x, y, z in vox:
        k = (int(x // 0.5), int(y // 0.5))
        if z > hmax.get(k, -1e9):
            hmax[k] = z
        r = math.sqrt(x * x + y * y)
        s = int((math.atan2(y, x) + math.pi) / (2 * math.pi) * 72) % 72
        if z > -1.2 and r < near[s]:
            near[s] = r
    tall = sum(1 for h in hmax.values() if h > 0.5)
    return (time.perf_counter() - t) * 1000, (tall, near)


def lidar_branch(args):
    which, vox = args
    return lidar_n3((vox,)) if which == 3 else lidar_n4((vox,))


def lidar_n5(cells, near):
    occ = [0] * 72
    for cx, cy in cells:
        s = int((math.atan2(cy + 0.5, cx + 0.5) + math.pi) / (2 * math.pi) * 72) % 72
        occ[s] += 1
    return sum(1 for s in range(72) if occ[s] == 0 and near[s] > 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workload", choices=["synthetic", "lidar"], default="synthetic")
    ap.add_argument("--hog", choices=["mem", "cpu"], default="mem")
    ap.add_argument("--hog_mult", type=int, default=1)
    ap.add_argument("--lidar_dir", default="lidar_xyz")
    ap.add_argument("--calib_frames", type=int, default=3000)
    ap.add_argument("--val_frames", type=int, default=2000)
    ap.add_argument("--period_ms", type=float, default=100.0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    calibrate()
    pool = Pool(2, initializer=calibrate)
    hog_fn = hog_mem if a.hog == "mem" else hog_cpu
    hogs = []
    nominal = {"n1": 2.0, "n2": 6.0, "n3": 12.0, "n4": 13.0, "n5": 0.5}
    scans = sorted(glob.glob(os.path.join(a.lidar_dir, "*.xyz"))) if a.workload == "lidar" else []
    if a.workload == "lidar" and not scans:
        raise SystemExit("no .xyz scans in " + a.lidar_dir)

    def set_k(k):
        want = k * a.hog_mult
        while len(hogs) < want:
            e = Event(); p = Process(target=hog_fn, args=(e,), daemon=True); p.start(); hogs.append((p, e))
        while len(hogs) > want:
            p, e = hogs.pop(); e.set(); p.join(timeout=2)

    total = a.calib_frames + a.val_frames
    k, t_switch = 0, time.monotonic()
    f = open(a.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["global_frame_id", "regime", "split", "bag_frame_idx", "dur_node1_ms", "dur_node2_ms",
                "dur_node3_ms", "dur_node4_ms", "dur_node5_ms", "end_to_end_ms", "hog_k", "seed"])
    next_rel = time.monotonic()
    for fr in range(total):
        split = "calib" if fr < a.calib_frames else "val"
        if time.monotonic() >= t_switch:
            levels = [0, 1, 2] if split == "calib" else [0, 1, 2, 3]
            k = rng.choice([x for x in levels if x != k])
            set_k(k)
            t_switch = time.monotonic() + rng.expovariate(1 / 4.0)
        while time.monotonic() < next_rel:
            time.sleep(0.0005)
        t0 = time.perf_counter()
        if a.workload == "synthetic":
            jit = lambda m: m * math.exp(rng.gauss(0, 0.25))
            t = time.perf_counter(); work(jit(nominal["n1"])); d1 = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); work(jit(nominal["n2"])); d2 = (time.perf_counter() - t) * 1000
            d3, d4 = pool.map(branch, [(jit(nominal["n3"]),), (jit(nominal["n4"]),)])
            t = time.perf_counter(); work(jit(nominal["n5"])); d5 = (time.perf_counter() - t) * 1000
            idx = fr % 164
        else:
            idx = fr % len(scans)
            t = time.perf_counter(); pts = lidar_n1(scans[idx]); d1 = (time.perf_counter() - t) * 1000
            t = time.perf_counter(); vox = lidar_n2(pts); d2 = (time.perf_counter() - t) * 1000
            (d3, cells), (d4, (tall, near)) = pool.map(lidar_branch, [(3, vox), (4, vox)])
            t = time.perf_counter(); lidar_n5(cells, near); d5 = (time.perf_counter() - t) * 1000
        e2e = (time.perf_counter() - t0) * 1000
        w.writerow([fr, f"K{k}", split, idx, f"{d1:.4f}", f"{d2:.4f}", f"{d3:.4f}", f"{d4:.4f}",
                    f"{d5:.4f}", f"{e2e:.4f}", k, a.seed])
        next_rel = max(next_rel + a.period_ms / 1000, time.monotonic())
        if fr % 500 == 0:
            f.flush(); print(f"frame {fr}/{total} K={k} e2e={e2e:.1f} ms", flush=True)
    set_k(0); pool.close(); f.close()


if __name__ == "__main__":
    main()
