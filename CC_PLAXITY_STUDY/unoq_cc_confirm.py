#!/usr/bin/env python3
"""UNO Q CONFIRMATORY run for CC-plaxity (frozen 2026-11-06, before any board data was collected).

Python standard library only. Run ON the Arduino UNO Q (Debian, 4 x A53):
    python3 unoq_cc_confirm.py --out unoq_cc_traces.csv --seed 1
Repeat with --seed 1..5 (five independent runs). Copy the CSVs back to D:\\RTKK\\CC_PLAXITY_STUDY\\board\\
and analyze with:  python cc_plaxity.py --unoq board/   (loader 'unoq').

Pipeline (same shape as the LiDAR DAG): N1 -> N2 -> (N3 || N4 on two worker processes) -> N5.
Each node does calibrated CPU+memory work with lognormal jitter. Background contention = K busy
memory-streaming processes; K follows a hidden random-telegraph schedule (dwell ~ Exp(mean 4 s)).
Detectors never see K.

Frozen hypotheses (evaluated with the frozen code in cc_plaxity.py, no re-tuning):
 H1  cp0 AUROC: CC-plaxity > Toba-2026 independence plaxity (= 0.5 by construction), CI excludes 0.
 H2  cp1 Brier: CC-within (no temporal) < independence plaxity, block-bootstrap CI of the difference excludes 0.
     (AUROC cannot differ at cp1: both scores are monotone in N1's time; the coupling term only fixes calibration.)
 H3  Unseen-contention calibration: frames with K = 3 never appear in calibration. On those frames, ECE at cp0
     of CC-plaxity < ECE of deadline-conditional LR. (This is the claim that was only exploratory on laptop data.)
 Kill: if H3 fails on >= 3 of 5 seeds, drop the "structure helps under shift" claim.
Deadline: D = the 85th percentile of end-to-end latency on the seed's calibration frames (rounded to 1 ms),
fixed before validation frames are scored.
"""
import argparse, csv, math, os, random, time
from multiprocessing import Process, Pool, Event

BUF = bytearray(4 * 1024 * 1024)            # 4 MB: larger than the A53 shared L2, so contention is real


def work(units):
    """~units * 1 ms of mixed ALU/memory work on an idle core (calibrated by calibrate())."""
    n = int(units * SCALE)
    s, b, L = 0, BUF, len(BUF)
    for i in range(n):
        j = (i * 4099) % L
        s += b[j]
        b[j] = (s & 0xFF)
    return s


SCALE = 1.0


def calibrate():
    global SCALE
    SCALE = 20000.0
    t = time.perf_counter(); work(20); dt = (time.perf_counter() - t) * 1000
    SCALE = SCALE * 20 / max(dt, 1e-3)


def hog(stop):
    b = bytearray(16 * 1024 * 1024); L = len(b); i = 0
    while not stop.is_set():
        for _ in range(200000):
            b[i] = (b[i] + 1) & 0xFF; i = (i + 4097) % L


def branch(args):
    units, = args
    t = time.perf_counter(); work(units); return (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="unoq_cc_traces.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--calib_frames", type=int, default=3000)
    ap.add_argument("--val_frames", type=int, default=2000)
    ap.add_argument("--period_ms", type=float, default=100.0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    calibrate()
    pool = Pool(2, initializer=calibrate)
    hogs = []                                   # (process, stop_event)
    nominal = {"n1": 2.0, "n2": 6.0, "n3": 12.0, "n4": 13.0, "n5": 0.5}   # ms of idle work (LiDAR-like)

    def set_k(k):
        while len(hogs) < k:
            e = Event(); p = Process(target=hog, args=(e,), daemon=True); p.start(); hogs.append((p, e))
        while len(hogs) > k:
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
        if time.monotonic() >= t_switch:        # hidden random-telegraph contention schedule
            levels = [0, 1, 2] if split == "calib" else [0, 1, 2, 3]   # K=3 only in validation (H3)
            k = rng.choice([x for x in levels if x != k])
            set_k(k)
            t_switch = time.monotonic() + rng.expovariate(1 / 4.0)
        jit = lambda m: m * math.exp(rng.gauss(0, 0.25))
        while time.monotonic() < next_rel:
            time.sleep(0.0005)
        t0 = time.perf_counter()
        t = time.perf_counter(); work(jit(nominal["n1"])); d1 = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); work(jit(nominal["n2"])); d2 = (time.perf_counter() - t) * 1000
        d3, d4 = pool.map(branch, [(jit(nominal["n3"]),), (jit(nominal["n4"]),)])
        t = time.perf_counter(); work(jit(nominal["n5"])); d5 = (time.perf_counter() - t) * 1000
        e2e = (time.perf_counter() - t0) * 1000
        w.writerow([fr, f"K{k}", split, fr % 164, f"{d1:.4f}", f"{d2:.4f}", f"{d3:.4f}", f"{d4:.4f}",
                    f"{d5:.4f}", f"{e2e:.4f}", k, a.seed])
        next_rel = max(next_rel + a.period_ms / 1000, time.monotonic())
        if fr % 500 == 0:
            f.flush(); print(f"frame {fr}/{total} K={k} e2e={e2e:.1f} ms", flush=True)
    set_k(0); pool.close(); f.close()


if __name__ == "__main__":
    main()
