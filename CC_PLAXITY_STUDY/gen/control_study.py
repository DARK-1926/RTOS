"""Trace-driven control case study: what does an early warning buy a controller?

Plant: linearised inverted pendulum, x = [angle, angular rate], dx/dt = [[0, 1], [w0^2, 0]] x + [0, 1]^T u + noise,
w0 = 3 rad/s (a ~1.1 m pendulum), zero-order hold at the pipeline period h (0.1 s lidar, 0.2 s drone). LQR (Q = diag(1, 0.1), R = 0.1).
The perception pipeline of the trace supplies the state estimate. Per frame, from the REAL trace and the monitor's decision:
  mode 0  on time, not warned : fresh estimate (noise sv)                      -> u = -K xhat
  mode 1  warned (alarm at checkpoint 0 or 1, i.e. before the frame is late)   -> fast fallback estimate (noise sf = ratio * sv), always on time
  mode 2  unwarned deadline miss: no update, the actuator holds the previous command
A false alarm therefore costs estimate quality; an unwarned miss costs a stale command on an unstable plant.
Metrics over the run (mean over noise seeds): LQR cost J, crashes per 1000 frames (|angle| > 0.5 rad, state reset), p99 |angle|.
Everything except the frame outcomes and alarms is simulated; this is a case study, not a hardware-in-the-loop result.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are

W0, SW, SV, RATIO, CRASH, NSEED = 3.0, 0.01, 0.002, 10.0, 0.5, 20
Q, R = np.diag([1.0, 0.1]), np.array([[0.1]])


def plant(h, w0=W0):
    A = np.array([[0.0, 1.0], [w0 ** 2, 0.0]]); B = np.array([[0.0], [1.0]])
    M = expm(np.block([[A, B], [np.zeros((1, 3))]]) * h)
    Ad, Bd = M[:2, :2], M[:2, 2:]
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return Ad, Bd, K


def simulate(mode, h, w0=W0, ratio=RATIO, nseed=NSEED):
    """mode: int array per frame (0 fresh, 1 fallback, 2 hold). Returns dict of mean metrics over noise seeds."""
    Ad, Bd, K = plant(h, w0); n = len(mode); J, C, P99 = [], [], []
    for s in range(nseed):
        rng = np.random.default_rng(1000 + s); x = np.zeros(2); u = 0.0; cost = 0.0; crash = 0; ang = np.empty(n)
        wn = rng.normal(0, SW, (n, 2)); vn = rng.normal(0, 1.0, (n, 2))
        for t in range(n):
            if mode[t] == 0:
                u = float(-K @ (x + SV * vn[t]))
            elif mode[t] == 1:
                u = float(-K @ (x + ratio * SV * vn[t]))
            cost += x @ Q @ x + R[0, 0] * u * u; ang[t] = abs(x[0])
            x = Ad @ x + Bd[:, 0] * u + wn[t]
            if abs(x[0]) > CRASH:
                crash += 1; x = np.zeros(2); u = 0.0
        J.append(cost / n); C.append(1000.0 * crash / n); P99.append(np.quantile(ang, 0.99))
    return dict(J=float(np.mean(J)), crash_per_1000=float(np.mean(C)), p99_angle=float(np.mean(P99)))


def modes(y, first=None, kind="monitor"):
    y = np.asarray(y)
    if kind == "none":
        return np.where(y == 1, 2, 0)
    if kind == "always":
        return np.ones(len(y), int)
    if kind == "oracle":
        return np.where(y == 1, 1, 0)
    return np.where(first >= 0, 1, np.where(y == 1, 2, 0))
