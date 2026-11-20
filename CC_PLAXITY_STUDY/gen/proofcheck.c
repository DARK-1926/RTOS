/* Independent exhaustive check of the paper's window theorems (Addendum R).
 * Written from scratch; it shares no code with q_theorems.py, and it uses a different method. q_theorems.py explores abstract
 * reachable states. This program enumerates every concrete sequence of
 *   per frame j: predictor decision p_j (warn / not), outcome late_j (0/1), and delay d_j in {0..L-1}
 *   (frame j's outcome is unknown at the releases j+1 .. j+d_j and known from release j+d_j+1 on),
 * up to horizon T, simulates the rule, and checks every window of K frames. Integer arithmetic only.
 *
 * Checks (each for every m < K, every lag L in 1..LMAX, K in 2..KMAX, horizon T):
 *  C1 Theorem 3  (rule Wu, counts running unwarned frames): <= m unwarned misses in every K-window.
 *  C2 Theorem 4  (rule W, level 2 when count >= m; level-2 outcomes adversarial): <= m late non-level-2 frames per K-window,
 *                hence late <= m + late level-2 frames.
 *  C3 Prop. 3(3) (rule counting KNOWN outcomes only): max unwarned misses per window over all sequences == m + L - 1 (tight).
 *  C4 Prop. 2    (minimality): whenever Wu forces frame t, the adversary continuation "every counted running frame late, t late,
 *                t not warned" violates the bound; checked on every reachable prefix.
 *  C5 negative control: Wu with the count threshold off by one (forces only when count > m) must be caught violating C1.
 *  C6 Theorem 2  (long-run bound, one-sided, lag L): exhaustive over alarm/outcome choices allowed by the threshold, alpha = 1/4,
 *                gamma = 1 (theta scaled by 4 to stay integer), several theta_1: sum e_t <= alpha T + (theta_1 + (L+1) gamma)/gamma + L.
 * Build: gcc -O2 -o proofcheck proofcheck.c     Run: ./proofcheck
 */
#include <stdio.h>
#include <string.h>

#define TMAX 16
static int T, K, M, L, MODE; /* MODE 1: Wu, 2: W, 3: known-only, 5: Wu off-by-one */
static int p[TMAX], late[TMAX], d[TMAX], forced[TMAX];
static long long nseq, viol, minfail; static int worst;

static int known(int j, int t) { return t >= j + d[j] + 1; }        /* outcome of j known at the release of t */

static int count_at(int t) {
  int c = 0;
  for (int j = (t - K + 1 > 0 ? t - K + 1 : 0); j < t; j++) {
    int warned = p[j] || forced[j];
    if (MODE == 1 || MODE == 5) c += known(j, t) ? (late[j] && !warned) : !warned;
    else if (MODE == 2) c += known(j, t) ? late[j] : !forced[j];   /* forced = level 2 */
    else if (MODE == 3) c += known(j, t) && late[j] && !warned;
  }
  return c;
}

static void check_windows(int n) {      /* windows ending at n-1 only (earlier ones were checked at shorter prefixes) */
  int s = 0, a = n - K > 0 ? n - K : 0;
  for (int j = a; j < n; j++) {
    if (MODE == 2) s += late[j] && !forced[j];
    else s += late[j] && !(p[j] || forced[j]);
  }
  if (s > worst) worst = s;
  if (MODE != 3 && s > M) viol++;
}

static void minimality(int t) {          /* C4: called when Wu forces t */
  int s = 0;
  for (int j = (t - K + 1 > 0 ? t - K + 1 : 0); j < t; j++) {
    int warned = p[j] || forced[j];
    if (warned) continue;
    if (known(j, t)) s += late[j]; else s += 1;   /* adversary makes running unwarned frames late */
  }
  if (s + 1 <= M) minfail++;                       /* t late and not warned: must exceed m */
}

static void rec(int t) {
  if (t == T) { nseq++; return; }
  int c = count_at(t);
  int f = (MODE == 5) ? (c > M) : (c >= M);
  if (MODE == 3) f = (c >= M);
  forced[t] = f;
  if (f && MODE == 1) minimality(t);
  for (int pp = 0; pp <= (MODE == 2 ? 0 : 1); pp++)
    for (int ll = 0; ll <= 1; ll++)
      for (int dd = 0; dd < L; dd++) {
        p[t] = pp; late[t] = ll; d[t] = dd;
        check_windows(t + 1);
        rec(t + 1);
      }
}

static int run(int mode, int k, int m, int lag, int horizon) {
  MODE = mode; K = k; M = m; L = lag; T = horizon; nseq = viol = minfail = 0; worst = 0;
  rec(0); return worst;
}

/* C6: exhaustive long-run bound with integers. theta scaled by 4: alarm possible iff theta4 < 4 (score <= 1 can exceed theta < 1),
 * no-alarm possible iff theta4 >= 0 (score >= 0 cannot exceed theta >= 0). e_t = late and no alarm. Update with lag L. */
static long long c6n, c6v; static int TH0, LAG6, T6; static int e6[64];
static void rec6(int t, int th4) {
  if (t == T6) {
    int s = 0; for (int i = 0; i < T6; i++) s += e6[i];
    /* bound: sum e <= alpha*T + (theta1 + (L+1)*gamma)/gamma + L, alpha = 1/4, gamma = 1, theta1 = TH0/4 -> times 4 */
    if (4 * s > T6 + TH0 + 4 * (LAG6 + 1) + 4 * LAG6) c6v++;
    c6n++; return;
  }
  for (int al = 0; al <= 1; al++) {
    if (al && !(th4 < 4)) continue;
    if (!al && !(th4 >= 0)) continue;
    for (int lt = 0; lt <= 1; lt++) {
      e6[t] = lt && !al;
      int nth = th4;
      if (t - LAG6 >= 0) nth = th4 - 4 * e6[t - LAG6] + 1;     /* theta <- theta - gamma (e - alpha), scaled by 4 */
      rec6(t + 1, nth);
    }
  }
}

int main(void) {
  int ok = 1;
  printf("C1-C5: exhaustive over predictor decision, outcome and delay of every frame\n");
  for (int k = 2; k <= 5; k++) for (int m = 1; m < k; m++) for (int lag = 1; lag <= 3; lag++) {
    int h = (lag == 1) ? 12 : (lag == 2 ? 9 : 8);
    if (k >= 5 && lag >= 2) h = 8;
    run(1, k, m, lag, h); long long v1 = viol, mf = minfail, n1 = nseq;
    run(2, k, m, lag, h + 2); long long v2 = viol;
    int w3 = run(3, k, m, lag, h);
    run(5, k, m, lag, h); long long v5 = viol;
    int bound3 = m + lag - 1; if (bound3 > k) bound3 = k;
    int good = v1 == 0 && v2 == 0 && mf == 0 && w3 == bound3 && v5 > 0;
    if (!good) ok = 0;
    printf("K=%d m=%d L=%d T=%2d seq=%10lld  C1 viol=%lld  C2 viol=%lld  C3 worst=%d (expect %d)  C4 fail=%lld  C5 caught=%s  %s\n",
           k, m, lag, h, n1, v1, v2, w3, bound3, mf, v5 > 0 ? "yes" : "NO", good ? "OK" : "PROBLEM");
  }
  printf("C6: long-run bound, exhaustive over alarm/outcome choices allowed by the threshold\n");
  for (LAG6 = 0; LAG6 <= 2; LAG6++) for (TH0 = 0; TH0 <= 4; TH0 += 2) for (T6 = 6; T6 <= 14; T6 += 4) {
    c6n = c6v = 0; rec6(0, TH0);
    if (c6v) ok = 0;
    printf("L=%d theta1=%.2f T=%2d sequences=%9lld violations=%lld\n", LAG6, TH0 / 4.0, T6, c6n, c6v);
  }
  printf(ok ? "ALL CHECKS PASSED\n" : "SOME CHECK FAILED\n");
  return !ok;
}
