/* Exhaustive check of the cost theorem (Addendum S): under rule Wu, for every sequence of predictor decisions, outcomes and
 * completion delays d_j in {0..L-1},   m * F  <=  (K-1) * U  +  (L-1) * N,
 * F = frames forced by Wu, U = unwarned misses, N = unwarned frames (counted while running). Also reports, for L = 1,
 * the worst ratio m*F / ((K-1)*U) over all sequences (tightness).                Build: gcc -O2 -o pcc proofcheck_cost.c */
#include <stdio.h>
#define TMAX 16
static int T, K, M, L, p[TMAX], late[TMAX], d[TMAX], fo[TMAX]; static long long nseq, viol; static double worst;
static int known(int j, int t) { return t >= j + d[j] + 1; }
static void rec(int t) {
  if (t == T) { int F = 0, U = 0, N = 0;
    for (int j = 0; j < T; j++) { F += fo[j]; int w = p[j] || fo[j]; U += late[j] && !w; N += !w; }
    nseq++; if ((long long)M * F > (long long)(K - 1) * U + (long long)(L - 1) * N) viol++;
    if (U > 0) { double r = (double)M * F / ((K - 1) * U); if (r > worst) worst = r; } return; }
  int c = 0; for (int j = t - K + 1 > 0 ? t - K + 1 : 0; j < t; j++) { int w = p[j] || fo[j]; c += known(j, t) ? (late[j] && !w) : !w; }
  fo[t] = c >= M;
  for (int pp = 0; pp <= 1; pp++) for (int ll = 0; ll <= 1; ll++) for (int dd = 0; dd < L; dd++) { p[t] = pp; late[t] = ll; d[t] = dd; rec(t + 1); }
}
int main(void) { int ok = 1;
  for (int k = 2; k <= 6; k++) for (int m = 1; m < k; m++) for (int lag = 1; lag <= 3; lag++) {
    int h = lag == 1 ? 13 : (lag == 2 ? 10 : 8); if (k == 6 && lag > 1) h = 8;
    K = k; M = m; L = lag; T = h; nseq = viol = 0; worst = 0; rec(0); if (viol) ok = 0;
    printf("K=%d m=%d L=%d T=%2d seq=%11lld violations=%lld worst m*F/((K-1)U)=%.3f\n", k, m, lag, h, nseq, viol, worst); }
  printf(ok ? "ALL CHECKS PASSED\n" : "SOME CHECK FAILED\n"); return !ok; }
