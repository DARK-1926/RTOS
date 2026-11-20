/* kp_monitor.c: Kalman plaxity (KP) in C, for the per-frame overhead measurement and C==Python check (Addendum E).
 * Params from kp_export.py (text):  lam q q0 me2e | med[5] | b[5] | v[5] | NQ | B[4][NQ]
 * Input CSV (from kp_export.py):    n1,n2,F,n5,e2e,has_prev   (one frame per line, time order)
 * Output CSV: p0,p1,p2,p3,us   -- P(miss) at release, N2 start, N3||N4 start, N5 start; us = monitor cost per frame
 * The frames are replayed causally exactly as a running monitor would see them: the cross-frame state is updated
 * only after a frame's e2e is known.
 * Build: gcc -O2 -o kp_monitor kp_monitor.c -lm      Run: kp_monitor params.txt frames.csv D out.csv
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#define NN 5
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static double lam, q, q0, me2e, med[NN], b[NN], v[NN]; static int NQ; static double *B[4], *logB[4];
/* P(exp(z)*B > slack), z ~ N(mu, sd^2), B uniform over the NQ quantiles */
static double pmiss(int k, double mu, double sd, double slack) {
    if (slack <= 0) return 1.0;
    double ls = log(slack), acc = 0, is = 1.0 / (sd * M_SQRT2);
    for (int i = 0; i < NQ; i++) acc += 0.5 * erfc((ls - logB[k][i] - mu) * is);
    return acc / NQ;
}
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: kp_monitor params.txt frames.csv D out.csv\n"); return 2; }
    FILE *fp = fopen(argv[1], "r"); if (!fp) return 3;
    if (fscanf(fp, "%lf %lf %lf %lf", &lam, &q, &q0, &me2e) != 4) return 4;
    for (int j = 0; j < NN; j++) if (fscanf(fp, "%lf", &med[j]) != 1) return 4;
    for (int j = 0; j < NN; j++) if (fscanf(fp, "%lf", &b[j]) != 1) return 4;
    for (int j = 0; j < NN; j++) if (fscanf(fp, "%lf", &v[j]) != 1) return 4;
    if (fscanf(fp, "%d", &NQ) != 1) return 4;
    for (int k = 0; k < 4; k++) { B[k] = malloc(8 * NQ); logB[k] = malloc(8 * NQ);
        for (int i = 0; i < NQ; i++) { if (fscanf(fp, "%lf", &B[k][i]) != 1) return 4; logB[k][i] = log(B[k][i]); } }
    fclose(fp);
    double D = atof(argv[3]);
    FILE *fi = fopen(argv[2], "r"), *fo = fopen(argv[4], "w"); if (!fi || !fo) return 5;
    fprintf(fo, "p0,p1,p2,p3,us\n");
    double x[4], e2e; int hp; double cur = 0; int have = 0;
    while (fscanf(fi, "%lf,%lf,%lf,%lf,%lf,%d", &x[0], &x[1], &x[2], &x[3], &e2e, &hp) == 6) {
        double t0 = now_us(), p[4];
        if (!hp) have = 0;
        double m = have ? cur : 0.0, qq = have ? q : q0;
        double prec = 1.0 / qq, num = m / qq, el = 0;
        for (int k = 0; k < 4; k++) {
            if (k > 0) { int j = k - 1; double xx = log((x[j] > 1e-3 ? x[j] : 1e-3) / med[j]) - b[j];
                         prec += 1.0 / v[j]; num += xx / v[j]; el += x[j]; }
            p[k] = pmiss(k, num / prec, sqrt(1.0 / prec), D - el);
        }
        double z = log(e2e / me2e); cur = have ? lam * cur + (1 - lam) * z : z; have = 1;   /* end of frame */
        double us = now_us() - t0;
        fprintf(fo, "%.6f,%.6f,%.6f,%.6f,%.2f\n", p[0], p[1], p[2], p[3], us);
    }
    fclose(fi); fclose(fo); return 0;
}
