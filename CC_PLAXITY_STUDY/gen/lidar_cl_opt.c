/* lidar_cl_opt.c (EXPLORATORY overhead variant of lidar_cl.c, written after D5 failed; same math):
 *   log-KDE precomputed once at load (no log() per grid point), grid points with prior < 1e-9 skipped.
 * lidar_cl.c: compiled LiDAR pipeline WITH inline deadline-miss monitors and closed-loop degradation (Addendum D).
 *
 * Pipeline, hogs and CSV schema are those of lidar_c.c (frozen, G8), plus:
 *  - scene-disjoint scan schedule: frames cycle over the scan indices listed in <idx_file>
 *  - inline monitors at every checkpoint (all computed on every frame, so every policy pays the same overhead):
 *      CC     CC-plaxity (one latent slowdown, sticky HMM, KDE evidence, remaining-time tables), exported by export_tables.py
 *      CCSC   CC with the self-check: fall back to NAIVE when >= 7 of CC's last 20 release-time PIT values are in the 5% tails
 *      NAIVE  NaiveScale at every checkpoint (calib remaining-time samples scaled by last valid e2e / median calib e2e)
 *      TOBA   equation-level Toba-2026 plaxity-CDFs (Eqs. 8-12) checked at the start of every node
 *    Alarm rule for every method: P(meet) < theta_v with Toba's variable thresholds (Eqs. 13, 15-18: job-count workload, th_min 0.8, th_max 1.0).
 *  - closed loop (policy != none): on the first alarm of a frame,
 *      alarm before N2 starts  -> keep every 2nd point before N2 (N2, N3, N4 all get cheaper)    [degraded = 1]
 *      alarm before N3/N4      -> N3 and N4 use every 2nd voxel                                  [degraded = 2]
 *      later alarms            -> logged only (nothing left to save)
 *  - per-frame monitor cost (us) for each monitor, measured with CLOCK_MONOTONIC.
 * Degraded frames give partial evidence: nodes after the degradation point are not used to update CC, and NAIVE
 * keeps the last undegraded e2e. PIT values are only recorded for undegraded frames.
 *
 * Build: gcc -O2 -o lidar_cl lidar_cl.c -lm -lpthread
 * Run:   lidar_cl run <scan_dir> <idx_file> <out.csv> <seed> <nframes> <calib|val> <period_ms> <hog_mult> <tables.bin|-> <none|cc|ccsc|naive|toba>
 *        lidar_cl quality <scan_dir> <out.csv>      (per-scan output change caused by each degradation level)
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define NSCAN_MAX 2048
#define PTS_MAX 40000
#define GRID1 100
#define GRID05 200
#define NSEC 72
#define HOG_MAX 64

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void sleep_ms(double ms) { struct timespec t; t.tv_sec = (time_t)(ms / 1e3); t.tv_nsec = (long)((ms - t.tv_sec * 1e3) * 1e6); nanosleep(&t, NULL); }

static uint64_t rs;
static uint64_t rnd(void) { uint64_t z = (rs += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static double urand(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

typedef struct { pthread_t th; volatile int stop; } hog_t;
static hog_t hogs[HOG_MAX]; static int nhog = 0;
static void *hog_fn(void *a) {
    hog_t *h = (hog_t *)a; size_t L = 16u << 20; volatile unsigned char *b = malloc(L); size_t i = 0;
    memset((void *)b, 1, L);
    while (!h->stop) for (int k = 0; k < 200000; k++) { b[i] = (unsigned char)(b[i] + 1); i = (i + 4097) % L; }
    free((void *)b); return NULL;
}
static void set_hogs(int want) {
    while (nhog < want) { hogs[nhog].stop = 0; pthread_create(&hogs[nhog].th, NULL, hog_fn, &hogs[nhog]); nhog++; }
    while (nhog > want) { nhog--; hogs[nhog].stop = 1; pthread_join(hogs[nhog].th, NULL); }
}

/* ---------------- pipeline (identical to lidar_c.c, plus decimation hooks) ---------------- */
typedef struct { float x, y, z; } pt;
typedef struct { uint64_t key; float x, y, z; } kp;
static pt pts[PTS_MAX], vox[PTS_MAX]; static kp keyed[PTS_MAX]; static float rawbuf[PTS_MAX * 3];
static int npts, nvox, vstep = 1;
static int n1_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    size_t n = fread(rawbuf, sizeof(float), PTS_MAX * 3, f); fclose(f); npts = 0;
    for (size_t i = 0; i + 2 < n; i += 3) { float x = rawbuf[i], y = rawbuf[i + 1], z = rawbuf[i + 2];
        if (z > -3.0f && z < 4.0f && x * x + y * y < 2500.0f) { pts[npts].x = x; pts[npts].y = y; pts[npts].z = z; npts++; } }
    return npts;
}
static void decimate_points(void) { int j = 0; for (int i = 0; i < npts; i += 2) pts[j++] = pts[i]; npts = j; }
static int cmpk(const void *a, const void *b) { uint64_t ka = ((const kp *)a)->key, kb = ((const kp *)b)->key; return (ka > kb) - (ka < kb); }
static void n2_voxel(float v) {
    for (int i = 0; i < npts; i++) {
        uint64_t ix = (uint64_t)(int64_t)floorf((pts[i].x + 64.f) / v), iy = (uint64_t)(int64_t)floorf((pts[i].y + 64.f) / v),
                 iz = (uint64_t)(int64_t)floorf((pts[i].z + 8.f) / v);
        keyed[i].key = (ix << 40) | (iy << 20) | iz; keyed[i].x = pts[i].x; keyed[i].y = pts[i].y; keyed[i].z = pts[i].z; }
    qsort(keyed, npts, sizeof(kp), cmpk); nvox = 0;
    for (int i = 0; i < npts;) { int j = i; double sx = 0, sy = 0, sz = 0;
        while (j < npts && keyed[j].key == keyed[i].key) { sx += keyed[j].x; sy += keyed[j].y; sz += keyed[j].z; j++; }
        vox[nvox].x = (float)(sx / (j - i)); vox[nvox].y = (float)(sy / (j - i)); vox[nvox].z = (float)(sz / (j - i)); nvox++; i = j; }
}
static int cell(float c, float res, int n) { int i = (int)floorf((c + 50.f) / res); return i < 0 ? 0 : (i >= n ? n - 1 : i); }
static unsigned char obst[GRID1 * GRID1];
static double n3_ground(void) {
    double t = now_ms(); static float zmin[GRID1 * GRID1]; static int cnt[GRID1 * GRID1];
    for (int i = 0; i < GRID1 * GRID1; i++) { zmin[i] = 1e9f; cnt[i] = 0; }
    for (int i = 0; i < nvox; i += vstep) { int c = cell(vox[i].x, 1.f, GRID1) * GRID1 + cell(vox[i].y, 1.f, GRID1); if (vox[i].z < zmin[c]) zmin[c] = vox[i].z; }
    for (int i = 0; i < nvox; i += vstep) { int c = cell(vox[i].x, 1.f, GRID1) * GRID1 + cell(vox[i].y, 1.f, GRID1); if (vox[i].z - zmin[c] > 0.3f) cnt[c]++; }
    for (int i = 0; i < GRID1 * GRID1; i++) obst[i] = cnt[i] >= 2;
    return now_ms() - t;
}
static float nearsec[NSEC]; static int tall;
static void *n4_bev(void *out) {
    double t = now_ms(); static float hmax[GRID05 * GRID05];
    for (int i = 0; i < GRID05 * GRID05; i++) hmax[i] = -1e9f;
    for (int s = 0; s < NSEC; s++) nearsec[s] = 99.f;
    for (int i = 0; i < nvox; i += vstep) { float x = vox[i].x, y = vox[i].y, z = vox[i].z;
        int c = cell(x, .5f, GRID05) * GRID05 + cell(y, .5f, GRID05); if (z > hmax[c]) hmax[c] = z;
        float r = sqrtf(x * x + y * y); int s = (int)((atan2f(y, x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC;
        if (z > -1.2f && r < nearsec[s]) nearsec[s] = r; }
    tall = 0; for (int i = 0; i < GRID05 * GRID05; i++) tall += hmax[i] > 0.5f;
    *(double *)out = now_ms() - t; return NULL;
}
static uint64_t freemask[2];
static int n5_fuse(void) {
    int occ[NSEC] = {0}, freec = 0; freemask[0] = freemask[1] = 0;
    for (int cx = 0; cx < GRID1; cx++) for (int cy = 0; cy < GRID1; cy++) if (obst[cx * GRID1 + cy]) {
        float x = cx - 50.f + .5f, y = cy - 50.f + .5f; occ[(int)((atan2f(y, x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC]++; }
    for (int s = 0; s < NSEC; s++) if (occ[s] == 0 && nearsec[s] > 10.f) { freec++; freemask[s / 64] |= 1ULL << (s % 64); }
    return freec;
}
static int run_frame(const char *path, int mode, double *d) {   /* mode 0 full, 1 decimate points, 2 decimate voxels */
    double t; t = now_ms(); n1_load(path); d[0] = now_ms() - t;
    if (mode == 1) decimate_points();
    vstep = (mode == 2) ? 2 : 1;
    t = now_ms(); n2_voxel(0.2f); d[1] = now_ms() - t;
    pthread_t w; d[3] = 0; pthread_create(&w, NULL, n4_bev, &d[3]); d[2] = n3_ground(); pthread_join(w, NULL);
    t = now_ms(); int fc = n5_fuse(); d[4] = now_ms() - t; vstep = 1; return fc;
}

/* ---------------- monitor tables (export_tables.py) ---------------- */
static int G, M, MN, NZ, NB[7];
static double Dl, rho, med, beta[5], *grid, *pi_, thr[7], zg0, zg1, *fg[3];
static float *R[4], *NR[4], *cdf[7];
#define EV 3   /* evidence nodes n1, n2, F */
static double *lfg[3];
static int load_tables(const char *fn) {
    FILE *f = fopen(fn, "rb"); if (!f) return -1; int32_t h[10];
    if (fread(h, 4, 10, f) != 10) { return -2; }
    G = h[0]; M = h[1]; MN = h[2]; NZ = h[3]; for (int v = 1; v <= 6; v++) NB[v] = h[3 + v];
    double s[8]; if (fread(s, 8, 8, f) != 8) return -3; Dl = s[0]; rho = s[1]; med = s[2]; zg0 = s[3]; zg1 = s[4];
    if (fread(beta, 8, 5, f) != 5) return -4;
    grid = malloc(8 * G); pi_ = malloc(8 * G); if (fread(grid, 8, G, f) != (size_t)G || fread(pi_, 8, G, f) != (size_t)G) return -5;
    if (fread(thr + 1, 8, 6, f) != 6) return -6;
    for (int c = 0; c < EV; c++) { fg[c] = malloc(8 * NZ); if (fread(fg[c], 8, NZ, f) != (size_t)NZ) return -7;
        lfg[c] = malloc(8 * NZ); for (int i = 0; i < NZ; i++) lfg[c][i] = fg[c][i] > 1e-250 ? log(fg[c][i] + 1e-300) : -690.7755; }
    for (int k = 0; k < 4; k++) { R[k] = malloc(4 * (size_t)G * M); if (fread(R[k], 4, (size_t)G * M, f) != (size_t)G * M) return -8; }
    for (int k = 0; k < 4; k++) { NR[k] = malloc(4 * MN); if (fread(NR[k], 4, MN, f) != (size_t)MN) return -9; }
    for (int v = 1; v <= 6; v++) { cdf[v] = malloc(4 * NB[v]); if (fread(cdf[v], 4, NB[v], f) != (size_t)NB[v]) return -10; }
    fclose(f); return 0;
}

static double kde_log(int c, double z) {   /* interpolate the precomputed log-density (approximation of log of interpolated KDE) */
    double u = (z - zg0) / (zg1 - zg0) * (NZ - 1); if (u < 0 || u > NZ - 1) return -690.7755;
    int i = (int)u; double a = u - i; return (i + 1 < NZ) ? lfg[c][i] * (1 - a) + lfg[c][i + 1] * a : lfg[c][i];
}
static int ub(const float *a, int n, double x) { int lo = 0, hi = n; while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m; } return lo; }
/* posterior weights for `nev` completed evidence nodes, from prior pr */
static void post_w(const double *pr, const double *ev, int nev, double *w) {
    double mx = -1e300;
    double le[3]; for (int c = 0; c < nev; c++) le[c] = log(ev[c]);
    for (int g = 0; g < G; g++) { if (pr[g] < 1e-9) { w[g] = -1e300; continue; }
        double l = 0; for (int c = 0; c < nev; c++) l += kde_log(c, le[c] - beta[c] * grid[g]); w[g] = l; if (l > mx) mx = l; }
    double s = 0; for (int g = 0; g < G; g++) { w[g] = (pr[g] > 1e-300 && w[g] - mx > -700) ? pr[g] * exp(w[g] - mx) : 0; s += w[g]; }
    for (int g = 0; g < G; g++) w[g] /= s;
}
static double cc_p(const double *w, int k, double lax) {
    double p = 0; for (int g = 0; g < G; g++) if (w[g] > 1e-9) p += w[g] * (1.0 - (double)ub(R[k] + (size_t)g * M, M, lax) / M);
    return p;
}
static double naive_p(int k, double lax, double s) { return 1.0 - (double)ub(NR[k], MN, lax / s) / MN; }
static double toba_meet(int v, double ast) { int b = (int)lround((ast > 0 ? ast : 0) * 10.0); return b < NB[v] ? cdf[v][b] : 0.0; }

/* ---------------- quality mode ---------------- */
static int popc(uint64_t x) { int c = 0; while (x) { x &= x - 1; c++; } return c; }
static int quality(const char *dir, const char *out) {
    FILE *o = fopen(out, "w"); fprintf(o, "scan_idx,free_full,free_m1,free_m2,sector_diff_m1,sector_diff_m2,tall_full,tall_m1,tall_m2\n");
    char p[512]; double d[5];
    for (int i = 0; i < NSCAN_MAX; i++) {
        snprintf(p, sizeof p, "%s/%04d.xyz", dir, i); FILE *f = fopen(p, "rb"); if (!f) break; fclose(f);
        int f0 = run_frame(p, 0, d); uint64_t m0a = freemask[0], m0b = freemask[1]; int t0 = tall;
        int f1 = run_frame(p, 1, d); int s1 = popc(freemask[0] ^ m0a) + popc(freemask[1] ^ m0b); int t1 = tall;
        int f2 = run_frame(p, 2, d); int s2 = popc(freemask[0] ^ m0a) + popc(freemask[1] ^ m0b); int t2 = tall;
        fprintf(o, "%d,%d,%d,%d,%d,%d,%d,%d,%d\n", i, f0, f1, f2, s1, s2, t0, t1, t2);
    }
    fclose(o); return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "quality")) return quality(argv[2], argv[3]);
    if (argc < 12) { fprintf(stderr, "usage: see header\n"); return 2; }
    const char *dir = argv[2]; int seed = atoi(argv[5]), nfr = atoi(argv[6]), val = !strcmp(argv[7], "val"), mult = atoi(argv[9]);
    double period = atof(argv[8]); const char *pol = argv[11];
    int policy = !strcmp(pol, "cc") ? 1 : !strcmp(pol, "ccsc") ? 2 : !strcmp(pol, "naive") ? 3 : !strcmp(pol, "toba") ? 4 : 0;
    int mon = strcmp(argv[10], "-") != 0;
    if (mon) { int e = load_tables(argv[10]); if (e) { fprintf(stderr, "tables error %d\n", e); return 2; } }
    rs = 0x1234567ULL + (uint64_t)seed * 0x9E3779B97F4A7C15ULL + (uint64_t)val * 0xD1B54A32D192ED03ULL;
    static int idx[NSCAN_MAX]; int nidx = 0; FILE *fi = fopen(argv[3], "r");
    while (nidx < NSCAN_MAX && fscanf(fi, "%d", &idx[nidx]) == 1) { nidx++; }
    fclose(fi);
    static char path[512];
    FILE *out = fopen(argv[4], "w");
    fprintf(out, "global_frame_id,regime,split,bag_frame_idx,dur_node1_ms,dur_node2_ms,dur_node3_ms,dur_node4_ms,dur_node5_ms,"
                 "end_to_end_ms,hog_k,seed,policy,alarm_cp,degraded,p0_cc,p0_ccsc,p0_naive,pmeet1_toba,sc_flag,pit,"
                 "us_cc,us_ccsc_extra,us_naive,us_toba,us_total,freec\n");
    double *prior = malloc(8 * G + 8), *post = malloc(8 * G + 8), *w = malloc(8 * G + 8);
    int have_post = 0; double last_e2e = -1; int pit_hist[20] = {0}, npit = 0;
    int k = 0; double t_switch = now_ms(), next_rel = now_ms();
    for (int fr = 0; fr < nfr; fr++) {
        if (now_ms() >= t_switch) {
            int lv[4], nl = 0, top = val ? 3 : 2;
            for (int x = 0; x <= top; x++) if (x != k) lv[nl++] = x;
            k = lv[rnd() % nl]; set_hogs(k * mult); t_switch = now_ms() + (-log(1.0 - urand())) * 4000.0;
        }
        while (now_ms() < next_rel) sleep_ms(0.5);
        int si = idx[fr % nidx]; snprintf(path, sizeof path, "%s/%04d.xyz", dir, si);
        double t0 = now_ms(), d[5] = {0}, us_cc = 0, us_sc = 0, us_nv = 0, us_tb = 0, tm;
        double p0cc = NAN, p0sc = NAN, p0nv = NAN, pm1 = NAN, s = 1.0; int flag = 0, alarm = -1, degr = 0;
        /* ---- release (cp0) ---- */
        if (mon) {
            tm = now_ms();
            for (int g = 0; g < G; g++) prior[g] = have_post ? rho * post[g] + (1 - rho) * pi_[g] : pi_[g];
            p0cc = cc_p(prior, 0, Dl); us_cc += (now_ms() - tm) * 1e3;
            tm = now_ms(); int cnt = 0; for (int j = 0; j < (npit < 20 ? npit : 20); j++) cnt += pit_hist[j]; flag = cnt >= 7; us_sc += (now_ms() - tm) * 1e3;
            tm = now_ms(); s = last_e2e > 0 ? last_e2e / med : 1.0; p0nv = naive_p(0, Dl, s); us_nv += (now_ms() - tm) * 1e3;
            p0sc = flag ? p0nv : p0cc;
            tm = now_ms(); pm1 = toba_meet(1, 0); us_tb += (now_ms() - tm) * 1e3;
            double pm = policy == 1 ? 1 - p0cc : policy == 2 ? 1 - p0sc : policy == 3 ? 1 - p0nv : policy == 4 ? pm1 : 2;
            if (pm < thr[1]) { alarm = 0; degr = 1; }
        }
        /* ---- N1 ---- */
        double t = now_ms(); n1_load(path); d[0] = now_ms() - t;
        double ev[3]; ev[0] = d[0] > 1e-3 ? d[0] : 1e-3;
        if (mon) {   /* cp1: before N2 */
            double e1 = now_ms() - t0, pcc, pnv, pm;
            tm = now_ms(); post_w(prior, ev, 1, w); pcc = cc_p(w, 1, Dl - e1); us_cc += (now_ms() - tm) * 1e3;
            tm = now_ms(); pnv = naive_p(1, Dl - e1, s); us_nv += (now_ms() - tm) * 1e3;
            tm = now_ms(); double tb = toba_meet(2, e1); us_tb += (now_ms() - tm) * 1e3;
            pm = policy == 1 ? 1 - pcc : policy == 2 ? 1 - (flag ? pnv : pcc) : policy == 3 ? 1 - pnv : policy == 4 ? tb : 2;
            if (alarm < 0 && pm < thr[2]) { alarm = 1; degr = 1; }
        }
        if (degr == 1) decimate_points();
        t = now_ms(); n2_voxel(0.2f); d[1] = now_ms() - t; ev[1] = d[1] > 1e-3 ? d[1] : 1e-3;
        if (mon) {   /* cp2: before N3 || N4 */
            double e2 = now_ms() - t0, pcc, pnv, pm, th = thr[3] < thr[4] ? thr[3] : thr[4];
            tm = now_ms(); post_w(prior, ev, 2, w); pcc = cc_p(w, 2, Dl - e2); us_cc += (now_ms() - tm) * 1e3;
            tm = now_ms(); pnv = naive_p(2, Dl - e2, s); us_nv += (now_ms() - tm) * 1e3;
            tm = now_ms(); double tb = toba_meet(3, e2), tb4 = toba_meet(4, e2); us_tb += (now_ms() - tm) * 1e3;
            int hit = policy == 4 ? (tb < thr[3] || tb4 < thr[4]) : (policy ? (1 - (policy == 1 ? pcc : policy == 2 ? (flag ? pnv : pcc) : pnv)) < th : 0);
            (void)pm; if (alarm < 0 && hit) { alarm = 2; degr = 2; }
        }
        vstep = degr == 2 ? 2 : 1;
        pthread_t wt; pthread_create(&wt, NULL, n4_bev, &d[3]); d[2] = n3_ground(); pthread_join(wt, NULL); vstep = 1;
        double F = d[2] > d[3] ? d[2] : d[3]; ev[2] = F > 1e-3 ? F : 1e-3;
        if (mon) {   /* cp3: before N5 (alarm logged only) */
            double e3 = now_ms() - t0, pcc, pnv;
            tm = now_ms(); post_w(prior, ev, 3, w); pcc = cc_p(w, 3, Dl - e3); us_cc += (now_ms() - tm) * 1e3;
            tm = now_ms(); pnv = naive_p(3, Dl - e3, s); us_nv += (now_ms() - tm) * 1e3;
            tm = now_ms(); double tb = toba_meet(5, e3); us_tb += (now_ms() - tm) * 1e3;
            double pm = policy == 1 ? 1 - pcc : policy == 2 ? 1 - (flag ? pnv : pcc) : policy == 3 ? 1 - pnv : policy == 4 ? tb : 2;
            if (alarm < 0 && pm < thr[5]) alarm = 3;
        }
        t = now_ms(); int fc = n5_fuse(); d[4] = now_ms() - t;
        if (mon && policy == 4 && alarm < 0) { double e4 = now_ms() - t0; if (toba_meet(6, e4) < thr[6]) alarm = 4; }
        double e2e = now_ms() - t0, pit = NAN;
        /* ---- end of frame: update monitor state (counted as CC / NAIVE cost) ---- */
        if (mon) {
            tm = now_ms();
            int nev = degr == 1 ? 1 : degr == 2 ? 2 : 3;
            post_w(prior, ev, nev, post); have_post = 1;
            us_cc += (now_ms() - tm) * 1e3;
            if (!degr) {
                tm = now_ms(); pit = cc_p(prior, 0, e2e);   /* P_CC(e2e > observed) at release */
                for (int j = 19; j > 0; j--) pit_hist[j] = pit_hist[j - 1];
                pit_hist[0] = (pit < 0.05 || pit > 0.95); if (npit < 20) npit++;
                us_sc += (now_ms() - tm) * 1e3;
                last_e2e = e2e;
            }
        }
        fprintf(out, "%d,K%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%s,%d,%d,%.5f,%.5f,%.5f,%.5f,%d,%.5f,%.2f,%.2f,%.2f,%.2f,%.2f,%d\n",
                fr, k, val ? "val" : "calib", si, d[0], d[1], d[2], d[3], d[4], e2e, k, seed, pol, alarm, degr,
                p0cc, p0sc, p0nv, pm1, flag, pit, us_cc, us_sc, us_nv, us_tb, us_cc + us_sc + us_nv + us_tb, fc);
        next_rel = fmax(next_rel + period, now_ms());
        if (fr % 500 == 0) { fflush(out); printf("frame %d/%d K=%d e2e=%.2f ms\n", fr, nfr, k, e2e); fflush(stdout); }
    }
    set_hogs(0); fclose(out); return 0;
}
