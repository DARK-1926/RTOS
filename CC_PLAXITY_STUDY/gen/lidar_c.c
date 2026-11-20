/* lidar_c.c: compiled real-LiDAR pipeline for the CC-plaxity G8 study (PROTOCOL_GENERALIZATION.md, Addendum B).
 *
 * Same DAG and CSV schema as cc_general.py:  N1 -> N2 -> (N3 || N4) -> N5
 *   N1  read scan file (float32 x,y,z) + range/height crop
 *   N2  voxel downsample (0.2 m): pack voxel keys, sort, average per voxel
 *   N3  ground segmentation on a 1 m BEV grid (min-z, then obstacle-cell count)   [main thread]
 *   N4  0.5 m BEV max-height map + 72-sector nearest return                        [worker thread]
 *   N5  sector free-space fusion
 * Hidden contention: K memory-streaming hog THREADS (16 MB buffer each) x hog_mult, random-telegraph
 * schedule (dwell ~ Exp(mean 4 s)); K in {0,1,2} for calib frames, {0,1,2,3} for val frames.
 * No artificial jitter: latency variation comes from scene content and contention only.
 *
 * Build (board):  ~/tc/gcc -O2 -o lidar_c lidar_c.c -lm -lpthread
 * Build (laptop): gcc -O2 -o lidar_c.exe lidar_c.c -lm -lpthread
 * Run: lidar_c <scan_dir> <out.csv> <seed> <calib_frames> <val_frames> <period_ms> <hog_mult>
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
#define GRID1 100      /* 1 m cells over +-50 m   */
#define GRID05 200     /* 0.5 m cells over +-50 m */
#define NSEC 72
#define HOG_MAX 64

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static void sleep_ms(double ms) {
    struct timespec t; t.tv_sec = (time_t)(ms / 1e3); t.tv_nsec = (long)((ms - t.tv_sec * 1e3) * 1e6);
    nanosleep(&t, NULL);
}

/* ---------------- RNG (splitmix64) ---------------- */
static uint64_t rs;
static uint64_t rnd(void) { uint64_t z = (rs += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static double urand(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

/* ---------------- hogs ---------------- */
typedef struct { pthread_t th; volatile int stop; } hog_t;
static hog_t hogs[HOG_MAX];
static int nhog = 0;
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

/* ---------------- pipeline data ---------------- */
typedef struct { float x, y, z; } pt;
typedef struct { uint64_t key; float x, y, z; } kp;
static pt pts[PTS_MAX], vox[PTS_MAX];
static kp keyed[PTS_MAX];
static float rawbuf[PTS_MAX * 3];
static int npts, nvox;

static int n1_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    size_t n = fread(rawbuf, sizeof(float), PTS_MAX * 3, f); fclose(f);
    npts = 0;
    for (size_t i = 0; i + 2 < n; i += 3) {
        float x = rawbuf[i], y = rawbuf[i + 1], z = rawbuf[i + 2];
        if (z > -3.0f && z < 4.0f && x * x + y * y < 2500.0f) { pts[npts].x = x; pts[npts].y = y; pts[npts].z = z; npts++; }
    }
    return npts;
}
static int cmpk(const void *a, const void *b) {
    uint64_t ka = ((const kp *)a)->key, kb = ((const kp *)b)->key; return (ka > kb) - (ka < kb);
}
static void n2_voxel(float v) {
    for (int i = 0; i < npts; i++) {
        uint64_t ix = (uint64_t)(int64_t)floorf((pts[i].x + 64.f) / v), iy = (uint64_t)(int64_t)floorf((pts[i].y + 64.f) / v),
                 iz = (uint64_t)(int64_t)floorf((pts[i].z + 8.f) / v);
        keyed[i].key = (ix << 40) | (iy << 20) | iz; keyed[i].x = pts[i].x; keyed[i].y = pts[i].y; keyed[i].z = pts[i].z;
    }
    qsort(keyed, npts, sizeof(kp), cmpk);
    nvox = 0;
    for (int i = 0; i < npts;) {
        int j = i; double sx = 0, sy = 0, sz = 0;
        while (j < npts && keyed[j].key == keyed[i].key) { sx += keyed[j].x; sy += keyed[j].y; sz += keyed[j].z; j++; }
        vox[nvox].x = (float)(sx / (j - i)); vox[nvox].y = (float)(sy / (j - i)); vox[nvox].z = (float)(sz / (j - i)); nvox++;
        i = j;
    }
}
static int cell(float c, float res, int n) { int i = (int)floorf((c + 50.f) / res); return i < 0 ? 0 : (i >= n ? n - 1 : i); }

static unsigned char obst[GRID1 * GRID1];
static double n3_ground(void) {                 /* returns ms; fills obst */
    double t = now_ms();
    static float zmin[GRID1 * GRID1]; static int cnt[GRID1 * GRID1];
    for (int i = 0; i < GRID1 * GRID1; i++) { zmin[i] = 1e9f; cnt[i] = 0; }
    for (int i = 0; i < nvox; i++) { int c = cell(vox[i].x, 1.f, GRID1) * GRID1 + cell(vox[i].y, 1.f, GRID1);
                                    if (vox[i].z < zmin[c]) zmin[c] = vox[i].z; }
    for (int i = 0; i < nvox; i++) { int c = cell(vox[i].x, 1.f, GRID1) * GRID1 + cell(vox[i].y, 1.f, GRID1);
                                    if (vox[i].z - zmin[c] > 0.3f) cnt[c]++; }
    for (int i = 0; i < GRID1 * GRID1; i++) obst[i] = cnt[i] >= 2;
    return now_ms() - t;
}
static float nearsec[NSEC]; static int tall;
static void *n4_bev(void *out) {
    double t = now_ms();
    static float hmax[GRID05 * GRID05];
    for (int i = 0; i < GRID05 * GRID05; i++) hmax[i] = -1e9f;
    for (int s = 0; s < NSEC; s++) nearsec[s] = 99.f;
    for (int i = 0; i < nvox; i++) {
        float x = vox[i].x, y = vox[i].y, z = vox[i].z;
        int c = cell(x, .5f, GRID05) * GRID05 + cell(y, .5f, GRID05); if (z > hmax[c]) hmax[c] = z;
        float r = sqrtf(x * x + y * y); int s = (int)((atan2f(y, x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC;
        if (z > -1.2f && r < nearsec[s]) nearsec[s] = r;
    }
    tall = 0; for (int i = 0; i < GRID05 * GRID05; i++) tall += hmax[i] > 0.5f;
    *(double *)out = now_ms() - t; return NULL;
}
static int n5_fuse(void) {
    int occ[NSEC] = {0}, freec = 0;
    for (int cx = 0; cx < GRID1; cx++) for (int cy = 0; cy < GRID1; cy++) if (obst[cx * GRID1 + cy]) {
        float x = cx - 50.f + .5f, y = cy - 50.f + .5f;
        occ[(int)((atan2f(y, x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC]++;
    }
    for (int s = 0; s < NSEC; s++) freec += (occ[s] == 0 && nearsec[s] > 10.f);
    return freec;
}

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: lidar_c dir out seed calib val period_ms hog_mult\n"); return 2; }
    const char *dir = argv[1]; int seed = atoi(argv[3]), ncal = atoi(argv[4]), nval = atoi(argv[5]), mult = atoi(argv[7]);
    double period = atof(argv[6]);
    rs = 0x1234567ULL + (uint64_t)seed * 0x9E3779B97F4A7C15ULL;
    static char paths[NSCAN_MAX][512]; int nscan = 0;
    for (; nscan < NSCAN_MAX; nscan++) {
        snprintf(paths[nscan], sizeof paths[nscan], "%s/%04d.xyz", dir, nscan);
        FILE *f = fopen(paths[nscan], "rb"); if (!f) break; fclose(f);
    }
    if (!nscan) { fprintf(stderr, "no scans in %s\n", dir); return 2; }
    FILE *out = fopen(argv[2], "w");
    fprintf(out, "global_frame_id,regime,split,bag_frame_idx,dur_node1_ms,dur_node2_ms,dur_node3_ms,dur_node4_ms,"
                 "dur_node5_ms,end_to_end_ms,hog_k,seed\n");
    int total = ncal + nval, k = 0; double t_switch = now_ms(), next_rel = now_ms(); volatile int sink = 0;
    for (int fr = 0; fr < total; fr++) {
        int val = fr >= ncal;
        if (now_ms() >= t_switch) {
            int lv[4], nl = 0, top = val ? 3 : 2;
            for (int x = 0; x <= top; x++) if (x != k) lv[nl++] = x;
            k = lv[rnd() % nl]; set_hogs(k * mult);
            t_switch = now_ms() + (-log(1.0 - urand())) * 4000.0;
        }
        while (now_ms() < next_rel) sleep_ms(0.5);
        int idx = fr % nscan; double t0 = now_ms(), t, d1, d2, d3, d4 = 0, d5;
        t = now_ms(); n1_load(paths[idx]); d1 = now_ms() - t;
        t = now_ms(); n2_voxel(0.2f); d2 = now_ms() - t;
        pthread_t w; pthread_create(&w, NULL, n4_bev, &d4);
        d3 = n3_ground();
        pthread_join(w, NULL);
        t = now_ms(); sink += n5_fuse() + tall; d5 = now_ms() - t;
        double e2e = now_ms() - t0;
        fprintf(out, "%d,K%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d\n", fr, k, val ? "val" : "calib", idx,
                d1, d2, d3, d4, d5, e2e, k, seed);
        next_rel = fmax(next_rel + period, now_ms());
        if (fr % 500 == 0) { fflush(out); printf("frame %d/%d K=%d e2e=%.2f ms\n", fr, total, k, e2e); fflush(stdout); }
    }
    set_hogs(0); fclose(out); return sink == -1;
}
