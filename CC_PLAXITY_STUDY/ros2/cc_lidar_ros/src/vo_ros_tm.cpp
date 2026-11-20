// vo_ros_tm.cpp -- Addendum H Part 2: a drone visual-odometry front end as a real ROS 2 (rclcpp, Jazzy) application, with the
// SAME DAG shape, executor, telemetry sampler and CSV format as lidar_ros_tm.cpp (so the analysis code is unchanged).
// Data: EuRoC MAV cam0 (752x480 grayscale) from an AscTec Firefly hexacopter, raw 8-bit frames %04d.gray.
//   N1 camera  (timer, period P): load frame                                    -> /vo/image   (sensor_msgs/Image, mono8)
//   N2 pyramid (sub /vo/image):   5x5 Gaussian blur + 3-level pyramid            -> /vo/pyr     (sensor_msgs/Image, packed levels)
//   N3 fast    (sub /vo/pyr):     FAST-9 corners, score, 3x3 NMS, grid bucketing -> /vo/corners (Float32MultiArray)
//   N4 klt     (sub /vo/pyr):     pyramidal Lucas-Kanade tracking of the current feature set from the previous frame -> /vo/tracks
//   N5 pose    (joins corners + tracks on frame id): RANSAC homography (200 hypotheses, DLT) on the tracks, then feature
//              replenishment (inliers + new corners on a free 10 px grid, max 300), handed to N4 for the next frame.
// Latency is data dependent (texture -> corners, motion/blur -> tracks and iterations) and contention dependent (real interferers).
// Usage: vo_ros_tm <frame_dir> <idx_file> <out.csv> <nframes> <period_ms> <calib|val> <regime_file> <seed>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>

using Img = sensor_msgs::msg::Image; using FA = std_msgs::msg::Float32MultiArray;
static constexpr int W0 = 752, H0 = 480, NLEV = 3, MAXF = 300, WIN = 4, ITERS = 10, RANSAC_IT = 200;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rec { double t0 = 0, s[6] = {0}, e[6] = {0}; int k = -1, scan = -1, freec = -1, done = 0;
             long long st[8] = {0}; long long own = 0, rdelay = 0; int prun = 0; double us_tm = 0; };
// ---- telemetry sampler: identical to lidar_ros_tm.cpp (Round G v3)
struct Snap { long long st[8] = {0}; long long own = 0; int prun = 0; double t_ms = 0, us_cost = 0; };
static std::mutex SNAPMU; static Snap SNAP;
static void sample_once() {
  Rec rr; Rec &r = rr; double t = now_ms();
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  r.st[0] = ru.ru_nivcsw; r.st[1] = ru.ru_nvcsw;
  r.st[2] = ru.ru_utime.tv_sec * 1000000LL + ru.ru_utime.tv_usec; r.st[3] = ru.ru_stime.tv_sec * 1000000LL + ru.ru_stime.tv_usec;
  r.own = r.st[2] + r.st[3];
  static int lfd = open("/proc/loadavg", O_RDONLY);
  if (lfd >= 0) { char buf[128]; ssize_t n = pread(lfd, buf, sizeof buf - 1, 0);
    if (n > 0) { buf[n] = 0; double a, b, c; int run = 0, tot = 0; if (sscanf(buf, "%lf %lf %lf %d/%d", &a, &b, &c, &run, &tot) == 5) r.prun = run; } }
  static int sfd = open("/proc/stat", O_RDONLY);
  if (sfd >= 0) { static char sb[8192]; ssize_t n = pread(sfd, sb, sizeof sb - 1, 0);
    if (n > 0) { sb[n] = 0; long long u, ni, sy, id, io, irq, sirq, stl;
      if (sscanf(sb, "cpu %lld %lld %lld %lld %lld %lld %lld %lld", &u, &ni, &sy, &id, &io, &irq, &sirq, &stl) == 8) {
        r.st[4] = u + sy + irq + sirq; r.st[5] = ni; r.st[6] = id + io; r.st[7] = stl; } } }
  Snap s; for (int i = 0; i < 8; i++) s.st[i] = r.st[i]; s.own = r.own; s.prun = r.prun; s.t_ms = now_ms(); s.us_cost = (s.t_ms - t) * 1e3;
  std::lock_guard<std::mutex> g(SNAPMU); SNAP = s;
}
static void telemetry(Rec &r) {
  double t = now_ms(); Snap s; { std::lock_guard<std::mutex> g(SNAPMU); s = SNAP; }
  for (int i = 0; i < 8; i++) r.st[i] = s.st[i]; r.own = s.own; r.prun = s.prun; r.rdelay = (long long)((t - s.t_ms) * 1000);
  r.us_tm = (now_ms() - t) * 1e3;
}
static std::vector<Rec> REC;
static std::atomic<int> REGIME{0};
static std::atomic<int> NDONE{0};

// ---- image pyramid packed in one buffer: level l is (W0>>l) x (H0>>l), levels concatenated
static int lw(int l) { return W0 >> l; } static int lh(int l) { return H0 >> l; }
static size_t loff(int l) { size_t o = 0; for (int i = 0; i < l; i++) o += (size_t)lw(i) * lh(i); return o; }
static inline float bil(const uint8_t *im, int w, int h, float x, float y) {
  if (x < 0) x = 0; if (y < 0) y = 0; if (x > w - 1.001f) x = w - 1.001f; if (y > h - 1.001f) y = h - 1.001f;
  int xi = (int)x, yi = (int)y; float ax = x - xi, ay = y - yi; const uint8_t *p = im + yi * w + xi;
  return (1 - ay) * ((1 - ax) * p[0] + ax * p[1]) + ay * ((1 - ax) * p[w] + ax * p[w + 1]);
}
static void set_img(Img &m, int frame, const uint8_t *d, size_t n) {
  m.header.frame_id = std::to_string(frame); m.height = 1; m.width = n; m.encoding = "mono8"; m.step = n; m.data.assign(d, d + n);
}

class N1 : public rclcpp::Node {
 public:
  N1(std::string dir, std::vector<int> idx, int nfr, double period) : Node("vo_n1_camera"), dir_(dir), idx_(idx), nfr_(nfr) {
    pub_ = create_publisher<Img>("/vo/image", rclcpp::QoS(10).reliable()); buf_.resize(W0 * H0);
    timer_ = create_wall_timer(std::chrono::microseconds((long)(period * 1000)), [this] { tick(); });
  }
 private:
  void tick() {
    if (fr_ >= nfr_) { timer_->cancel(); return; }
    int f = fr_++; Rec &r = REC[f]; r.t0 = now_ms(); r.k = REGIME.load(); r.scan = idx_[f % idx_.size()];
    r.s[1] = 0; telemetry(r);
    char path[512]; snprintf(path, sizeof path, "%s/%04d.gray", dir_.c_str(), r.scan);
    FILE *fp = fopen(path, "rb"); size_t n = fp ? fread(buf_.data(), 1, W0 * H0, fp) : 0; if (fp) fclose(fp);
    if (n < (size_t)W0 * H0) std::fill(buf_.begin() + n, buf_.end(), 0);
    Img m; set_img(m, f, buf_.data(), buf_.size()); m.width = W0; m.height = H0; m.step = W0;
    r.e[1] = now_ms() - r.t0;
    pub_->publish(m);
  }
  std::string dir_; std::vector<int> idx_; int nfr_, fr_ = 0; std::vector<uint8_t> buf_;
  rclcpp::Publisher<Img>::SharedPtr pub_; rclcpp::TimerBase::SharedPtr timer_;
};

class N2 : public rclcpp::Node {
 public:
  N2() : Node("vo_n2_pyramid") {
    pub_ = create_publisher<Img>("/vo/pyr", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<Img>("/vo/image", rclcpp::QoS(10).reliable(), [this](Img::UniquePtr m) { cb(*m); });
  }
 private:
  void cb(const Img &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[2] = now_ms() - r.t0;
    std::vector<uint8_t> pyr(loff(NLEV)); std::vector<float> tmp(W0 * H0);
    const uint8_t *src = m.data.data(); const float k5[5] = {1 / 16.f, 4 / 16.f, 6 / 16.f, 4 / 16.f, 1 / 16.f};
    for (int y = 0; y < H0; y++) for (int x = 0; x < W0; x++) { float s = 0;
      for (int i = -2; i <= 2; i++) { int xx = std::min(std::max(x + i, 0), W0 - 1); s += k5[i + 2] * src[y * W0 + xx]; } tmp[y * W0 + x] = s; }
    uint8_t *L0 = pyr.data();
    for (int y = 0; y < H0; y++) for (int x = 0; x < W0; x++) { float s = 0;
      for (int i = -2; i <= 2; i++) { int yy = std::min(std::max(y + i, 0), H0 - 1); s += k5[i + 2] * tmp[yy * W0 + x]; }
      L0[y * W0 + x] = (uint8_t)std::min(255.f, s + 0.5f); }
    for (int l = 1; l < NLEV; l++) { const uint8_t *a = pyr.data() + loff(l - 1); uint8_t *b = pyr.data() + loff(l); int w = lw(l), h = lh(l), pw = lw(l - 1);
      for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
        b[y * w + x] = (uint8_t)((a[2 * y * pw + 2 * x] + a[2 * y * pw + 2 * x + 1] + a[(2 * y + 1) * pw + 2 * x] + a[(2 * y + 1) * pw + 2 * x + 1] + 2) / 4); }
    Img out; set_img(out, f, pyr.data(), pyr.size());
    r.e[2] = now_ms() - r.t0;
    pub_->publish(out);
  }
  rclcpp::Publisher<Img>::SharedPtr pub_; rclcpp::Subscription<Img>::SharedPtr sub_;
};

class N3 : public rclcpp::Node {   // FAST-9 on level 0, threshold 20
 public:
  N3() : Node("vo_n3_fast") {
    pub_ = create_publisher<FA>("/vo/corners", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<Img>("/vo/pyr", rclcpp::QoS(10).reliable(), [this](Img::UniquePtr m) { cb(*m); });
  }
 private:
  void cb(const Img &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[3] = now_ms() - r.t0;
    const uint8_t *im = m.data.data(); const int T = 20;
    static const int cx[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1}, cy[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
    std::vector<float> score(W0 * H0, 0.f);
    for (int y = 3; y < H0 - 3; y++) for (int x = 3; x < W0 - 3; x++) {
      int p = im[y * W0 + x], hi = p + T, lo = p - T, v[16], nb = 0, nd = 0;
      for (int i = 0; i < 16; i += 4) { int q = im[(y + cy[i]) * W0 + x + cx[i]]; nb += q > hi; nd += q < lo; }
      if (nb < 2 && nd < 2) continue;
      for (int i = 0; i < 16; i++) v[i] = im[(y + cy[i]) * W0 + x + cx[i]];
      int best = 0, run = 0, sgn = 0;
      for (int i = 0; i < 32; i++) { int q = v[i & 15], s = q > hi ? 1 : (q < lo ? -1 : 0);
        if (s != 0 && s == sgn) run++; else { run = s ? 1 : 0; sgn = s; } best = std::max(best, run); }
      if (best >= 9) { float sc = 0; for (int i = 0; i < 16; i++) sc += std::max(0, std::abs(v[i] - p) - T); score[y * W0 + x] = sc; }
    }
    struct C { float s; int x, y; }; std::vector<C> cs;
    for (int y = 4; y < H0 - 4; y++) for (int x = 4; x < W0 - 4; x++) { float s = score[y * W0 + x]; if (s <= 0) continue;
      bool mx = true; for (int dy = -1; dy <= 1 && mx; dy++) for (int dx = -1; dx <= 1; dx++) if ((dx || dy) && score[(y + dy) * W0 + x + dx] > s) { mx = false; break; }
      if (mx) cs.push_back({s, x, y}); }
    std::sort(cs.begin(), cs.end(), [](const C &a, const C &b) { return a.s > b.s; });
    const int GX = 16, GY = 10; std::vector<int> cnt(GX * GY, 0); FA o;
    for (auto &c : cs) { int g = (c.y * GY / H0) * GX + c.x * GX / W0; if (cnt[g] >= 4) continue; cnt[g]++;
      o.data.push_back(c.x); o.data.push_back(c.y); if ((int)o.data.size() >= 2 * 600) break; }
    o.data.push_back((float)f);
    r.e[3] = now_ms() - r.t0;
    pub_->publish(o);
  }
  rclcpp::Publisher<FA>::SharedPtr pub_; rclcpp::Subscription<Img>::SharedPtr sub_;
};

static std::mutex SEEDMU; static std::vector<float> SEED; static int SEED_FRAME = -1;   // N5 -> N4 feature hand-off

class N4 : public rclcpp::Node {   // pyramidal LK, 9x9 window, 3 levels, <= 10 iterations per level
 public:
  N4() : Node("vo_n4_klt") {
    pub_ = create_publisher<FA>("/vo/tracks", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<Img>("/vo/pyr", rclcpp::QoS(10).reliable(), [this](Img::UniquePtr m) { cb(*m); });
  }
 private:
  void cb(const Img &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[4] = now_ms() - r.t0;
    std::vector<float> pts; { std::lock_guard<std::mutex> g(SEEDMU); pts = SEED; }
    FA o;
    if (!prev_.empty() && !pts.empty()) {
      const uint8_t *P = prev_.data(), *Q = m.data.data();
      for (size_t i = 0; i + 1 < pts.size(); i += 2) {
        float px = pts[i], py = pts[i + 1], dx = 0, dy = 0; bool ok = true;
        for (int l = NLEV - 1; l >= 0 && ok; l--) {
          int w = lw(l), h = lh(l); const uint8_t *A = P + loff(l), *B = Q + loff(l); float sc = 1.f / (1 << l), x0 = px * sc, y0 = py * sc;
          if (l < NLEV - 1) { dx *= 2; dy *= 2; }
          float Ix[81], Iy[81], I0[81]; double gxx = 0, gxy = 0, gyy = 0; int n = 0;
          for (int v = -WIN; v <= WIN; v++) for (int u = -WIN; u <= WIN; u++, n++) {
            float x = x0 + u, y = y0 + v; I0[n] = bil(A, w, h, x, y);
            Ix[n] = 0.5f * (bil(A, w, h, x + 1, y) - bil(A, w, h, x - 1, y)); Iy[n] = 0.5f * (bil(A, w, h, x, y + 1) - bil(A, w, h, x, y - 1));
            gxx += Ix[n] * Ix[n]; gxy += Ix[n] * Iy[n]; gyy += Iy[n] * Iy[n]; }
          double det = gxx * gyy - gxy * gxy; if (det < 1e-3 * 81 * 81) { ok = false; break; }
          for (int it = 0; it < ITERS; it++) { double bx = 0, by = 0; n = 0;
            for (int v = -WIN; v <= WIN; v++) for (int u = -WIN; u <= WIN; u++, n++) {
              float e = I0[n] - bil(B, w, h, x0 + u + dx, y0 + v + dy); bx += e * Ix[n]; by += e * Iy[n]; }
            float ux = (float)((gyy * bx - gxy * by) / det), uy = (float)((gxx * by - gxy * bx) / det); dx += ux; dy += uy;
            if (ux * ux + uy * uy < 0.03f * 0.03f) break; }
        }
        float qx = px + dx, qy = py + dy;
        if (ok && qx >= 4 && qy >= 4 && qx < W0 - 4 && qy < H0 - 4) { o.data.push_back(px); o.data.push_back(py); o.data.push_back(qx); o.data.push_back(qy); }
      }
    }
    prev_.assign(m.data.begin(), m.data.end());
    o.data.push_back((float)f);
    r.e[4] = now_ms() - r.t0;
    pub_->publish(o);
  }
  std::vector<uint8_t> prev_;
  rclcpp::Publisher<FA>::SharedPtr pub_; rclcpp::Subscription<Img>::SharedPtr sub_;
};

static bool solve8(double A[8][9], double *h) {   // Gaussian elimination with partial pivoting
  for (int c = 0; c < 8; c++) { int p = c; for (int i = c + 1; i < 8; i++) if (std::fabs(A[i][c]) > std::fabs(A[p][c])) p = i;
    if (std::fabs(A[p][c]) < 1e-12) return false; if (p != c) for (int j = 0; j < 9; j++) std::swap(A[p][j], A[c][j]);
    for (int i = c + 1; i < 8; i++) { double k = A[i][c] / A[c][c]; for (int j = c; j < 9; j++) A[i][j] -= k * A[c][j]; } }
  for (int i = 7; i >= 0; i--) { double s = A[i][8]; for (int j = i + 1; j < 8; j++) s -= A[i][j] * h[j]; h[i] = s / A[i][i]; }
  return true;
}

class N5 : public rclcpp::Node {
 public:
  N5() : Node("vo_n5_pose"), rng_(12345) {
    a_ = create_subscription<FA>("/vo/corners", rclcpp::QoS(10).reliable(), [this](FA::UniquePtr m) { int f = (int)m->data.back(); m->data.pop_back(); cor_[f] = std::move(m->data); try_fuse(f); });
    b_ = create_subscription<FA>("/vo/tracks", rclcpp::QoS(10).reliable(), [this](FA::UniquePtr m) { int f = (int)m->data.back(); m->data.pop_back(); trk_[f] = std::move(m->data); try_fuse(f); });
  }
 private:
  void try_fuse(int f) {
    if (!cor_.count(f) || !trk_.count(f)) return;
    Rec &r = REC[f]; r.s[5] = now_ms() - r.t0;
    auto &cs = cor_[f]; auto &tr = trk_[f]; int nt = tr.size() / 4; std::vector<char> best_in(nt, 0); int best = 0;
    if (nt >= 8) { std::uniform_int_distribution<int> U(0, nt - 1);
      for (int it = 0; it < RANSAC_IT; it++) { int id[4];
        for (int k = 0; k < 4; k++) { bool dup; do { id[k] = U(rng_); dup = false; for (int j = 0; j < k; j++) dup |= id[j] == id[k]; } while (dup); }
        double A[8][9];
        for (int k = 0; k < 4; k++) { double x = tr[4 * id[k]], y = tr[4 * id[k] + 1], u = tr[4 * id[k] + 2], v = tr[4 * id[k] + 3];
          double r1[9] = {x, y, 1, 0, 0, 0, -u * x, -u * y, u}, r2[9] = {0, 0, 0, x, y, 1, -v * x, -v * y, v};
          std::memcpy(A[2 * k], r1, sizeof r1); std::memcpy(A[2 * k + 1], r2, sizeof r2); }
        double h[8]; if (!solve8(A, h)) continue; int inl = 0; std::vector<char> in(nt, 0);
        for (int i = 0; i < nt; i++) { double x = tr[4 * i], y = tr[4 * i + 1], w = h[6] * x + h[7] * y + 1; if (std::fabs(w) < 1e-9) continue;
          double u = (h[0] * x + h[1] * y + h[2]) / w - tr[4 * i + 2], v = (h[3] * x + h[4] * y + h[5]) / w - tr[4 * i + 3];
          if (u * u + v * v < 4.0) { in[i] = 1; inl++; } }
        if (inl > best) { best = inl; best_in.swap(in); } } }
    std::vector<float> seed; std::vector<char> occ(76 * 48, 0);   // 10 px occupancy grid
    for (int i = 0; i < nt && (int)seed.size() < 2 * MAXF; i++) if (best_in[i]) { float u = tr[4 * i + 2], v = tr[4 * i + 3];
      int g = (int)(v / 10) * 76 + (int)(u / 10); if (occ[g]) continue; occ[g] = 1; seed.push_back(u); seed.push_back(v); }
    for (size_t i = 0; i + 1 < cs.size() && (int)seed.size() < 2 * MAXF; i += 2) { int g = (int)(cs[i + 1] / 10) * 76 + (int)(cs[i] / 10);
      if (occ[g]) continue; occ[g] = 1; seed.push_back(cs[i]); seed.push_back(cs[i + 1]); }
    { std::lock_guard<std::mutex> g(SEEDMU); if (f > SEED_FRAME) { SEED.swap(seed); SEED_FRAME = f; } }
    r.freec = best; r.e[5] = now_ms() - r.t0; r.done = 1;
    cor_.erase(f); trk_.erase(f); NDONE++;
  }
  std::mt19937 rng_; std::map<int, std::vector<float>> cor_, trk_;
  rclcpp::Subscription<FA>::SharedPtr a_, b_;
};

int main(int argc, char **argv) {
  if (argc < 9) { fprintf(stderr, "usage: vo_ros_tm <frame_dir> <idx_file> <out.csv> <nframes> <period_ms> <calib|val> <regime_file> <seed>\n"); return 2; }
  std::string dir = argv[1], out = argv[3], split = argv[6], regf = argv[7]; int nfr = atoi(argv[4]), seed = atoi(argv[8]);
  double period = atof(argv[5]);
  std::vector<int> idx; { std::ifstream fi(argv[2]); int v; while (fi >> v) idx.push_back(v); }
  REC.assign(nfr, Rec());
  rclcpp::init(argc, argv);
  std::atomic<bool> stop{false};
  std::thread tmth([&] { while (!stop) { sample_once(); std::this_thread::sleep_for(std::chrono::milliseconds(20)); } });
  sample_once();
  std::thread rt([&] { while (!stop) { std::ifstream f(regf); int k; if (f >> k) REGIME = k; std::this_thread::sleep_for(std::chrono::milliseconds(20)); } });
  auto n1 = std::make_shared<N1>(dir, idx, nfr, period); auto n2 = std::make_shared<N2>(); auto n3 = std::make_shared<N3>();
  auto n4 = std::make_shared<N4>(); auto n5 = std::make_shared<N5>();
  rclcpp::executors::MultiThreadedExecutor ex(rclcpp::ExecutorOptions(), 4);
  for (auto n : std::vector<rclcpp::Node::SharedPtr>{n1, n2, n3, n4, n5}) ex.add_node(n);
  std::thread spin([&] { ex.spin(); });
  double tstart = now_ms(); int last = -1;
  while (rclcpp::ok()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    int d = NDONE.load();
    if (d / 500 != last / 500) { printf("done %d/%d K=%d\n", d, nfr, REGIME.load()); fflush(stdout); last = d; }
    if (d >= nfr || now_ms() - tstart > nfr * period * 3 + 60000) break;
  }
  ex.cancel(); spin.join(); stop = true; rt.join(); tmth.join();
  FILE *fo = fopen(out.c_str(), "w");
  fprintf(fo, "frame,regime,split,scan,seed,done,s1,e1,s2,e2,s3,e3,s4,e4,s5,e5,freec,"
              "nivcsw,nvcsw,cpu_user_us,cpu_sys_us,stat_busy,stat_nice,stat_idle,stat_steal,own_cpu_us,snap_age_us,procs_running,us_tm\n");
  for (int f = 0; f < nfr; f++) { Rec &r = REC[f];
    fprintf(fo, "%d,K%d,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%.2f\n",
            f, r.k, split.c_str(), r.scan, seed, r.done, r.s[1], r.e[1], r.s[2], r.e[2], r.s[3], r.e[3], r.s[4], r.e[4], r.s[5], r.e[5],
            r.freec, r.st[0], r.st[1], r.st[2], r.st[3], r.st[4], r.st[5], r.st[6], r.st[7], r.own, r.rdelay, r.prun, r.us_tm); }
  fclose(fo);
  rclcpp::shutdown();
  return 0;
}
