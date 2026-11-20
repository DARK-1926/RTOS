// lidar_ros_cl.cpp -- G12 (Addendum F): the G11 ROS 2 LiDAR DAG with INLINE monitors and closed-loop degradation.
//
// Same nodes, kernels, topics, QoS and executor as lidar_ros.cpp (G11). Added:
//  * KP-64 (Kalman plaxity, 64 quantiles) evaluated inline at all four checkpoints of every frame, under every policy
//    (so every policy pays the same monitor cost): cp0 release (N1 timer start), cp1 N2 start, cp2 first fork-branch
//    start, cp3 N5 start. Node mapping as in the G11 conversion: n1 = s2, n2 = fork - s2, F = max(e3,e4) - fork.
//    The cross-frame state is updated at the end of each UNDEGRADED frame (degraded frames are faster by design and
//    would bias the state; same rule as lidar_cl.c).
//  * ACI on the actionable target: alarm if max(P0, P1) > theta; when a frame ends,
//    theta <- theta - gamma * (1{miss and no actionable alarm} - alpha).
//  * Toba-eq meet probabilities at the start of nodes 1 and 2 (plaxity CDFs + job-count thresholds from the export).
//  * Policies: none (log only), kp (KP+ACI alarm -> degrade), toba (Toba-eq alarm at node 1 or 2 -> degrade),
//    random (degrade at release with probability p, seeded).
//    Degradation = keep every 2nd point (at N1 if the alarm is at release, else at N2 start before voxelization).
//  * Per-frame KP cost (all four checkpoints + end-of-frame update), Toba cost, in microseconds.
// Usage: lidar_ros_cl <scan_dir> <idx_file> <out.csv> <nframes> <period_ms> <split> <regime_file> <seed> <params> <policy> <p_random>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
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

using PC2 = sensor_msgs::msg::PointCloud2;
static constexpr int PTS_MAX = 40000, GRID1 = 100, GRID05 = 200, NSEC = 72, NN = 5;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------- monitor parameters ----------------
static double LAM, Q, Q0, ME2E, DL, THETA0, ALPHA, GAMMA, MED[NN], BB[NN], VV[NN], THR[7];
static int NQ; static std::vector<double> LOGB[4]; static std::vector<double> CDF[7];
static bool load_params(const char *fn) {
  std::ifstream f(fn); if (!f) return false;
  f >> LAM >> Q >> Q0 >> ME2E >> DL >> THETA0 >> ALPHA >> GAMMA;
  for (auto *a : {MED, BB, VV}) for (int j = 0; j < NN; j++) f >> a[j];
  f >> NQ; for (int k = 0; k < 4; k++) { LOGB[k].resize(NQ); for (int i = 0; i < NQ; i++) { double b; f >> b; LOGB[k][i] = std::log(b); } }
  for (int v = 1; v <= 6; v++) f >> THR[v];
  for (int v = 1; v <= 6; v++) { int n; f >> n; CDF[v].resize(n); for (int i = 0; i < n; i++) f >> CDF[v][i]; }
  return (bool)f;
}
// Upper normal tail Q(x) = 1 - Phi(x) from a 16385-point table on [-8, 8], linear interpolation (max error ~1e-7).
static constexpr int TN = 16384; static constexpr double TLO = -8.0, TSTEP = 16.0 / TN; static float QT[TN + 1];
static void init_qt() { for (int i = 0; i <= TN; i++) QT[i] = (float)(0.5 * std::erfc((TLO + i * TSTEP) / 1.41421356237309504880)); }
static inline double qtail(double x) {
  double u = (x - TLO) / TSTEP; if (u <= 0) return 1.0; if (u >= TN) return 0.0;
  int i = (int)u; double w = u - i; return QT[i] + w * (QT[i + 1] - QT[i]);
}
static double kp_p(int k, double mu, double sd, double slack) {
  if (slack <= 0) return 1.0;
  double ls = std::log(slack) - mu, is = 1.0 / sd, acc = 0; const double *lb = LOGB[k].data();
  for (int i = 0; i < NQ; i++) acc += qtail((ls - lb[i]) * is);
  return acc / NQ;
}
static double toba_meet(int v, double ast) { long b = std::lround((ast > 0 ? ast : 0) * 10.0); return b < (long)CDF[v].size() ? CDF[v][b] : 0.0; }

// shared monitor state (frames rarely overlap at a 100 ms period; a mutex keeps it exact when they do)
static std::mutex MU; static double CUR = 0; static int HAVE = 0; static double THETA = 0;
static std::string POLICY; static double PRAND = 0; static std::mt19937_64 RNG;

struct Rec {
  double t0 = 0, s[6] = {0}, e[6] = {0}; int k = -1, scan = -1, freec = -1, done = 0;
  double prec = 0, num = 0, p[4] = {NAN, NAN, NAN, NAN}, tb[3] = {NAN, NAN, NAN}, theta = NAN, us_kp = 0, us_tb = 0;
  int alarm = -1, degr = 0, npts = 0; std::atomic<int> cp2{0};
};
static std::vector<Rec> REC;
static std::atomic<int> REGIME{0}, NDONE{0};

static void kp_obs(Rec &r, int j, double x) {        // in-frame Gaussian update with node j's observed duration
  double xx = std::log((x > 1e-3 ? x : 1e-3) / MED[j]) - BB[j]; r.prec += 1.0 / VV[j]; r.num += xx / VV[j];
}
static void kp_check(Rec &r, int k, double elapsed) { r.p[k] = kp_p(k, r.num / r.prec, std::sqrt(1.0 / r.prec), DL - elapsed); }

struct P3 { float x, y, z; };
static void to_msg(const std::vector<P3> &v, int frame, PC2 &m) {
  m.header.frame_id = std::to_string(frame); m.height = 1; m.width = v.size(); m.is_dense = true; m.is_bigendian = false;
  m.fields.resize(3); const char *nm[3] = {"x", "y", "z"};
  for (int i = 0; i < 3; i++) { m.fields[i].name = nm[i]; m.fields[i].offset = 4 * i; m.fields[i].datatype = 7; m.fields[i].count = 1; }
  m.point_step = 12; m.row_step = 12 * v.size(); m.data.resize(m.row_step);
  if (!v.empty()) std::memcpy(m.data.data(), v.data(), m.row_step);
}
static void from_msg(const PC2 &m, std::vector<P3> &v) { v.resize(m.width); if (m.width) std::memcpy(v.data(), m.data.data(), 12 * m.width); }
static int cell(float c, float res, int n) { int i = (int)std::floor((c + 50.f) / res); return i < 0 ? 0 : (i >= n ? n - 1 : i); }
static void halve(std::vector<P3> &p) { size_t j = 0; for (size_t i = 0; i < p.size(); i += 2) p[j++] = p[i]; p.resize(j); }

class N1 : public rclcpp::Node {
 public:
  N1(std::string dir, std::vector<int> idx, int nfr, double period) : Node("cc_n1_sensor"), dir_(dir), idx_(idx), nfr_(nfr) {
    pub_ = create_publisher<PC2>("/cc/points", rclcpp::QoS(10).reliable());
    raw_.resize(PTS_MAX * 3);
    timer_ = create_wall_timer(std::chrono::microseconds((long)(period * 1000)), [this] { tick(); });
  }
 private:
  void tick() {
    if (fr_ >= nfr_) { timer_->cancel(); return; }
    int f = fr_++; Rec &r = REC[f]; r.t0 = now_ms(); r.k = REGIME.load(); r.scan = idx_[f % idx_.size()];
    { // ---- cp0: release ----
      std::lock_guard<std::mutex> g(MU);
      double t = now_ms();
      double m = HAVE ? CUR : 0.0, q = HAVE ? Q : Q0; r.prec = 1.0 / q; r.num = m / q; kp_check(r, 0, 0.0);
      r.us_kp += (now_ms() - t) * 1e3;
      t = now_ms(); r.tb[0] = toba_meet(1, 0.0); r.us_tb += (now_ms() - t) * 1e3;
      r.theta = THETA;
      bool hit = POLICY == "kp" ? r.p[0] > THETA : POLICY == "toba" ? r.tb[0] < THR[1] :
                 POLICY == "random" ? std::uniform_real_distribution<double>(0, 1)(RNG) < PRAND : false;
      if (POLICY == "kp" && r.p[0] > THETA) r.alarm = 0;          // logged alarm (actionable) for the kp policy
      if (POLICY != "kp" && POLICY != "none" && hit) r.alarm = 0;
      if (POLICY == "none" && r.p[0] > THETA) r.alarm = 0;        // what KP+ACI WOULD do (no action)
      if (hit && POLICY != "none") r.degr = 1;
    }
    char path[512]; snprintf(path, sizeof path, "%s/%04d.xyz", dir_.c_str(), r.scan);
    std::vector<P3> pts; pts.reserve(PTS_MAX);
    FILE *fp = fopen(path, "rb");
    size_t n = fp ? fread(raw_.data(), sizeof(float), PTS_MAX * 3, fp) : 0; if (fp) fclose(fp);
    for (size_t i = 0; i + 2 < n; i += 3) { float x = raw_[i], y = raw_[i + 1], z = raw_[i + 2];
      if (z > -3.0f && z < 4.0f && x * x + y * y < 2500.0f) pts.push_back({x, y, z}); }
    if (r.degr) halve(pts);
    r.npts = pts.size();
    PC2 m; to_msg(pts, f, m);
    r.e[1] = now_ms() - r.t0;
    pub_->publish(m);
  }
  std::string dir_; std::vector<int> idx_; int nfr_, fr_ = 0; std::vector<float> raw_;
  rclcpp::Publisher<PC2>::SharedPtr pub_; rclcpp::TimerBase::SharedPtr timer_;
};

class N2 : public rclcpp::Node {
 public:
  N2() : Node("cc_n2_voxel") {
    pub_ = create_publisher<PC2>("/cc/voxels", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<PC2>("/cc/points", rclcpp::QoS(10).reliable(), [this](PC2::UniquePtr m) { cb(*m); });
  }
 private:
  struct KP { uint64_t key; float x, y, z; };
  void cb(const PC2 &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[2] = now_ms() - r.t0;
    std::vector<P3> pts; from_msg(m, pts);
    { // ---- cp1: N2 start (actionable) ----
      std::lock_guard<std::mutex> g(MU);
      double t = now_ms(); kp_obs(r, 0, r.s[2]); kp_check(r, 1, r.s[2]); r.us_kp += (now_ms() - t) * 1e3;
      t = now_ms(); r.tb[1] = toba_meet(2, r.s[2]); r.us_tb += (now_ms() - t) * 1e3;
      if (r.alarm < 0) {
        bool kh = r.p[1] > r.theta, th = r.tb[1] < THR[2];
        if ((POLICY == "kp" || POLICY == "none") && kh) r.alarm = 1;
        if (POLICY == "toba" && th) r.alarm = 1;
        if (r.alarm == 1 && (POLICY == "kp" || POLICY == "toba") && !r.degr) { r.degr = 2; halve(pts); }
      }
    }
    const float v = 0.2f;
    std::vector<KP> keyed(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
      uint64_t ix = (uint64_t)(int64_t)std::floor((pts[i].x + 64.f) / v), iy = (uint64_t)(int64_t)std::floor((pts[i].y + 64.f) / v),
               iz = (uint64_t)(int64_t)std::floor((pts[i].z + 8.f) / v);
      keyed[i] = {(ix << 40) | (iy << 20) | iz, pts[i].x, pts[i].y, pts[i].z}; }
    std::qsort(keyed.data(), keyed.size(), sizeof(KP), [](const void *a, const void *b) {
      uint64_t ka = ((const KP *)a)->key, kb = ((const KP *)b)->key; return (ka > kb) - (ka < kb); });
    std::vector<P3> vox; vox.reserve(keyed.size());
    for (size_t i = 0; i < keyed.size();) { size_t j = i; double sx = 0, sy = 0, sz = 0;
      while (j < keyed.size() && keyed[j].key == keyed[i].key) { sx += keyed[j].x; sy += keyed[j].y; sz += keyed[j].z; j++; }
      vox.push_back({(float)(sx / (j - i)), (float)(sy / (j - i)), (float)(sz / (j - i))}); i = j; }
    PC2 out; to_msg(vox, f, out);
    r.e[2] = now_ms() - r.t0;
    pub_->publish(out);
  }
  rclcpp::Publisher<PC2>::SharedPtr pub_; rclcpp::Subscription<PC2>::SharedPtr sub_;
};

static void fork_check(Rec &r, double s) {   // ---- cp2: first fork branch to start ----
  int z = 0; if (!r.cp2.compare_exchange_strong(z, 1)) return;
  std::lock_guard<std::mutex> g(MU);
  double t = now_ms(); kp_obs(r, 1, s - r.s[2]); kp_check(r, 2, s); r.us_kp += (now_ms() - t) * 1e3;
}

class N3 : public rclcpp::Node {
 public:
  N3() : Node("cc_n3_ground") {
    pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>("/cc/obst", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<PC2>("/cc/voxels", rclcpp::QoS(10).reliable(), [this](PC2::UniquePtr m) { cb(*m); });
  }
 private:
  void cb(const PC2 &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[3] = now_ms() - r.t0; fork_check(r, r.s[3]);
    std::vector<P3> vox; from_msg(m, vox);
    std::vector<float> zmin(GRID1 * GRID1, 1e9f); std::vector<int> cnt(GRID1 * GRID1, 0);
    for (auto &p : vox) { int c = cell(p.x, 1.f, GRID1) * GRID1 + cell(p.y, 1.f, GRID1); if (p.z < zmin[c]) zmin[c] = p.z; }
    for (auto &p : vox) { int c = cell(p.x, 1.f, GRID1) * GRID1 + cell(p.y, 1.f, GRID1); if (p.z - zmin[c] > 0.3f) cnt[c]++; }
    std_msgs::msg::UInt8MultiArray o; o.data.resize(GRID1 * GRID1 + 4);
    for (int i = 0; i < GRID1 * GRID1; i++) o.data[i] = cnt[i] >= 2;
    std::memcpy(&o.data[GRID1 * GRID1], &f, 4);
    r.e[3] = now_ms() - r.t0;
    pub_->publish(o);
  }
  rclcpp::Publisher<std_msgs::msg::UInt8MultiArray>::SharedPtr pub_; rclcpp::Subscription<PC2>::SharedPtr sub_;
};

class N4 : public rclcpp::Node {
 public:
  N4() : Node("cc_n4_bev") {
    pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/cc/nearsec", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<PC2>("/cc/voxels", rclcpp::QoS(10).reliable(), [this](PC2::UniquePtr m) { cb(*m); });
  }
 private:
  void cb(const PC2 &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[4] = now_ms() - r.t0; fork_check(r, r.s[4]);
    std::vector<P3> vox; from_msg(m, vox);
    std::vector<float> hmax(GRID05 * GRID05, -1e9f); std_msgs::msg::Float32MultiArray o; o.data.assign(NSEC + 2, 99.f);
    for (auto &p : vox) { int c = cell(p.x, .5f, GRID05) * GRID05 + cell(p.y, .5f, GRID05); if (p.z > hmax[c]) hmax[c] = p.z;
      float rr = std::sqrt(p.x * p.x + p.y * p.y); int s = (int)((std::atan2(p.y, p.x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC;
      if (p.z > -1.2f && rr < o.data[s]) o.data[s] = rr; }
    int tall = 0; for (float h : hmax) tall += h > 0.5f;
    o.data[NSEC] = (float)tall; o.data[NSEC + 1] = (float)f;
    r.e[4] = now_ms() - r.t0;
    pub_->publish(o);
  }
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_; rclcpp::Subscription<PC2>::SharedPtr sub_;
};

class N5 : public rclcpp::Node {
 public:
  N5() : Node("cc_n5_fuse") {
    a_ = create_subscription<std_msgs::msg::UInt8MultiArray>("/cc/obst", rclcpp::QoS(10).reliable(),
        [this](std_msgs::msg::UInt8MultiArray::UniquePtr m) { int f; std::memcpy(&f, &m->data[GRID1 * GRID1], 4);
          obst_[f] = std::move(m->data); try_fuse(f); });
    b_ = create_subscription<std_msgs::msg::Float32MultiArray>("/cc/nearsec", rclcpp::QoS(10).reliable(),
        [this](std_msgs::msg::Float32MultiArray::UniquePtr m) { int f = (int)m->data[NSEC + 1]; near_[f] = std::move(m->data); try_fuse(f); });
  }
 private:
  void try_fuse(int f) {
    if (!obst_.count(f) || !near_.count(f)) return;
    Rec &r = REC[f]; r.s[5] = now_ms() - r.t0;
    { // ---- cp3: N5 start ----
      std::lock_guard<std::mutex> g(MU);
      double fork = std::min(r.s[3], r.s[4]), join = std::max(r.e[3], r.e[4]);
      double t = now_ms(); kp_obs(r, 2, join - fork); kp_check(r, 3, join); r.us_kp += (now_ms() - t) * 1e3;
    }
    auto &ob = obst_[f]; auto &ns = near_[f]; int occ[NSEC] = {0}, freec = 0;
    for (int cx = 0; cx < GRID1; cx++) for (int cy = 0; cy < GRID1; cy++) if (ob[cx * GRID1 + cy]) {
      float x = cx - 50.f + .5f, y = cy - 50.f + .5f; occ[(int)((std::atan2(y, x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC]++; }
    for (int s = 0; s < NSEC; s++) if (occ[s] == 0 && ns[s] > 10.f) freec++;
    r.freec = freec; r.e[5] = now_ms() - r.t0;
    { // ---- end of frame: KP state (undegraded frames only) and ACI update ----
      std::lock_guard<std::mutex> g(MU);
      double t = now_ms();
      if (!r.degr) { double z = std::log(r.e[5] / ME2E); CUR = HAVE ? LAM * CUR + (1 - LAM) * z : z; HAVE = 1; }
      double err = (r.e[5] > DL && r.alarm < 0) ? 1.0 : 0.0;   // unwarned actionable miss (alarm < 0: none at cp0/cp1)
      THETA -= GAMMA * (err - ALPHA);
      r.us_kp += (now_ms() - t) * 1e3;
    }
    r.done = 1; obst_.erase(f); near_.erase(f); NDONE++;
  }
  std::map<int, std::vector<uint8_t>> obst_; std::map<int, std::vector<float>> near_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr a_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr b_;
};

int main(int argc, char **argv) {
  if (argc < 12) { fprintf(stderr, "usage: lidar_ros_cl <scan_dir> <idx> <out.csv> <nframes> <period_ms> <split> <regime_file> <seed> <params> <policy> <p_random>\n"); return 2; }
  std::string dir = argv[1], out = argv[3], split = argv[6], regf = argv[7]; int nfr = atoi(argv[4]), seed = atoi(argv[8]);
  double period = atof(argv[5]); POLICY = argv[10]; PRAND = atof(argv[11]);
  if (!load_params(argv[9])) { fprintf(stderr, "bad params\n"); return 3; }
  init_qt(); THETA = THETA0; RNG.seed(0xC0FFEEULL + seed);
  std::vector<int> idx; { std::ifstream fi(argv[2]); int v; while (fi >> v) idx.push_back(v); }
  REC = std::vector<Rec>(nfr);
  rclcpp::init(argc, argv);
  std::atomic<bool> stop{false};
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
    if (d / 500 != last / 500) { printf("done %d/%d K=%d theta=%.4f\n", d, nfr, REGIME.load(), THETA); fflush(stdout); last = d; }
    if (d >= nfr || now_ms() - tstart > nfr * period * 3 + 60000) break;
  }
  ex.cancel(); spin.join(); stop = true; rt.join();
  FILE *fo = fopen(out.c_str(), "w");
  fprintf(fo, "frame,regime,split,scan,seed,policy,done,s1,e1,s2,e2,s3,e3,s4,e4,s5,e5,freec,npts,alarm,degraded,"
              "p0,p1,p2,p3,theta,tb1,tb2,us_kp,us_tb\n");
  for (int f = 0; f < nfr; f++) { Rec &r = REC[f];
    fprintf(fo, "%d,K%d,%s,%d,%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f\n",
            f, r.k, split.c_str(), r.scan, seed, POLICY.c_str(), r.done, r.s[1], r.e[1], r.s[2], r.e[2], r.s[3], r.e[3], r.s[4], r.e[4],
            r.s[5], r.e[5], r.freec, r.npts, r.alarm, r.degr, r.p[0], r.p[1], r.p[2], r.p[3], r.theta, r.tb[0], r.tb[1], r.us_kp, r.us_tb); }
  fclose(fo);
  rclcpp::shutdown();
  return 0;
}
