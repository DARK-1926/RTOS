// lidar_ros_tm.cpp -- G11 pipeline + OS contention telemetry logged at every release (Round G).
// Derived from lidar_ros.cpp -- the G8/G10 LiDAR DAG as a real ROS 2 (rclcpp, Jazzy) application (Addendum E, G11).
//
//   N1 sensor  (timer, period P):   load nuScenes scan, crop           -> /cc/points   (sensor_msgs/PointCloud2)
//   N2 voxel   (sub /cc/points):    0.2 m voxel grid                   -> /cc/voxels   (PointCloud2)
//   N3 ground  (sub /cc/voxels):    1 m min-z grid, obstacle cells     -> /cc/obst     (std_msgs/UInt8MultiArray)
//   N4 bev     (sub /cc/voxels):    0.5 m height map, near-range/sector-> /cc/nearsec  (std_msgs/Float32MultiArray)
//   N5 fuse    (sub obst+nearsec):  free-sector fusion (joins on frame id)
// The compute kernels are byte-for-byte the ones in lidar_cl.c.  All five nodes live in one process on a
// MultiThreadedExecutor (4 threads), the usual Autoware component-container layout; messages go through rmw
// (intra-process comms OFF), so serialization, DDS and executor scheduling are part of every measured latency.
// Contention is NOT synthetic: it comes from K concurrent instances of the ROS 2 Autoware reference system
// (ros-realtime/reference-system), started and stopped by run_g11.sh, which writes the current K to a file.
//
// Logged per frame (ms relative to the release t0 = start of N1's timer callback): start/end of every node.
// Usage: lidar_ros <scan_dir> <idx_file> <out.csv> <nframes> <period_ms> <calib|val> <regime_file> <seed>
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
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

using PC2 = sensor_msgs::msg::PointCloud2;
static constexpr int PTS_MAX = 40000, GRID1 = 100, GRID05 = 200, NSEC = 72;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Rec { double t0 = 0, s[6] = {0}, e[6] = {0}; int k = -1, scan = -1, freec = -1, done = 0;
             long long st[8] = {0}; long long own = 0, rdelay = 0; int prun = 0; double us_tm = 0; };
// ---- OS contention telemetry v2 (no root, cheap): st[0] = involuntary context switches of this process (getrusage),
//      st[1] = voluntary context switches, st[2..3] = user/system CPU us of this process, prun = runnable tasks (/proc/loadavg),
//      rdelay = summed run-queue delay (ns) of our threads (/proc/self/task/<tid>/schedstat, tid list cached, refreshed every 200 frames).
#include <dirent.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <unistd.h>
static std::vector<std::string> TASKS; static int TASK_AGE = 1 << 30;
// Telemetry sampler (Round G v3): a background thread samples the counters every 20 ms into a seqlock-free snapshot
// guarded by a mutex; the release path only copies the latest snapshot (critical-path cost ~1 us instead of ~0.5 ms).
struct Snap { long long st[8] = {0}; long long own = 0; int prun = 0; double t_ms = 0, us_cost = 0; };
static std::mutex SNAPMU; static Snap SNAP;
static void sample_once() {
  Rec rr; Rec &r = rr; double t = now_ms();
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  r.st[0] = ru.ru_nivcsw; r.st[1] = ru.ru_nvcsw;
  r.st[2] = ru.ru_utime.tv_sec * 1000000LL + ru.ru_utime.tv_usec; r.st[3] = ru.ru_stime.tv_sec * 1000000LL + ru.ru_stime.tv_usec;
  r.own = r.st[2] + r.st[3];
  static int lfd = open("/proc/loadavg", O_RDONLY);   // opened once; pread at offset 0 regenerates the content
  if (lfd >= 0) { char buf[128]; ssize_t n = pread(lfd, buf, sizeof buf - 1, 0);
    if (n > 0) { buf[n] = 0; double a, b, c; int run = 0, tot = 0; if (sscanf(buf, "%lf %lf %lf %d/%d", &a, &b, &c, &run, &tot) == 5) r.prun = run; } }
  // (per-thread schedstat dropped: ~70 us per /proc file x ~20 ROS threads = ~1.4 ms per frame; rdelay stays 0)
  // /proc/stat cpu line (USER_HZ ticks) via a persistent fd -> st[4] = user+system+irq+softirq (busy, all processes),
  // st[5] = nice (the SCHED_IDLE+nice19 spinners), st[6] = idle+iowait, st[7] = steal. Measured cost ~77 us.
  static int sfd = open("/proc/stat", O_RDONLY);
  if (sfd >= 0) { static char sb[8192]; ssize_t n = pread(sfd, sb, sizeof sb - 1, 0);
    if (n > 0) { sb[n] = 0; long long u, ni, sy, id, io, irq, sirq, stl;
      if (sscanf(sb, "cpu %lld %lld %lld %lld %lld %lld %lld %lld", &u, &ni, &sy, &id, &io, &irq, &sirq, &stl) == 8) {
        r.st[4] = u + sy + irq + sirq; r.st[5] = ni; r.st[6] = id + io; r.st[7] = stl; } } }

  Snap s; for (int i = 0; i < 8; i++) s.st[i] = r.st[i]; s.own = r.own; s.prun = r.prun; s.t_ms = now_ms(); s.us_cost = (s.t_ms - t) * 1e3;
  std::lock_guard<std::mutex> g(SNAPMU); SNAP = s;
}
static void telemetry(Rec &r) {   // critical path: copy the latest snapshot
  double t = now_ms(); Snap s; { std::lock_guard<std::mutex> g(SNAPMU); s = SNAP; }
  for (int i = 0; i < 8; i++) r.st[i] = s.st[i]; r.own = s.own; r.prun = s.prun; r.rdelay = (long long)((t - s.t_ms) * 1000);  // rdelay_ns column now = snapshot age (us)
  r.us_tm = (now_ms() - t) * 1e3;
}
static std::vector<Rec> REC;
static std::atomic<int> REGIME{0};
static std::atomic<int> NDONE{0};

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
    r.s[1] = 0; telemetry(r);
    char path[512]; snprintf(path, sizeof path, "%s/%04d.xyz", dir_.c_str(), r.scan);
    std::vector<P3> pts; pts.reserve(PTS_MAX);
    FILE *fp = fopen(path, "rb");
    size_t n = fp ? fread(raw_.data(), sizeof(float), PTS_MAX * 3, fp) : 0; if (fp) fclose(fp);
    for (size_t i = 0; i + 2 < n; i += 3) { float x = raw_[i], y = raw_[i + 1], z = raw_[i + 2];
      if (z > -3.0f && z < 4.0f && x * x + y * y < 2500.0f) pts.push_back({x, y, z}); }
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
    std::vector<P3> pts; from_msg(m, pts); const float v = 0.2f;
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

class N3 : public rclcpp::Node {
 public:
  N3() : Node("cc_n3_ground") {
    pub_ = create_publisher<std_msgs::msg::UInt8MultiArray>("/cc/obst", rclcpp::QoS(10).reliable());
    sub_ = create_subscription<PC2>("/cc/voxels", rclcpp::QoS(10).reliable(), [this](PC2::UniquePtr m) { cb(*m); });
  }
 private:
  void cb(const PC2 &m) {
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[3] = now_ms() - r.t0;
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
    int f = std::stoi(m.header.frame_id); Rec &r = REC[f]; r.s[4] = now_ms() - r.t0;
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
    auto &ob = obst_[f]; auto &ns = near_[f]; int occ[NSEC] = {0}, freec = 0;
    for (int cx = 0; cx < GRID1; cx++) for (int cy = 0; cy < GRID1; cy++) if (ob[cx * GRID1 + cy]) {
      float x = cx - 50.f + .5f, y = cy - 50.f + .5f; occ[(int)((std::atan2(y, x) + (float)M_PI) / (2.f * (float)M_PI) * NSEC) % NSEC]++; }
    for (int s = 0; s < NSEC; s++) if (occ[s] == 0 && ns[s] > 10.f) freec++;
    r.freec = freec; r.e[5] = now_ms() - r.t0; r.done = 1;
    obst_.erase(f); near_.erase(f); NDONE++;
  }
  std::map<int, std::vector<uint8_t>> obst_; std::map<int, std::vector<float>> near_;
  rclcpp::Subscription<std_msgs::msg::UInt8MultiArray>::SharedPtr a_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr b_;
};

int main(int argc, char **argv) {
  if (argc < 9) { fprintf(stderr, "usage: lidar_ros <scan_dir> <idx_file> <out.csv> <nframes> <period_ms> <calib|val> <regime_file> <seed>\n"); return 2; }
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
