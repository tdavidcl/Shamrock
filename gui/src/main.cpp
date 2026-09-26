// Shamrock control GUI — C++ app with dockable panes (Dear ImGui docking branch).
//
// Four panes (Viewer, Graph, Script, Profile) live in one dock area arranged by kitty-style layouts
// (Stack, Tall, Fat, Grid, Horizontal, Vertical, Splits). Tabs can be dragged onto drop targets, dividers
// resized; interactive runs remember everything in shamrock_gui_layout.ini. All data is demo data from
// DemoSimulation (deterministic 60 fps clock and SplitMix64 seeds for --screenshot / --bench).
//
//   ./shamrock_gui                        interactive
//   ./shamrock_gui --layout grid          stack | tall | fat | grid | horizontal | vertical | splits
//   ./shamrock_gui --profile              start with the Profile pane shown
//   ./shamrock_gui --ui-scale 1.5         start at 150 %
//   ./shamrock_gui --screenshot shot.png  render 45 frames (or --frames N), save PNG, exit
//   ./shamrock_gui --bench 300            print per-frame CPU timings as JSON

#include "imgui.h"
#include "imgui_internal.h"  // DockBuilder API (layout presets)
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "TextEditor.h"

#include <GLFW/glfw3.h>
#if defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
static fs::path g_assets = "assets";
static constexpr double PI = 3.14159265358979323846;

// ════════════════════════════════════════════════════════════════════════════
//  Design tokens (taken from the mockup)
// ════════════════════════════════════════════════════════════════════════════
static constexpr ImU32 rgb_u32(int r, int g, int b, int a = 255) {
    return (ImU32(a) << 24) | (ImU32(b) << 16) | (ImU32(g) << 8) | ImU32(r);
}
static constexpr int hexv(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }
static constexpr ImU32 rgba(const char* h, double a = 1.0) {
    return rgb_u32(hexv(h[1]) * 16 + hexv(h[2]), hexv(h[3]) * 16 + hexv(h[4]), hexv(h[5]) * 16 + hexv(h[6]),
                   int(a * 255 + 0.5));
}

namespace C {
constexpr ImU32 APP_BG = rgba("#141517"), PANEL = rgba("#1b1c1f"), CANVAS = rgba("#17181b"),
                DIVIDER = rgba("#2e3035"), BORDER = rgba("#34363c"), BUTTON = rgba("#202125"),
                NODE = rgba("#232428"), NODE_BORDER = rgba("#3a3c42"), CARD = rgba("#111214"),
                DARK = rgba("#0b0c10"), GRID_DOT = rgba("#2b2d33"), ROW_HL = rgba("#2a2b30");
constexpr ImU32 TEXT = rgba("#e7e5df"), TEXT_2 = rgba("#c9c6be"), TEXT_3 = rgba("#a9a69e"), ROW = rgba("#b3b0a8"),
                MUTED = rgba("#8e8b84"), DIM = rgba("#6f6d67"), GUTTER = rgba("#5f5d58");
constexpr ImU32 ACCENT = rgba("#e8a33d"), ACCENT_BG = rgba("#2a2418"), ACCENT_TEXT = rgba("#f4e2c0"),
                ON_ACCENT = rgba("#1b1407"), WARM_BG = rgba("#262015"), WARM_BORDER = rgba("#4a3b22"),
                WARM_TEXT = rgba("#f1d7a8");
constexpr ImU32 TEAL = rgba("#4fb3a9"), TEAL_TEXT = rgba("#8fb8b2"), PILL_BG = rgba("#1a2624"),
                PILL_BORDER = rgba("#2f4d49"), PILL_TEXT = rgba("#cfe9e4"), BLUE = rgba("#6f9be8"),
                GRAY = rgba("#8a8780");
struct HeaderStyle { ImU32 bg, fg, chip; };
constexpr HeaderStyle INPUT{rgba("#22403c"), rgba("#cfe9e4"), rgba("#2d5550")};
constexpr HeaderStyle SOLVER{rgba("#43301f"), rgba("#f3dcc2"), rgba("#5c4029")};
}  // namespace C

// ════════════════════════════════════════════════════════════════════════════
//  Fonts and small drawing helpers
// ════════════════════════════════════════════════════════════════════════════
struct Fonts { static inline ImFont *sans, *medium, *semibold, *mono; };

static void load_fonts() {
    ImGuiIO& io = ImGui::GetIO();
    auto load = [&](const char* name, bool merge = false) {
        ImFontConfig cfg;
        cfg.MergeMode = merge;
        std::string path = (g_assets / "fonts" / (std::string(name) + ".ttf")).string();
        ImFont* f = io.Fonts->AddFontFromFileTTF(path.c_str(), 13.0f, &cfg);
        IM_ASSERT(f && "font not found: run from the project folder or pass --assets");
        return f;
    };
    Fonts::sans = load("IBMPlexSans-Regular");
    Fonts::medium = load("IBMPlexSans-Medium");
    Fonts::semibold = load("IBMPlexSans-SemiBold");
    Fonts::mono = load("IBMPlexMono-Regular");
    load("IBMPlexSans-Regular", true);  // Plex Mono has no Greek (ρ): borrow it from Sans
}

static inline ImVec2 V(double x, double y) { return ImVec2(float(x), float(y)); }

// Global UI scale. The app is laid out in *logical* pixels (the mockup's 1x sizes); these helpers
// convert to physical pixels when drawing, placing widgets and reading the mouse.
struct UI {
    static inline double scale = 1.0;
    static inline ImVec2 origin{0, 0};  // main viewport position (scaling is done around it)
    static constexpr double MIN = 0.5, MAX = 3.0, STEP = 0.1;
};
static inline ImVec2 P(ImVec2 p) {
    const float s = float(UI::scale);
    return ImVec2(UI::origin.x + (p.x - UI::origin.x) * s, UI::origin.y + (p.y - UI::origin.y) * s);
}
static inline float S(double v) { return float(v * UI::scale); }
static ImVec2 mouse_pos() {
    ImVec2 m = ImGui::GetIO().MousePos;
    return V(UI::origin.x + (m.x - UI::origin.x) / UI::scale, UI::origin.y + (m.y - UI::origin.y) / UI::scale);
}
static ImVec2 mouse_delta() {
    ImVec2 d = ImGui::GetIO().MouseDelta;
    return V(d.x / UI::scale, d.y / UI::scale);
}
static void set_cursor(ImVec2 p) { ImGui::SetCursorScreenPos(P(p)); }
static bool invisible_button(const char* id, ImVec2 size, ImGuiButtonFlags flags = 0) {
    return ImGui::InvisibleButton(id, V(std::max(size.x * UI::scale, 1.0), std::max(size.y * UI::scale, 1.0)), flags);
}

// Same calls as ImDrawList, but taking logical coordinates and sizes.
struct SDL {
    ImDrawList* d = nullptr;
    void AddRectFilled(ImVec2 a, ImVec2 b, ImU32 c, float r = 0, ImDrawFlags f = 0) { d->AddRectFilled(P(a), P(b), c, S(r), f); }
    void AddRect(ImVec2 a, ImVec2 b, ImU32 c, float r = 0, ImDrawFlags f = 0, float th = 1) { d->AddRect(P(a), P(b), c, S(r), f, S(th)); }
    void AddRectFilledMultiColor(ImVec2 a, ImVec2 b, ImU32 c1, ImU32 c2, ImU32 c3, ImU32 c4) { d->AddRectFilledMultiColor(P(a), P(b), c1, c2, c3, c4); }
    void AddLine(ImVec2 a, ImVec2 b, ImU32 c, float th = 1) { d->AddLine(P(a), P(b), c, S(th)); }
    void AddCircle(ImVec2 c, float r, ImU32 col, int seg = 0, float th = 1) { d->AddCircle(P(c), S(r), col, seg, S(th)); }
    void AddCircleFilled(ImVec2 c, float r, ImU32 col, int seg = 0) { d->AddCircleFilled(P(c), S(r), col, seg); }
    void AddTriangle(ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col, float th = 1) { d->AddTriangle(P(a), P(b), P(c), col, S(th)); }
    void AddTriangleFilled(ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col) { d->AddTriangleFilled(P(a), P(b), P(c), col); }
    void AddConvexPolyFilled(const ImVec2* pts, int n, ImU32 col) { auto q = scaled(pts, n); d->AddConvexPolyFilled(q.data(), n, col); }
    void AddPolyline(const ImVec2* pts, int n, ImU32 col, ImDrawFlags f, float th) { auto q = scaled(pts, n); d->AddPolyline(q.data(), n, col, f, S(th)); }
    void AddBezierCubic(ImVec2 a, ImVec2 b, ImVec2 c, ImVec2 e, ImU32 col, float th, int seg = 0) { d->AddBezierCubic(P(a), P(b), P(c), P(e), col, S(th), seg); }
    void AddText(ImFont* f, float size, ImVec2 pos, ImU32 col, const char* s) { d->AddText(f, S(size), P(pos), col, s); }
    void AddImage(ImTextureRef t, ImVec2 a, ImVec2 b) { d->AddImage(t, P(a), P(b)); }
    void AddImageRounded(ImTextureRef t, ImVec2 a, ImVec2 b, ImVec2 uv0, ImVec2 uv1, ImU32 col, float r) { d->AddImageRounded(t, P(a), P(b), uv0, uv1, col, S(r)); }
    void AddImageQuad(ImTextureRef t, ImVec2 a, ImVec2 b, ImVec2 c, ImVec2 e) { d->AddImageQuad(t, P(a), P(b), P(c), P(e)); }
    void PushClipRect(ImVec2 a, ImVec2 b, bool intersect = false) { d->PushClipRect(P(a), P(b), intersect); }
    void PopClipRect() { d->PopClipRect(); }
private:
    static std::vector<ImVec2> scaled(const ImVec2* pts, int n) {
        std::vector<ImVec2> q(n);
        for (int i = 0; i < n; ++i) q[i] = P(pts[i]);
        return q;
    }
};
static SDL g_sdl[16];
static int g_sdl_next = 0;
static SDL* window_draw_list() {  // one wrapper per window drawn in the frame (main window + panes)
    SDL* s = &g_sdl[g_sdl_next++ % 16];
    s->d = ImGui::GetWindowDrawList();
    return s;
}

struct TextKey {
    const ImFont* font; int size100; std::string s; int scale100;
    bool operator==(const TextKey& o) const {
        return font == o.font && size100 == o.size100 && s == o.s && scale100 == o.scale100;
    }
};
struct TextKeyHash {
    size_t operator()(const TextKey& k) const {
        return std::hash<std::string>()(k.s) ^ (std::hash<const void*>()(k.font) << 1) ^ size_t(k.size100) * 31;
    }
};
static std::unordered_map<TextKey, double, TextKeyHash> g_text_cache;

static double text_w(ImFont* font, double size, const std::string& s) {
    TextKey key{font, int(std::lround(size * 100)), s, int(std::lround(UI::scale * 100))};
    auto it = g_text_cache.find(key);
    if (it != g_text_cache.end()) return it->second;
    ImGui::PushFont(font, float(size * UI::scale));
    double w = ImGui::CalcTextSize(s.c_str()).x / UI::scale;
    ImGui::PopFont();
    if (g_text_cache.size() > 4000) g_text_cache.clear();
    g_text_cache.emplace(std::move(key), w);
    return w;
}

static void draw_text(SDL* dl, ImFont* font, double size, double x, double y, ImU32 col, const std::string& s) {
    dl->AddText(font, float(size), V(x, y), col, s.c_str());
}
static void draw_text_vc(SDL* dl, ImFont* font, double size, double x, double cy, ImU32 col,
                         const std::string& s) {
    dl->AddText(font, float(size), V(x, cy - size * 0.66), col, s.c_str());
}

static void tooltip(const char* text) {
    ImGui::PushFont(Fonts::sans, S(13));
    ImGui::SetItemTooltip("%s", text);
    ImGui::PopFont();
}

struct Hit { bool clicked, hovered; };
// Optional observer of every clickable area (physical pixels); used by tools such as the demo recorder.
static void (*g_on_hit)(const char* id, ImVec2 min, ImVec2 max) = nullptr;
static Hit hit(const char* id, double x, double y, double w, double h) {
    set_cursor(V(x, y));
    bool clicked = invisible_button(id, V(std::max(w, 1.0), std::max(h, 1.0)));
    bool hovered = ImGui::IsItemHovered();
    if (g_on_hit) g_on_hit(id, P(V(x, y)), P(V(x + w, y + h)));
    if (hovered) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return {clicked, hovered};
}

static ImU32 lighten(ImU32 col, int amount = 14) {
    int r = col & 255, g = (col >> 8) & 255, b = (col >> 16) & 255, a = (col >> 24) & 255;
    return rgb_u32(std::min(r + amount, 255), std::min(g + amount, 255), std::min(b + amount, 255), a);
}

static bool framed_button(SDL* dl, const char* id, double x, double y, double w, double h, ImU32 bg,
                          const ImU32* border, double radius = 5.0) {
    Hit r = hit(id, x, y, w, h);
    dl->AddRectFilled(V(x, y), V(x + w, y + h), r.hovered ? lighten(bg) : bg, float(radius));
    if (border) dl->AddRect(V(x + 0.5, y + 0.5), V(x + w - 0.5, y + h - 0.5), *border, float(radius));
    return r.clicked;
}
static bool framed_button(SDL* dl, const char* id, double x, double y, double w, double h, ImU32 bg,
                          ImU32 border, double radius = 5.0) {
    return framed_button(dl, id, x, y, w, h, bg, &border, radius);
}

struct Btn { bool clicked; double w; };
static Btn text_button(SDL* dl, const char* id, double x, double cy, const std::string& label,
                       bool warm = false, bool dot = false, double h = 26.0) {
    ImFont* font = Fonts::sans;
    const double size = 12.0, pad = 10.0, dot_w = dot ? 12.0 : 0.0;
    double w = pad * 2 + dot_w + text_w(font, size, label);
    double y = cy - h / 2;
    ImU32 bg = warm ? C::WARM_BG : C::BUTTON, border = warm ? C::WARM_BORDER : C::BORDER,
          fg = warm ? C::WARM_TEXT : C::TEXT_2;
    bool clicked = framed_button(dl, id, x, y, w, h, bg, border, 5.0);
    double tx = x + pad;
    if (dot) {
        dl->AddCircleFilled(V(tx + 3, cy), 3.0f, C::ACCENT);
        tx += dot_w;
    }
    draw_text_vc(dl, font, size, tx, cy, fg, label);
    return {clicked, w};
}

static void dashed_line(SDL* dl, double ax, double ay, double bx, double by, ImU32 col, double dash = 4.0,
                        double gap = 4.0, double thickness = 1.0) {
    double length = std::hypot(bx - ax, by - ay);
    if (length < 1e-3) return;
    double ux = (bx - ax) / length, uy = (by - ay) / length, t = 0.0;
    while (t < length) {
        double t2 = std::min(t + dash, length);
        dl->AddLine(V(ax + ux * t, ay + uy * t), V(ax + ux * t2, ay + uy * t2), col, float(thickness));
        t += dash + gap;
    }
}
static void draw_live_dot(SDL* dl, double cx, double cy, double r = 3.0) {
    dl->AddCircleFilled(V(cx, cy), float(r), C::ACCENT);
}

// ════════════════════════════════════════════════════════════════════════════
//  GPU textures and colormaps
// ════════════════════════════════════════════════════════════════════════════
struct GLTexture {
    int w = 0, h = 0;
    GLuint id = 0;
    ImTextureRef ref;
    void create(int w_, int h_, const unsigned char* data = nullptr) {
        w = w_; h = h_;
        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        ref = ImTextureRef(ImTextureID(intptr_t(id)));
    }
    void upload(const std::vector<uint8_t>& rgba_img) const {
        glBindTexture(GL_TEXTURE_2D, id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba_img.data());
    }
};

using Lut = std::array<std::array<uint8_t, 4>, 256>;
static Lut make_lut(const std::vector<const char*>& stops) {
    // same as numpy: np.interp on a 256-sample linspace, then truncation to uint8
    const size_t n = stops.size();
    std::vector<double> xs(n);
    for (size_t i = 0; i < n; ++i) xs[i] = double(i) / double(n - 1);
    xs[n - 1] = 1.0;
    Lut lut{};
    for (int i = 0; i < 256; ++i) {
        double t = i == 255 ? 1.0 : i * (1.0 / 255.0);
        size_t j = std::min<size_t>(size_t(std::upper_bound(xs.begin(), xs.end(), t) - xs.begin()), n - 1);
        j = j == 0 ? 0 : j - 1;
        for (int k = 0; k < 3; ++k) {
            const char* s = stops[j];
            const char* s2 = stops[std::min(j + 1, n - 1)];
            double f0 = hexv(s[1 + 2 * k]) * 16 + hexv(s[2 + 2 * k]);
            double f1 = hexv(s2[1 + 2 * k]) * 16 + hexv(s2[2 + 2 * k]);
            double v = (t >= xs[n - 1]) ? f1 : (f1 - f0) / (xs[j + 1] - xs[j]) * (t - xs[j]) + f0;
            lut[i][k] = uint8_t(v);
        }
        lut[i][3] = 255;
    }
    return lut;
}
static const Lut VIRIDIS = make_lut({"#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89",
                                     "#35b779", "#6ece58", "#fde725"});
static const Lut TRACER_LUT = make_lut({"#0b0c10", "#1c2a4a", "#3d5f9e", "#9ec0ff", "#e4eeff"});

static inline void colormap_px(float v01, const Lut& lut, uint8_t* out) {
    int idx = std::clamp(int(v01 * 255.0f), 0, 255);
    std::memcpy(out, lut[idx].data(), 4);
}
static ImU32 lut_color(const Lut& lut, double v01, double shade = 1.0) {
    const auto& c = lut[int(std::clamp(v01, 0.0, 1.0) * 255)];
    return rgb_u32(int(std::min(c[0] * shade, 255.0)), int(std::min(c[1] * shade, 255.0)),
                   int(std::min(c[2] * shade, 255.0)));
}
static ImU32 viridis_u32(double v01) { return lut_color(VIRIDIS, v01); }

// ════════════════════════════════════════════════════════════════════════════
//  Demo data source (stand-in for the remote solver)
// ════════════════════════════════════════════════════════════════════════════
struct FieldInfo { const char* id; const char* label; const char* cb_label; bool log; };
static const FieldInfo FIELDS[3] = {{"rho", "ρ", "ρ log", true}, {"p", "p", "p log", true},
                                    {"v", "|v|", "|v| lin", false}};

struct Rng {  // SplitMix64, bit-identical to the Python Rng
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint64_t next_u64() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return double(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
    double normal(double mu, double sigma) {
        double u1 = 1.0 - uniform(), u2 = uniform();
        return mu + sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
    }
};

struct Image { int w, h; std::vector<uint8_t> px; };

class DemoSimulation {
public:
    static constexpr double T_REF = 0.03214, R_REF = 0.37, UNTIL = 0.05;
    double t = T_REF;
    long long step = 18420;
    bool running = true;
    double step_ms = 4.1;
    std::map<std::string, std::deque<double>> history;

    // Per-step profiling scopes as the solver would report them (rank 0). Scopes are listed parents
    // first; `ema` is the running mean the flamegraph shows. In the real app this arrives as one more
    // subscription on the SSH channel.
    enum class Cat { Root, Gpu, Mpi, Host, Io };
    struct Scope { const char* name; int parent; Cat cat; double base_ms; double ema = 0; bool leaf = true; };
    std::vector<Scope> prof = {
        {"step", -1, Cat::Root, 0.03},
        {"hydro step", 0, Cat::Host, 0.05},
        {"reconstruct (PPM)", 1, Cat::Gpu, 0.62},
        {"riemann (HLLC)", 1, Cat::Gpu, 0.95},
        {"flux update", 1, Cat::Gpu, 0.48},
        {"halo exchange", 1, Cat::Mpi, 0.02},
        {"pack", 5, Cat::Gpu, 0.12},
        {"MPI_Isend / Irecv", 5, Cat::Mpi, 0.31},
        {"unpack", 5, Cat::Gpu, 0.10},
        {"boundaries", 1, Cat::Gpu, 0.12},
        {"CFL reduce", 1, Cat::Mpi, 0.18},
        {"advect tracers", 0, Cat::Host, 0.02},
        {"velocity interp", 11, Cat::Gpu, 0.30},
        {"RK2 push", 11, Cat::Gpu, 0.22},
        {"migrate", 11, Cat::Mpi, 0.10},
        {"diagnostics", 0, Cat::Host, 0.01},
        {"E_tot allreduce", 15, Cat::Mpi, 0.12},
        {"speed counter", 15, Cat::Host, 0.04},
        {"preview extract", 0, Cat::Host, 0.01},
        {"slice", 18, Cat::Gpu, 0.12},
        {"bin tracers", 18, Cat::Gpu, 0.08},
        {"D2H copy", 18, Cat::Gpu, 0.05},
        {"compress", 18, Cat::Host, 0.03},
        {"python driver", 0, Cat::Host, 0.10},
        {"checkpoint write", 0, Cat::Io, 0.0},
    };

    DemoSimulation() : rng_(3) {
        for (auto& s : prof)
            if (s.parent >= 0) prof[s.parent].leaf = false;
        for (int k = 0; k < 20; ++k) sample_profile();  // start from a settled mean
        Rng rng(7);
        const int n = 40000;
        ang_.resize(n);
        frac_.resize(n);
        for (int i = 0; i < n; ++i) {
            ang_[i] = rng.uniform() * 2.0 * PI;
            bool shell = rng.uniform() < 0.86;
            double u = rng.uniform();
            frac_[i] = shell ? 1.0 - 0.16 * std::pow(u, 0.6) : std::sqrt(u) * 0.8;
        }
        for (const char* k : {"E_tot", "dt", "speed"}) history[k];
        for (int i = 0; i < 160; ++i) push_history(T_REF * (0.25 + 0.75 * i / 159), i < 3);
    }

    double dt() const { return dt_at(t); }
    static double dt_at(double tt) { return 1.81e-6 * std::pow(tt / T_REF, 0.6); }

    void advance(double wall_dt) {
        if (!running) return;
        int steps = int(std::max(1.0, wall_dt * 240));
        for (int i = 0; i < steps; ++i) {
            t += dt() * 1.1;
            step += 1;
        }
        if (t >= UNTIL) t = T_REF * 0.6;  // loop the demo
        step_ms = 4.1 + rng_.normal(0, 0.08);
        push_history(t);
        sample_profile();
    }

    void sample_profile() {
        std::vector<double> v(prof.size(), 0.0);
        for (int i = int(prof.size()) - 1; i >= 0; --i) {  // children come after their parent
            const Scope& s = prof[i];
            double self = s.base_ms * std::max(0.0, 1 + prof_rng_.normal(0, 0.06));
            if (s.cat == Cat::Io) self = checkpoint_now_ ? 6.0 * (1 + prof_rng_.normal(0, 0.1)) : 0.0;
            v[i] += self;
            if (s.parent >= 0) v[s.parent] += v[i];
        }
        checkpoint_now_ = false;
        for (size_t i = 0; i < prof.size(); ++i) prof[i].ema += 0.08 * (v[i] - prof[i].ema);
    }

    double shock_radius() const { return R_REF * std::pow(t / T_REF, 0.4); }

    // float instantiation mirrors the numpy float32 slice maths, double mirrors `sample`
    template <class T>
    T profile(int field, T r, T theta) const {
        const double R = shock_radius();
        const T q = r / T(R);
        if (!(q < T(1)))  // outside the shock the value is constant: skip the trig (same result)
            return field == 0 ? T(1) : field == 1 ? T(1e-5) : T(0);
        const T ripple = T(1) + T(0.04) * std::sin(T(7) * theta + T(3) * q) *
                                    std::pow(std::clamp(q, T(0), T(1)), T(4));
        if (field == 0) return (T(0.02) + T(3.98) * std::pow(q, T(9))) * ripple;
        if (field == 1) {
            T ps = T(2.4 * std::pow(R_REF / R, 3.0));
            return ps * (T(0.36) + T(0.64) * std::pow(q, T(6))) * ripple;
        }
        T vs = T(1.9 * std::pow(R_REF / R, 1.5));
        return vs * q * ripple;
    }

    double sample(int field, double x, double y) const { return profile<double>(field, std::hypot(x, y), std::atan2(y, x)); }

    std::pair<double, double> value_range(int field) const {
        double R = shock_radius();
        if (field == 0) return {0.02, 4.0};
        if (field == 1) return {1e-5, 2.4 * std::pow(R_REF / R, 3.0)};
        return {0.0, 1.9 * std::pow(R_REF / R, 1.5)};
    }

    template <class T>
    static T normalize(int field, T f, double lo, double hi) {
        if (FIELDS[field].log)
            return (std::log10(std::max(f, T(lo))) - T(std::log10(lo))) / T(std::log10(hi) - std::log10(lo));
        return (f - T(lo)) / T(std::max(hi - lo, 1e-30));
    }

    Image slice(int field, int w, int h, double* lo_out = nullptr, double* hi_out = nullptr) const {
        const double aspect = double(w) / h;
        auto [lo, hi] = value_range(field);
        const double R = shock_radius();
        Image img{w, h, std::vector<uint8_t>(size_t(w) * h * 4)};
        std::vector<float> xs(w), ys(h);
        const double x0 = -0.5 * aspect, x1 = 0.5 * aspect, sx = (x1 - x0) / (w - 1), sy = -1.0 / (h - 1);
        for (int i = 0; i < w; ++i) xs[i] = float(i == w - 1 ? x1 : x0 + i * sx);
        for (int j = 0; j < h; ++j) ys[j] = float(j == h - 1 ? -0.5 : 0.5 + j * sy);
        // outside the shock the field is constant: colour it once (identical result, far fewer libm calls)
        uint8_t outside[4];
        colormap_px(normalize<float>(field, profile<float>(field, 2.0f * float(R) + 1.0f, 0.0f), lo, hi), VIRIDIS,
                    outside);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) {
                float X = xs[i], Y = ys[j];
                float r = std::hypot(X, Y);
                uint8_t* out = &img.px[(size_t(j) * w + i) * 4];
                if (r / float(R) < 1.0f)
                    colormap_px(normalize<float>(field, profile<float>(field, r, std::atan2(Y, X)), lo, hi), VIRIDIS, out);
                else
                    std::memcpy(out, outside, 4);
            }
        if (lo_out) *lo_out = lo;
        if (hi_out) *hi_out = hi;
        return img;
    }

    std::vector<std::array<double, 2>> tracers_xy(int count = -1) const {
        int n = count < 0 ? int(ang_.size()) : count;
        double R = shock_radius();
        std::vector<std::array<double, 2>> out(n);
        for (int i = 0; i < n; ++i) {
            double rad = frac_[i] * R;
            out[i] = {std::cos(ang_[i]) * rad, std::sin(ang_[i]) * rad};
        }
        return out;
    }

    Image tracers_image(int w, int h) const {
        auto xy = tracers_xy();
        const double aspect = double(w) / h;
        std::vector<float> bins(size_t(w) * h, 0.0f);
        for (auto& p : xy) {
            int ix = std::clamp(int((p[0] / aspect + 0.5) * w), 0, w - 1);
            int iy = std::clamp(int((0.5 - p[1]) * h), 0, h - 1);
            bins[size_t(iy) * w + ix] += 1.0f;
        }
        float mx = 0;
        for (float& b : bins) { b = std::log1p(b); mx = std::max(mx, b); }
        mx = std::max(mx, 1e-6f);
        Image img{w, h, std::vector<uint8_t>(size_t(w) * h * 4)};
        for (size_t k = 0; k < bins.size(); ++k) colormap_px(bins[k] / mx, TRACER_LUT, &img.px[k * 4]);
        return img;
    }

private:
    std::vector<double> ang_, frac_;
    Rng rng_;
    Rng prof_rng_{11};
    bool checkpoint_now_ = false;

    void push_history(double tt, bool warmup = false) {
        auto push = [&](const char* k, double v) {
            auto& d = history[k];
            d.push_back(v);
            if (d.size() > 160) d.pop_front();
        };
        push("E_tot", 1.0 + 2e-7 * (tt / T_REF) + rng_.normal(0, 1.2e-8));
        push("dt", dt_at(tt) * (1 + rng_.normal(0, 0.01)));
        double speed = 4.1e9 * (1 + rng_.normal(0, 0.03));
        if (warmup) speed *= 0.45;
        if (rng_.uniform() < 0.015) {  // an occasional checkpoint write
            speed *= 0.35;
            checkpoint_now_ = true;
        }
        push("speed", speed);
    }
};

// ════════════════════════════════════════════════════════════════════════════
//  Graph model + in-house canvas (compute nodes, data edges, links)
// ════════════════════════════════════════════════════════════════════════════
enum class RowKind { Param, In, RW, Out };
struct Row { RowKind kind; std::string label, value; ImU32 color = 0; bool highlight = false; };
enum class Preview { None, Slice, Tracers, Series };

struct Node {
    std::string id;
    bool compute;
    std::string title;
    double x, y, w;
    C::HeaderStyle style = C::SOLVER;
    bool gpu = false;
    std::vector<Row> rows;
    ImU32 color = 0;
    std::string meta;
    Preview preview = Preview::None;
    std::string footer;
    double h() const {
        if (compute) return 30 + 8 + 26.0 * rows.size();
        switch (preview) {
            case Preview::Slice: case Preview::Tracers: return 153;
            case Preview::Series: return 121;
            default: return 72;
        }
    }
};
struct Link { std::string src, src_port, dst, dst_port; ImU32 color; };

static std::pair<std::vector<Node>, std::vector<Link>> build_demo_graph() {
    const ImU32 A = C::ACCENT, G = C::GRAY, B = C::BLUE, T = C::TEAL;
    auto param = [](const char* l, const char* v, bool hl = false) { return Row{RowKind::Param, l, v, 0, hl}; };
    auto port = [](RowKind k, const char* l, ImU32 c) { return Row{k, l, "", c, false}; };
    std::vector<Node> nodes;
    nodes.push_back({"sedov", true, "Sedov init", 20, 150, 160, C::INPUT, false,
                     {param("E₀", "1.0"), param("n_tracers", "1.2M"), port(RowKind::Out, "mesh", G),
                      port(RowKind::Out, "state", A), port(RowKind::Out, "tracers", B)}});
    nodes.push_back({"hydro", true, "Hydro step", 460, 40, 160, C::SOLVER, true,
                     {port(RowKind::In, "mesh", G), port(RowKind::RW, "state", A), param("riemann", "hllc"),
                      param("cfl", "0.40", true), port(RowKind::Out, "diagnostics", T)}});
    nodes.push_back({"advect", true, "Advect tracers", 460, 300, 160, C::SOLVER, true,
                     {port(RowKind::In, "state", A), port(RowKind::RW, "tracers", B), param("scheme", "rk2")}});
    auto edge = [](const char* id, double x, double y, ImU32 c, const char* meta, Preview p, const char* footer) {
        Node n{id, false, id, x, y, 150};
        n.color = c; n.meta = meta; n.preview = p; n.footer = footer;
        return n;
    };
    nodes.push_back(edge("mesh", 240, 20, G, "amr", Preview::None, ""));
    nodes.push_back(edge("state", 240, 130, A, "ρ v p", Preview::Slice, "ρ · z=0.5 · 128²"));
    nodes.push_back(edge("tracers", 240, 306, B, "1.2M", Preview::Tracers, "xy proj · 128²"));
    Node diag = edge("diag", 700, 60, T, "series", Preview::Series, "E_tot, dt, speed · 1 Hz");
    diag.title = "diagnostics";
    nodes.push_back(diag);
    std::vector<Link> links = {
        {"sedov", "mesh", "mesh", "in", G},       {"sedov", "state", "state", "in", A},
        {"sedov", "tracers", "tracers", "in", B}, {"mesh", "out", "hydro", "mesh", G},
        {"state", "out", "hydro", "state", A},    {"state", "out", "advect", "state", A},
        {"tracers", "out", "advect", "tracers", B}, {"hydro", "diagnostics", "diag", "in", T},
    };
    return {nodes, links};
}

static void sparkline(SDL* dl, double x0, double y0, double x1, double y1, const std::deque<double>& values,
                      ImU32 col, double thickness = 1.6, double pad_frac = 0.12) {
    const size_t n = values.size();
    if (n < 2) return;
    auto [mn, mx] = std::minmax_element(values.begin(), values.end());
    double lo = *mn, hi = *mx, span = hi > lo ? hi - lo : 1.0;
    lo -= span * pad_frac;
    hi += span * pad_frac;
    std::vector<ImVec2> pts(n);
    const double step = (x1 - x0) / double(n - 1);
    for (size_t i = 0; i < n; ++i) {
        double x = i == n - 1 ? x1 : x0 + double(i) * step;
        pts[i] = V(x, y1 - (values[i] - lo) / (hi - lo) * (y1 - y0));
    }
    dl->AddPolyline(pts.data(), int(n), col, 0, float(thickness));
}

struct App;  // fwd

class GraphView {
public:
    std::vector<Node> nodes;
    std::vector<Link> links;
    double zoom = 1.0, pan[2] = {0, 0}, grid_offset[2] = {0, 0};
    bool user_view = false;
    double last_size[2] = {0, 0};
    std::string selected = "state", dragging;
    bool previews_on = true;

    GraphView(std::vector<Node> n, std::vector<Link> l) : nodes(std::move(n)), links(std::move(l)) {}

    Node* by_id(const std::string& id) {
        for (auto& n : nodes)
            if (n.id == id) return &n;
        return nullptr;
    }
    std::array<double, 2> to_screen(const double* origin, double wx, double wy) const {
        return {origin[0] + pan[0] + wx * zoom, origin[1] + pan[1] + wy * zoom};
    }
    std::array<double, 2> port_world(const Node& n, const std::string& port) const {
        if (!n.compute) return {port == "in" ? n.x : n.x + n.w, n.y + 16};
        for (size_t i = 0; i < n.rows.size(); ++i) {
            const Row& r = n.rows[i];
            if (r.label == port && r.kind != RowKind::Param)
                return {r.kind == RowKind::Out ? n.x + n.w : n.x, n.y + 34 + 26.0 * i + 13};
        }
        return {n.x, n.y};
    }
    void fit(double w, double h) {
        double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
        for (auto& n : nodes) {
            x0 = std::min(x0, n.x); y0 = std::min(y0, n.y);
            x1 = std::max(x1, n.x + n.w); y1 = std::max(y1, n.y + n.h());
        }
        x0 -= 20; y0 -= 20; x1 += 20; y1 += 36;
        zoom = std::max(0.35, std::min({w / (x1 - x0), h / (y1 - y0), 1.3}));
        pan[0] = (w - (x1 - x0) * zoom) / 2 - x0 * zoom;
        pan[1] = std::max(4.0, (h - (y1 - y0) * zoom) / 2) - y0 * zoom;
    }

    void draw(App& app, double x, double y, double w, double h);

private:
    void draw_grid(SDL* dl, double x, double y, double w, double h) const {
        const double step = 20.0;
        auto pymod = [](double a, double b) { double m = std::fmod(a, b); return m < 0 ? m + b : m; };
        const double ox = pymod(grid_offset[0], step), oy = pymod(grid_offset[1], step);
        for (double gx = x + ox; gx < x + w; gx += step)
            for (double gy = y + oy; gy < y + h; gy += step)
                dl->AddRectFilled(V(gx, gy), V(gx + 1.2, gy + 1.2), C::GRID_DOT);
    }

    void interact(double x, double y, double w, double h, const double* origin) {
        set_cursor(V(x, y));
        invisible_button("##graph_canvas", V(w, h),
                               ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
                                   ImGuiButtonFlags_MouseButtonMiddle);
        bool hovered = ImGui::IsItemHovered(), active = ImGui::IsItemActive();
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 m = mouse_pos(), md = mouse_delta();
        double mx = m.x, my = m.y;
        if (hovered && io.MouseWheel != 0.0f) {
            double old = zoom;
            zoom = std::max(0.3, std::min(zoom * std::pow(1.12, double(io.MouseWheel)), 3.0));
            double wx = (mx - x - pan[0]) / old, wy = (my - y - pan[1]) / old;
            pan[0] = mx - x - wx * zoom;
            pan[1] = my - y - wy * zoom;
            user_view = true;
        }
        if (hovered && ImGui::IsMouseClicked(0)) {
            dragging.clear();
            selected.clear();
            for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
                auto s = to_screen(origin, it->x, it->y);
                if (s[0] <= mx && mx <= s[0] + it->w * zoom && s[1] <= my && my <= s[1] + it->h() * zoom) {
                    selected = dragging = it->id;
                    break;
                }
            }
        }
        if (!dragging.empty() && ImGui::IsMouseDown(0)) {
            Node* n = by_id(dragging);
            n->x += md.x / zoom;
            n->y += md.y / zoom;
        }
        if (!ImGui::IsMouseDown(0)) dragging.clear();
        if (active && (ImGui::IsMouseDragging(1) || ImGui::IsMouseDragging(2))) {
            pan[0] += md.x; pan[1] += md.y;
            grid_offset[0] += md.x; grid_offset[1] += md.y;
            user_view = true;
        }
        if (hovered && ImGui::IsMouseDoubleClicked(0) && selected.empty()) {
            user_view = false;
            fit(w, h);
        }
    }

    void draw_link(SDL* dl, const double* origin, const Link& ln) {
        auto a = port_world(*by_id(ln.src), ln.src_port);
        auto b = port_world(*by_id(ln.dst), ln.dst_port);
        auto p1 = to_screen(origin, a[0], a[1]);
        auto p4 = to_screen(origin, b[0], b[1]);
        double k = std::max(std::abs(p4[0] - p1[0]) * 0.5, 30 * zoom);
        dl->AddBezierCubic(V(p1[0], p1[1]), V(p1[0] + k, p1[1]), V(p4[0] - k, p4[1]), V(p4[0], p4[1]), ln.color,
                           2.0f);
    }

    void port(SDL* dl, double cx, double cy, ImU32 color, bool diamond = false) const {
        double r_out = 5.0 * zoom, r_in = 3.2 * zoom;
        if (diamond) {
            double a = r_out * 1.25, b = r_in * 1.3;
            ImVec2 o[4] = {V(cx, cy - a), V(cx + a, cy), V(cx, cy + a), V(cx - a, cy)};
            ImVec2 i[4] = {V(cx, cy - b), V(cx + b, cy), V(cx, cy + b), V(cx - b, cy)};
            dl->AddConvexPolyFilled(o, 4, C::CANVAS);
            dl->AddConvexPolyFilled(i, 4, color);
        } else {
            dl->AddCircleFilled(V(cx, cy), float(r_out), C::CANVAS);
            dl->AddCircleFilled(V(cx, cy), float(r_in), color);
        }
    }

    void draw_compute(SDL* dl, const double* origin, const Node& n) {
        const double z = zoom;
        auto p0 = to_screen(origin, n.x, n.y);
        double x0 = p0[0], y0 = p0[1], x1 = x0 + n.w * z, y1 = y0 + n.h() * z;
        dl->AddRectFilled(V(x0, y0), V(x1, y1), C::NODE, float(8 * z));
        dl->AddRectFilled(V(x0, y0), V(x1, y0 + 30 * z), n.style.bg, float(8 * z), ImDrawFlags_RoundCornersTop);
        dl->AddRect(V(x0, y0), V(x1, y1), selected == n.id ? C::ACCENT : C::NODE_BORDER, float(8 * z), 0, 1.0f);
        draw_text_vc(dl, Fonts::medium, 13 * z, x0 + 12 * z, y0 + 15 * z, n.style.fg, n.title);
        if (n.gpu) {
            double cw = text_w(Fonts::mono, 11 * z, "GPU") + 12 * z, cx = x1 - 12 * z - cw;
            dl->AddRectFilled(V(cx, y0 + 7 * z), V(cx + cw, y0 + 23 * z), n.style.chip, float(4 * z));
            draw_text_vc(dl, Fonts::mono, 11 * z, cx + 6 * z, y0 + 15 * z, n.style.fg, "GPU");
        }
        for (size_t i = 0; i < n.rows.size(); ++i) {
            const Row& r = n.rows[i];
            double ry = y0 + (34 + 26.0 * i) * z, cy = ry + 13 * z;
            if (r.highlight) dl->AddRectFilled(V(x0 + 1, ry), V(x1 - 1, ry + 26 * z), C::ROW_HL);
            if (r.kind == RowKind::Param) {
                draw_text_vc(dl, Fonts::sans, 12 * z, x0 + 12 * z, cy, C::ROW, r.label);
                double vw = text_w(Fonts::mono, 12 * z, r.value);
                draw_text_vc(dl, Fonts::mono, 12 * z, x1 - 12 * z - vw, cy, C::TEXT, r.value);
            } else if (r.kind == RowKind::Out) {
                double lw = text_w(Fonts::sans, 12 * z, r.label);
                draw_text_vc(dl, Fonts::sans, 12 * z, x1 - 12 * z - lw, cy, C::ROW, r.label);
                port(dl, x1, cy, r.color);
            } else {
                draw_text_vc(dl, Fonts::sans, 12 * z, x0 + 12 * z, cy, C::ROW, r.label);
                port(dl, x0, cy, r.color, r.kind == RowKind::RW);
                if (r.kind == RowKind::RW) {
                    double tw = text_w(Fonts::mono, 11 * z, "rw") + 10 * z, tx = x1 - 12 * z - tw;
                    dl->AddRectFilled(V(tx, cy - 8 * z), V(tx + tw, cy + 8 * z), rgba("#2e3035"), float(3 * z));
                    draw_text_vc(dl, Fonts::mono, 11 * z, tx + 5 * z, cy, C::TEXT_2, "rw");
                }
            }
        }
    }

    void draw_edge(SDL* dl, const double* origin, const Node& n, App& app);

    void draw_legend(SDL* dl, double x, double y, double w, double h) const {
        const char* kinds[3] = {"edge", "dot", "diamond"};
        const char* labels[3] = {"data edge", "read / write", "in-place (rw)"};
        ImFont* f = Fonts::sans;
        const double s = 11.0;
        double widths[3], sum = 0;
        for (int i = 0; i < 3; ++i) { widths[i] = text_w(f, s, labels[i]) + 20; sum += widths[i]; }
        double lw = sum + 16 * 2 + 20, lx = x + w - 12 - lw, ly = y + h - 10 - 26;
        dl->AddRectFilled(V(lx, ly), V(lx + lw, ly + 26), C::CANVAS, 6);
        dl->AddRect(V(lx, ly), V(lx + lw, ly + 26), rgba("#2a2c31"), 6);
        double cx = lx + 10, cy = ly + 13;
        for (int i = 0; i < 3; ++i) {
            if (!std::strcmp(kinds[i], "edge")) {
                dl->AddRectFilled(V(cx, cy - 5), V(cx + 14, cy + 5), C::CARD, 3);
                dl->AddRect(V(cx, cy - 5), V(cx + 14, cy + 5), rgba("#5a5c62"), 3);
            } else if (!std::strcmp(kinds[i], "dot")) {
                dl->AddCircleFilled(V(cx + 4, cy), 4, C::TEXT_3);
            } else {
                ImVec2 d[4] = {V(cx + 4, cy - 5), V(cx + 9, cy), V(cx + 4, cy + 5), V(cx - 1, cy)};
                dl->AddConvexPolyFilled(d, 4, C::TEXT_3);
            }
            draw_text_vc(dl, f, s, cx + 20, cy, C::MUTED, labels[i]);
            cx += widths[i] + 16;
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════
//  Icons
// ════════════════════════════════════════════════════════════════════════════
static void icon_play(SDL* dl, double cx, double cy, ImU32 col, double s = 7.0) {
    dl->AddTriangleFilled(V(cx - s * 0.6, cy - s), V(cx + s, cy), V(cx - s * 0.6, cy + s), col);
}
static void icon_pause(SDL* dl, double cx, double cy, ImU32 col) {
    dl->AddLine(V(cx - 3.5, cy - 6), V(cx - 3.5, cy + 6), col, 2.0f);
    dl->AddLine(V(cx + 3.5, cy - 6), V(cx + 3.5, cy + 6), col, 2.0f);
}
static void icon_step(SDL* dl, double cx, double cy, ImU32 col) {
    dl->AddTriangle(V(cx - 5, cy - 6), V(cx + 3, cy), V(cx - 5, cy + 6), col, 1.8f);
    dl->AddLine(V(cx + 5.5, cy - 6), V(cx + 5.5, cy + 6), col, 2.0f);
}
static void icon_stop(SDL* dl, double cx, double cy, ImU32 col) {
    dl->AddRect(V(cx - 5, cy - 5), V(cx + 5, cy + 5), col, 1.0f, 0, 1.8f);
}
static void icon_viewer(SDL* dl, double cx, double cy, ImU32 col) {
    dl->AddRect(V(cx - 7, cy - 6), V(cx + 7, cy + 6), col, 2.0f, 0, 1.5f);
    dl->AddCircle(V(cx, cy), 3.2f, col, 0, 1.5f);
}
static void icon_graph(SDL* dl, double cx, double cy, ImU32 col) {
    dl->AddRect(V(cx - 8, cy - 6), V(cx - 2.5, cy - 1.5), col, 1.2f, 0, 1.4f);
    dl->AddRect(V(cx + 2.5, cy + 1.5), V(cx + 8, cy + 6), col, 1.2f, 0, 1.4f);
    dl->AddBezierCubic(V(cx - 2.5, cy - 3.8), V(cx + 1, cy - 3.8), V(cx - 1, cy + 3.8), V(cx + 2.5, cy + 3.8), col,
                       1.4f);
}
static void polyline3(SDL* dl, ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col, float t) {
    ImVec2 p[3] = {a, b, c};
    dl->AddPolyline(p, 3, col, 0, t);
}
static void icon_flame(SDL* dl, double cx, double cy, ImU32 col) {  // three stacked bars: a flamegraph
    dl->AddRectFilled(V(cx - 8, cy + 3), V(cx + 8, cy + 6.5), col, 1.0f);
    dl->AddRectFilled(V(cx - 8, cy - 1.5), V(cx + 3, cy + 2), col, 1.0f);
    dl->AddRectFilled(V(cx - 8, cy - 6), V(cx - 2, cy - 2.5), col, 1.0f);
}
static void icon_code(SDL* dl, double cx, double cy, ImU32 col) {
    polyline3(dl, V(cx - 3.5, cy - 5), V(cx - 8, cy), V(cx - 3.5, cy + 5), col, 1.5f);
    polyline3(dl, V(cx + 3.5, cy - 5), V(cx + 8, cy), V(cx + 3.5, cy + 5), col, 1.5f);
    dl->AddLine(V(cx + 1.5, cy - 6), V(cx - 1.5, cy + 6), col, 1.5f);
}
static void icon_layout(SDL* dl, double cx, double cy, ImU32 col, int lay) {
    const double x0 = cx - 9, y0 = cy - 7, x1 = cx + 9, y1 = cy + 7;
    auto L = [&](double ax, double ay, double bx, double by) { dl->AddLine(V(ax, ay), V(bx, by), col, 1.3f); };
    if (lay == 0) {  // stack: one pane in front of the others
        dl->AddRect(V(x0 + 4, y0), V(x1, y1 - 4), col, 1.5f, 0, 1.1f);
        dl->AddRectFilled(V(x0, y0 + 4), V(x1 - 4, y1), C::BUTTON, 1.5f);
        dl->AddRect(V(x0, y0 + 4), V(x1 - 4, y1), col, 1.5f, 0, 1.3f);
        return;
    }
    dl->AddRect(V(x0, y0), V(x1, y1), col, 1.5f, 0, 1.3f);
    switch (lay) {
        case 1: L(x0 + 10, y0, x0 + 10, y1); L(x0 + 10, cy, x1, cy); break;                  // tall
        case 2: L(x0, y0 + 8, x1, y0 + 8); L(cx, y0 + 8, cx, y1); break;                    // fat
        case 3: L(cx, y0, cx, y1); L(x0, cy, x1, cy); break;                                // grid
        case 4: L(x0 + 6, y0, x0 + 6, y1); L(x0 + 12, y0, x0 + 12, y1); break;              // horizontal
        case 5: L(x0, y0 + 4.7, x1, y0 + 4.7); L(x0, y0 + 9.3, x1, y0 + 9.3); break;        // vertical
        default: L(x0 + 7, y0, x0 + 7, y1); L(x0 + 7, cy - 1, x1, cy - 1); L(x0 + 12.5, cy - 1, x0 + 12.5, y1);  // splits
    }
}
static void icon_chevron(SDL* dl, double cx, double cy, ImU32 col) {
    polyline3(dl, V(cx - 3.5, cy - 1.5), V(cx, cy + 2), V(cx + 3.5, cy - 1.5), col, 1.6f);
}
static void icon_gear(SDL* dl, double cx, double cy, ImU32 col) {
    dl->AddCircle(V(cx, cy), 3.2f, col, 0, 1.5f);
    for (int k = 0; k < 8; ++k) {
        double a = k * PI / 4;
        dl->AddLine(V(cx + std::cos(a) * 5.5, cy + std::sin(a) * 5.5), V(cx + std::cos(a) * 8, cy + std::sin(a) * 8),
                    col, 1.5f);
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Formatting helpers (match Python's format specs)
// ════════════════════════════════════════════════════════════════════════════
static std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}
static std::string fmt_thousands(long long v) {
    std::string s = std::to_string(v), out;
    int n = int(s.size());
    for (int i = 0; i < n; ++i) {
        out += s[i];
        if ((n - i - 1) % 3 == 0 && i != n - 1) out += ',';
    }
    return out;
}
static std::string strip_zeros(std::string s) {
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

static const char* RUN_SCRIPT = R"(from hydro import Graph, nodes as n, data as d

g      = Graph("sedov_tracers")
mesh   = g.edge(d.Mesh(cells=(256, 256, 256)))
state  = g.edge(d.FluidState(fields=("rho", "v", "p")))
trc    = g.edge(d.Particles())
diag   = g.edge(d.Scalars(("E_tot", "dt", "cells_per_s")))

g.node(n.SedovInit(E0=1.0, n_tracers=1_200_000), out=[mesh, state, trc])
g.node(n.HydroStep(riemann="hllc", cfl=0.40), inp=[mesh], rw=[state], out=[diag])
g.node(n.AdvectTracers(scheme="rk2"), inp=[state], rw=[trc])

g.run(until=0.05)
)";

// ════════════════════════════════════════════════════════════════════════════
//  Application
// ════════════════════════════════════════════════════════════════════════════
// Kitty-style layouts. The dockspace is the single "tab"; every layout except Splits is recomputed from
// the ordered list of visible panes, Splits is the free-form tree the user builds by dragging.
enum class Lay { Stack, Tall, Fat, Grid, Horizontal, Vertical, Splits };
static const char* LAY_ID[7] = {"stack", "tall", "fat", "grid", "horizontal", "vertical", "splits"};
static const char* LAY_NAME[7] = {"Stack", "Tall", "Fat", "Grid", "Horizontal", "Vertical", "Splits"};
static Lay parse_lay(const std::string& s) {
    for (int i = 0; i < 7; ++i)
        if (s == LAY_ID[i]) return Lay(i);
    if (s == "columns") return Lay::Horizontal;  // names of the old presets
    if (s == "rows") return Lay::Fat;
    return Lay::Tall;
}
static constexpr double TOP_H = 52.0, STATUS_H = 28.0, PANE_HDR = 36.0;
struct PaneRect { double x, y, w, h; };

struct App {
    DemoSimulation sim;
    GraphView graph;
    // layout state
    Lay lay = Lay::Tall, lay_before_stack = Lay::Tall;
    float bias = 0.6f;         // Tall / Fat: share of the main area
    int full_size = 1;         // Tall / Fat: panes in the main area
    bool mirrored = false;     // Tall: main on the right, Fat: main at the bottom
    std::vector<char> order{'g', 'v', 's', 'f'};  // pane order; the first panes are the "main" ones
    char focused = 'g', pending_focus = 0;
    int place = 0;             // new panes: 0 after focused, 1 before focused, 2 first, 3 last
    int split_axis = 0;        // Splits, new panes: 0 auto (longer side), 1 side by side, 2 stacked
    ImGuiID dock_id_ = 0;
    ImVec2 body_px_{};
    std::string built_sig;
    bool sig_pending = false, profile_from_cli = false;
    std::map<char, bool> visible{{'v', true}, {'g', true}, {'s', true}, {'f', false}};  // profile pane off by default
    int field = 0;
    bool show_tracers = true, view3d = true;
    double yaw = -38 * PI / 180, pitch = 24 * PI / 180;
    TextEditor editor;
    GLTexture tex_main, tex_card_state, tex_card_tracers, logo;
    bool deterministic;
    long long frame = 0;
    int frames_left, bench_frames;
    double ui_scale = 1.0;  // applied at the start of each frame
    bool style_applied = false, first_frame = true, need_layout = true, layout_from_cli = false;
    std::map<char, bool> was_visible{{'v', true}, {'g', true}, {'s', true}, {'f', false}};
    // profile pane
    bool prof_live = true;
    int prof_focus = 0;
    std::vector<double> prof_snap;
    double next_prof = 0;
    bool screenshot, want_exit = false;
    std::map<std::string, std::vector<double>> timings{{"update", {}}, {"ui", {}}, {"frame", {}}};
    double prev_frame_start = -1, last_time = 0;
    double next_main = 0, next_cards = 0;
    double main_lo = 0.02, main_hi = 4.0, bytes_per_s = 0;
    std::vector<std::array<double, 2>> tracer_pts;

    App(std::string layout_, bool screenshot_, int frames, int bench)
        : graph(build_demo_graph().first, build_demo_graph().second), lay(parse_lay(layout_)),
          deterministic(screenshot_ || bench > 0), frames_left(frames), bench_frames(bench), screenshot(screenshot_) {
        last_time = now();
    }

    static double wall() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }
    double now() const { return deterministic ? double(frame) / 60.0 : wall(); }

    void post_init() {
        tex_main.create(512, 512);
        tex_card_state.create(136, 88);
        tex_card_tracers.create(136, 88);
        int lw, lh, ch;
        unsigned char* px = stbi_load((g_assets / "shamrock_logo.png").string().c_str(), &lw, &lh, &ch, 4);
        logo.create(lw, lh, px);
        stbi_image_free(px);
        editor.SetLanguage(TextEditor::Language::Python());
        editor.SetText(RUN_SCRIPT);
        editor.SetTabSize(4);
        editor.SetShowLineNumbersEnabled(true);
        editor.SetLineSpacing(1.15f);
        editor.SetShowWhitespacesEnabled(false);
        // Same built-in dark palette as the Python app; in C++ you can customise it with
        // editor.SetPalette(p) where p is a copy of TextEditor::GetDarkPalette().
        refresh_previews(true);
    }

    static inline ImGuiStyle base_style;

    static void setup_style() {
        ImGuiStyle& style = ImGui::GetStyle();
        style.FramePadding = ImVec2(12, 6);       // dock tab height = font + 2 * 6
        style.TabRounding = 0;
        style.TabBarBorderSize = 1;
        style.TabBarOverlineSize = 2;
        style.TabBorderSize = 0;
        style.DockingSeparatorSize = 1;
        style.WindowMenuButtonPosition = ImGuiDir_None;
        style.TabCloseButtonMinWidthSelected = 0;  // close cross only when hovered
        style.TabCloseButtonMinWidthUnselected = 0;
        style.WindowPadding = ImVec2(0, 0);
        style.WindowBorderSize = 0;
        style.ChildBorderSize = 0;
        style.WindowRounding = 0;
        style.ScrollbarSize = 10;
        style.ScrollbarRounding = 4;
        style.ItemSpacing = ImVec2(0, 0);
        const std::pair<ImGuiCol, ImU32> cols[] = {
            {ImGuiCol_WindowBg, C::APP_BG}, {ImGuiCol_ChildBg, C::CANVAS}, {ImGuiCol_ScrollbarBg, C::CANVAS},
            {ImGuiCol_ScrollbarGrab, C::BORDER}, {ImGuiCol_ScrollbarGrabHovered, C::NODE_BORDER},
            {ImGuiCol_ScrollbarGrabActive, C::MUTED}, {ImGuiCol_Text, C::TEXT}, {ImGuiCol_PopupBg, C::PANEL},
            {ImGuiCol_Border, C::BORDER}, {ImGuiCol_TextSelectedBg, rgba("#e8a33d", 0.25)},
            // docking: tab bars, drop preview, dividers
            {ImGuiCol_TitleBg, C::PANEL}, {ImGuiCol_TitleBgActive, C::PANEL}, {ImGuiCol_TitleBgCollapsed, C::PANEL},
            {ImGuiCol_Tab, C::PANEL}, {ImGuiCol_TabHovered, C::BUTTON}, {ImGuiCol_TabSelected, C::CANVAS},
            {ImGuiCol_TabSelectedOverline, C::ACCENT}, {ImGuiCol_TabDimmed, C::PANEL},
            {ImGuiCol_TabDimmedSelected, C::CANVAS}, {ImGuiCol_TabDimmedSelectedOverline, rgba("#e8a33d", 0.35)},
            {ImGuiCol_DockingPreview, rgba("#e8a33d", 0.30)}, {ImGuiCol_DockingEmptyBg, C::CANVAS},
            {ImGuiCol_Separator, C::DIVIDER}, {ImGuiCol_SeparatorHovered, rgba("#e8a33d", 0.6)},
            {ImGuiCol_SeparatorActive, C::ACCENT}, {ImGuiCol_Button, 0}, {ImGuiCol_ButtonHovered, C::ROW_HL},
            {ImGuiCol_ButtonActive, C::ACCENT_BG}, {ImGuiCol_FrameBg, C::BUTTON},
        };
        for (auto& [k, v] : cols) style.Colors[k] = ImGui::ColorConvertU32ToFloat4(v);
        base_style = style;
    }

    // ImGui's own widgets (dock tabs, dividers, drop overlay, scrollbars) follow the UI scale too.
    static void apply_scale(double s) {
        ImGuiStyle& style = ImGui::GetStyle();
        style = base_style;
        style.ScaleAllSizes(float(s));
        style.FontSizeBase = float(13 * s);
        style.DockingSeparatorSize = std::max(1.0f, float(s));
    }

    // --- data --------------------------------------------------------------
    void refresh_previews(bool force = false) {
        double t = now();
        double sent = 0;
        if (force || t >= next_main) {  // main pane: 10 Hz
            Image img = sim.slice(field, 512, 512, &main_lo, &main_hi);
            tex_main.upload(img.px);
            tracer_pts = sim.tracers_xy(520);
            next_main = t + 0.1;
            sent += 512 * 512 * 10;
        }
        if (force || t >= next_cards) {  // edge cards: 4 Hz
            if (graph.previews_on) {
                tex_card_state.upload(sim.slice(0, 136, 88).px);
                tex_card_tracers.upload(sim.tracers_image(136, 88).px);
            }
            next_cards = t + 0.25;
            sent += 2 * 136 * 88 * 4;
        }
        if (visible['f'] && prof_live && (force || t >= next_prof || prof_snap.empty())) {  // profile: 2 Hz
            prof_snap.resize(sim.prof.size());
            for (size_t i = 0; i < sim.prof.size(); ++i) prof_snap[i] = sim.prof[i].ema;
            next_prof = t + 0.5;
        }
        if (sent != 0) bytes_per_s = bytes_per_s != 0 ? 0.8 * bytes_per_s + 0.2 * sent : sent;
    }

    // --- frame -------------------------------------------------------------
    // Called before ImGui::NewFrame(): ImGui reads the base font size (used by dock tabs) at NewFrame.
    void pre_frame() {
        if (UI::scale != ui_scale || !style_applied) {
            UI::scale = ui_scale;
            apply_scale(UI::scale);
            style_applied = true;
        }
    }

    void gui() {
        double t0 = wall();
        double t = now();
        sim.advance(deterministic ? 1.0 / 60.0 : t - last_time);
        last_time = t;
        refresh_previews();
        double t1 = wall();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        UI::origin = vp->Pos;
        g_sdl_next = 0;
        ImGui::SetNextWindowPos(vp->Pos);
        ImGui::SetNextWindowSize(vp->Size);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                 ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::Begin("##shamrock_main", nullptr, flags);
        double W = vp->Size.x / UI::scale, H = vp->Size.y / UI::scale, X = vp->Pos.x, Y = vp->Pos.y;  // logical
        SDL* dl = window_draw_list();

        PaneRect body{X, Y + TOP_H, W, H - TOP_H - STATUS_H};
        const ImGuiID dock_id = ImGui::GetID("##body_dockspace");
        const ImVec2 body_px = V(body.w * UI::scale, body.h * UI::scale);
        dock_id_ = dock_id;
        body_px_ = body_px;
        if (first_frame) {  // Splits keeps its saved tree; the other layouts are recomputed from the pane list
            ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_id);
            need_layout = layout_from_cli || lay != Lay::Splits || !node || !node->IsSplitNode();
            first_frame = false;
        }
        handle_shortcuts();
        top_bar(dl, X, Y, W);
        apply_pending_tree();
        on_visibility_changes();
        if (need_layout) {
            build_layout(lay == Lay::Splits ? Lay::Tall : lay);  // a fresh Splits tree starts from Tall
            need_layout = false;
        }
        bool any = visible['v'] || visible['g'] || visible['s'] || visible['f'];
        drop_central_node(dock_id);
        dl->AddRectFilled(V(body.x, body.y), V(body.x + body.w, body.y + body.h), C::CANVAS);
        set_cursor(V(body.x, body.y));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, C::DIVIDER);  // idle divider colour (docking uses WindowBg)
        ImGui::DockSpace(dock_id, body_px, any ? ImGuiDockNodeFlags_None : ImGuiDockNodeFlags_KeepAliveOnly);
        ImGui::PopStyleColor();
        if (!any) {
            std::string msg = "All panes are hidden. Turn one back on from the toggles in the top bar.";
            double mw = text_w(Fonts::sans, 14, msg);
            draw_text_vc(dl, Fonts::sans, 14, body.x + (body.w - mw) / 2, body.y + body.h / 2, C::MUTED, msg);
        }
        status_bar(dl, X, Y + H - STATUS_H, W);

        set_cursor(V(X, Y));
        ImGui::Dummy(ImVec2(0, 0));
        ImGui::End();

        for (char key : {'v', 'g', 's', 'f'})
            if (visible[key]) dock_pane(key);
        for (char key : {'v', 'g', 's', 'f'}) was_visible[key] = visible[key];
        after_panes();

        double t2 = wall();
        if (bench_frames) {
            timings["update"].push_back(t1 - t0);
            timings["ui"].push_back(t2 - t1);
            if (prev_frame_start >= 0) timings["frame"].push_back(t0 - prev_frame_start);
            prev_frame_start = t0;
        }
        frame += 1;
        if (screenshot || bench_frames)
            if (--frames_left <= 0) want_exit = true;
    }

    // --- top bar -----------------------------------------------------------
    void top_bar(SDL* dl, double X, double Y, double W) {
        dl->AddRectFilled(V(X, Y), V(X + W, Y + TOP_H), C::PANEL);
        dl->AddLine(V(X, Y + TOP_H - 0.5), V(X + W, Y + TOP_H - 0.5), C::DIVIDER);
        double cy = Y + TOP_H / 2;

        double lx = X + 16, logo_h = 34.0, logo_w = logo_h * 298 / 96;
        dl->AddImage(logo.ref, V(lx, cy - logo_h / 2), V(lx + logo_w, cy + logo_h / 2));
        const std::string script = "run_sedov.py";
        double name_w = 12 + 13 + text_w(Fonts::mono, 12, script);

        // measure everything, then drop the least important parts until it fits the (logical) width
        std::string pill_name = "gpu-node-042", pill_via = "ssh";
        std::string layout_label = LAY_NAME[int(lay)];
        double group_w = 2 + 32 * 4 + 2 * 3 + 2 + 2;
        double scale_w = 24 + text_w(Fonts::mono, 12, "000%");
        std::pair<std::string, std::string> readouts[3] = {
            {"step", fmt_thousands(sim.step)}, {"t", fmt("%.3e", sim.t)}, {"dt", fmt("%.2e", sim.dt())}};
        std::string run_label = sim.running ? "Running" : "Paused";
        double run_w = 14 + 14 + 8 + text_w(Fonts::semibold, 13, run_label) + 14;
        double ro_w = 18 * 2;
        for (auto& [k, v] : readouts) ro_w += text_w(Fonts::mono, 12, k + " " + v);
        double buttons_w = run_w + 6 + 36 * 3 + 6 * 2;
        // levels: (show readouts, compact layout button + pill, show script name)
        const bool levels[4][3] = {{true, false, true}, {false, false, true}, {false, true, true}, {false, true, false}};
        bool show_ro = true, compact = false, show_name = true;
        double pill_w = 0, lay_w = 0, right_w = 0, rx = 0, left_end = 0, centre_w = 0;
        for (auto& lv : levels) {
            show_ro = lv[0]; compact = lv[1]; show_name = lv[2];
            pill_w = 12 + 8 + 8 + text_w(Fonts::mono, 12, pill_name) + 12;
            lay_w = 12 + 18 + 8 + 12 + 12;
            if (!compact) {
                pill_w += 8 + text_w(Fonts::sans, 12, pill_via);
                lay_w += text_w(Fonts::sans, 12, layout_label) + 8;
            }
            right_w = scale_w + 10 + group_w + 10 + lay_w + 10 + pill_w + 10 + 36;
            rx = X + W - 16 - right_w;
            left_end = lx + logo_w + (show_name ? name_w : 0);
            centre_w = buttons_w + (show_ro ? 16 + ro_w : 0);
            if (left_end + 24 + centre_w + 24 <= rx) break;
        }
        if (show_name) {
            double nx = lx + logo_w + 12;
            dl->AddLine(V(nx, cy - 10), V(nx, cy + 10), C::BORDER);
            draw_text_vc(dl, Fonts::mono, 12, nx + 13, cy, C::TEXT_3, script);
        }

        double cx = std::max(left_end + 24, left_end + (rx - left_end - centre_w) / 2);

        Hit r = hit("##run", cx, cy - 18, run_w, 36);
        dl->AddRectFilled(V(cx, cy - 18), V(cx + run_w, cy + 18), r.hovered ? lighten(C::ACCENT, 10) : C::ACCENT, 6);
        if (sim.running) icon_play(dl, cx + 14 + 6, cy, C::ON_ACCENT, 6);
        else icon_pause(dl, cx + 14 + 7, cy, C::ON_ACCENT);
        draw_text_vc(dl, Fonts::semibold, 13, cx + 14 + 14 + 8, cy, C::ON_ACCENT, run_label);
        if (r.clicked) sim.running = !sim.running;
        double bx = cx + run_w + 6;
        struct Ctl { const char* name; void (*icon)(SDL*, double, double, ImU32); };
        const Ctl ctls[3] = {{"Pause", icon_pause}, {"Step once", icon_step}, {"Stop", icon_stop}};
        for (const Ctl& c : ctls) {
            std::string id = std::string("##") + c.name;
            if (framed_button(dl, id.c_str(), bx, cy - 18, 36, 36, C::BUTTON, C::BORDER, 6)) {
                sim.running = false;
                if (!std::strcmp(c.name, "Step once")) { sim.step += 1; sim.t += sim.dt(); }
            }
            if (ImGui::IsItemHovered()) tooltip(c.name);
            c.icon(dl, bx + 18, cy, C::TEXT);
            bx += 36 + 6;
        }
        double tx = bx + 10;
        for (auto& [k, v] : readouts) {
            if (!show_ro) break;
            draw_text_vc(dl, Fonts::mono, 12, tx, cy, C::TEXT_3, k + " ");
            double kw = text_w(Fonts::mono, 12, k + " ");
            draw_text_vc(dl, Fonts::mono, 12, tx + kw, cy, C::TEXT, v);
            tx += text_w(Fonts::mono, 12, k + " " + v) + 18;
        }

        scale_control(dl, rx, cy, scale_w);
        double gx = rx + scale_w + 10;
        dl->AddRectFilled(V(gx, cy - 18), V(gx + group_w, cy + 18), C::CANVAS, 8);
        dl->AddRect(V(gx + 0.5, cy - 17.5), V(gx + group_w - 0.5, cy + 17.5), C::BORDER, 8);
        double px = gx + 3;
        struct Tog { char key; void (*icon)(SDL*, double, double, ImU32); const char* tip; };
        const Tog togs[4] = {{'v', icon_viewer, "viewer pane"}, {'g', icon_graph, "graph pane"},
                             {'s', icon_code, "script pane"}, {'f', icon_flame, "profile pane"}};
        for (const Tog& tg : togs) {
            bool on = visible[tg.key];
            std::string id = std::string("##toggle_") + tg.key;
            Hit h = hit(id.c_str(), px, cy - 16, 32, 32);
            if (on) dl->AddRectFilled(V(px, cy - 16), V(px + 32, cy + 16), C::ACCENT_BG, 6);
            else if (h.hovered) dl->AddRectFilled(V(px, cy - 16), V(px + 32, cy + 16), C::BUTTON, 6);
            tg.icon(dl, px + 16, cy, on ? C::ACCENT : C::DIM);
            if (h.hovered) tooltip((std::string(on ? "Hide " : "Show ") + tg.tip).c_str());
            if (h.clicked) visible[tg.key] = !on;
            px += 32 + 2;
        }
        double lx2 = gx + group_w + 10;
        if (framed_button(dl, "##layout", lx2, cy - 18, lay_w, 36, C::BUTTON, C::BORDER, 6))
            ImGui::OpenPopup("##layout_menu");
        if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen("##layout_menu")) tooltip("Layout  ·  ctrl+shift+L cycles");
        layout_menu(lx2, cy + 22);
        icon_layout(dl, lx2 + 12 + 9, cy, C::TEXT, int(lay));
        if (!compact) draw_text_vc(dl, Fonts::sans, 12, lx2 + 12 + 18 + 8, cy, C::TEXT, layout_label);
        icon_chevron(dl, lx2 + lay_w - 12 - 6, cy, C::MUTED);
        double ppx = lx2 + lay_w + 10;
        dl->AddRectFilled(V(ppx, cy - 16), V(ppx + pill_w, cy + 16), C::PILL_BG, 16);
        dl->AddRect(V(ppx + 0.5, cy - 15.5), V(ppx + pill_w - 0.5, cy + 15.5), C::PILL_BORDER, 16);
        dl->AddCircleFilled(V(ppx + 16, cy), 4, C::TEAL);
        draw_text_vc(dl, Fonts::mono, 12, ppx + 28, cy, C::PILL_TEXT, pill_name);
        if (!compact)
            draw_text_vc(dl, Fonts::sans, 12, ppx + 28 + text_w(Fonts::mono, 12, pill_name) + 8, cy, C::TEAL_TEXT,
                         pill_via);
        double sx = ppx + pill_w + 10;
        framed_button(dl, "##settings", sx, cy - 18, 36, 36, C::BUTTON, C::BORDER, 6);
        if (ImGui::IsItemHovered()) tooltip("Connection settings");
        icon_gear(dl, sx + 18, cy, C::TEXT_3);
    }

    // Rounded percentage box: scroll over it to change the UI scale, click to reset to 100 %.
    void scale_control(SDL* dl, double x, double cy, double w) {
        const double h = 28.0;
        Hit r = hit("##ui_scale", x, cy - h / 2, w, h);
        dl->AddRectFilled(V(x, cy - h / 2), V(x + w, cy + h / 2), r.hovered ? C::BUTTON : C::CANVAS, float(h / 2));
        dl->AddRect(V(x + 0.5, cy - h / 2 + 0.5), V(x + w - 0.5, cy + h / 2 - 0.5), r.hovered ? C::ACCENT : C::BORDER,
                    float(h / 2));
        std::string label = std::to_string(int(std::nearbyint(UI::scale * 100))) + "%";
        draw_text_vc(dl, Fonts::mono, 12, x + (w - text_w(Fonts::mono, 12, label)) / 2, cy,
                     r.hovered ? C::TEXT : C::TEXT_3, label);
        if (r.hovered) {
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                double steps = std::nearbyint(UI::scale / UI::STEP) + (wheel > 0 ? 1 : -1);
                ui_scale = std::min(UI::MAX, std::max(UI::MIN, steps * UI::STEP));
            }
            tooltip("UI scale · scroll to change · click to reset");
        }
        if (r.clicked) ui_scale = 1.0;
    }

    // --- layout ------------------------------------------------------------
    // --- docking ---------------------------------------------------------
    static const char* pane_window(char key) {
        return key == 'v' ? "Viewer###pane_v" : key == 'g' ? "Graph###pane_g" : key == 's' ? "Script###pane_s"
                                                                                  : "Profile###pane_f";
    }

    std::vector<char> visible_order() {
        std::vector<char> v;
        for (char k : order)
            if (visible[k]) v.push_back(k);
        return v;
    }
    static char key_of(const ImGuiWindow* w) {
        for (char k : {'v', 'g', 's', 'f'})
            if (!std::strcmp(w->Name, pane_window(k))) return k;
        return 0;
    }

    // Computed layouts. Hidden panes are left out; showing one rebuilds with it included.
    void build_layout(Lay l) {
        const ImGuiID dock_id = dock_id_;
        std::vector<char> panes = visible_order();
        const int n = int(panes.size());
        undock_hidden();
        ImGui::DockBuilderRemoveNode(dock_id);
        ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dock_id, body_px_);
        auto dock = [&](char k, ImGuiID id) { ImGui::DockBuilderDockWindow(pane_window(k), id); };
        auto slots = [&](ImGuiID node, int k, ImGuiAxis axis) {  // k equal slots along an axis
            std::vector<ImGuiID> ids;
            for (int i = 0; i < k - 1; ++i)
                ids.push_back(ImGui::DockBuilderSplitNode(node, axis == ImGuiAxis_X ? ImGuiDir_Left : ImGuiDir_Up,
                                                          1.0f / float(k - i), nullptr, &node));
            ids.push_back(node);
            return ids;
        };
        auto row = [&](ImGuiID node, const std::vector<char>& ps, ImGuiAxis axis) {
            auto ids = slots(node, int(ps.size()), axis);
            for (size_t i = 0; i < ps.size(); ++i) dock(ps[i], ids[i]);
        };
        if (n == 0) {
        } else if (n == 1 || l == Lay::Stack) {  // stack: all panes as tabs of one area, the focused one in front
            for (char k : panes) dock(k, dock_id);
            pending_focus = visible[focused] ? focused : panes[0];
        } else if (l == Lay::Horizontal) {
            row(dock_id, panes, ImGuiAxis_X);
        } else if (l == Lay::Vertical) {
            row(dock_id, panes, ImGuiAxis_Y);
        } else if (l == Lay::Tall || l == Lay::Fat) {
            const int m = std::clamp(full_size, 1, n - 1);
            std::vector<char> mains(panes.begin(), panes.begin() + m), rest(panes.begin() + m, panes.end());
            ImGuiDir dir = l == Lay::Tall ? (mirrored ? ImGuiDir_Right : ImGuiDir_Left)
                                          : (mirrored ? ImGuiDir_Down : ImGuiDir_Up);
            ImGuiID other = 0, main_id = ImGui::DockBuilderSplitNode(dock_id, dir, bias, nullptr, &other);
            const ImGuiAxis along = l == Lay::Tall ? ImGuiAxis_Y : ImGuiAxis_X;
            row(main_id, mains, along);
            row(other, rest, along);
        } else {  // grid: about sqrt(n) columns, filled column by column, counts differ by at most one
            const int cols = int(std::ceil(std::sqrt(double(n))));
            auto col_ids = slots(dock_id, cols, ImGuiAxis_X);
            const int base = n / cols, extra = n % cols;
            int k = 0;
            for (int c = 0; c < cols; ++c) {
                int cnt = base + (c < extra ? 1 : 0);
                row(col_ids[c], std::vector<char>(panes.begin() + k, panes.begin() + k + cnt), ImGuiAxis_Y);
                k += cnt;
            }
        }
        ImGui::DockBuilderFinish(dock_id);
        sig_pending = true;
    }

    // --- Splits: read the live dock tree into a small model, edit it, write it back -------------
    struct SNode {
        bool leaf = true;
        ImGuiAxis axis = ImGuiAxis_X;
        float ratio = 0.5f;
        std::vector<char> panes;
        std::unique_ptr<SNode> a, b;
    };
    std::unique_ptr<SNode> saved_splits;  // the Splits tree kept while zoomed into Stack

    static std::unique_ptr<SNode> read_tree(const ImGuiDockNode* n) {
        auto s = std::make_unique<SNode>();
        if (!n) return s;
        if (n->IsSplitNode()) {
            s->leaf = false;
            s->axis = n->SplitAxis;
            float a0 = n->ChildNodes[0]->Size[n->SplitAxis], a1 = n->ChildNodes[1]->Size[n->SplitAxis];
            s->ratio = a0 / std::max(1.0f, a0 + a1);
            s->a = read_tree(n->ChildNodes[0]);
            s->b = read_tree(n->ChildNodes[1]);
        } else {
            for (const ImGuiWindow* w : n->Windows)
                if (char k = key_of(w)) s->panes.push_back(k);
        }
        return s;
    }
    static void write_tree(const SNode& s, ImGuiID node) {
        if (s.leaf) {
            for (char k : s.panes) ImGui::DockBuilderDockWindow(pane_window(k), node);
            return;
        }
        ImGuiID second = 0;
        ImGuiID first = ImGui::DockBuilderSplitNode(node, s.axis == ImGuiAxis_X ? ImGuiDir_Left : ImGuiDir_Up,
                                                    std::clamp(s.ratio, 0.05f, 0.95f), nullptr, &second);
        write_tree(*s.a, first);
        write_tree(*s.b, second);
    }
    // Tree edits are applied in the next frame, from inside the main window (the builder needs the host
    // window current; the layout menu runs inside its own popup window).
    std::unique_ptr<SNode> pending_tree;
    // Empty areas (their pane is hidden) are dropped; a hidden pane shown later splits the focused area.
    static void prune(std::unique_ptr<SNode>& s) {
        if (s->leaf) return;
        prune(s->a);
        prune(s->b);
        if (s->a->leaf && s->a->panes.empty()) s = std::move(s->b);
        else if (s->b->leaf && s->b->panes.empty()) s = std::move(s->a);
    }
    void rebuild_splits(std::unique_ptr<SNode> root) {
        prune(root);
        pending_tree = std::move(root);
    }
    // Hidden panes are undocked before a rebuild: node ids are reused by the builder, so an old id could
    // put a pane shown later into an unrelated area.
    void undock_hidden() {
        for (char k : {'v', 'g', 's', 'f'})
            if (!visible[k] && ImGui::FindWindowByName(pane_window(k))) ImGui::DockBuilderDockWindow(pane_window(k), 0);
    }
    void apply_pending_tree() {
        if (!pending_tree) return;
        undock_hidden();
        ImGui::DockBuilderRemoveNode(dock_id_);
        ImGui::DockBuilderAddNode(dock_id_, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dock_id_, body_px_);
        write_tree(*pending_tree, dock_id_);
        ImGui::DockBuilderFinish(dock_id_);
        pending_tree.reset();
        sig_pending = true;
        need_layout = false;  // the edited tree wins over a recompute queued in the same frame
    }
    static SNode* find_parent_of(SNode* s, char k, SNode* parent = nullptr) {
        if (s->leaf) return std::find(s->panes.begin(), s->panes.end(), k) != s->panes.end() ? parent : nullptr;
        if (SNode* r = find_parent_of(s->a.get(), k, s)) return r;
        return find_parent_of(s->b.get(), k, s);
    }
    static void remove_pane(std::unique_ptr<SNode>& s, char k) {  // drops k; collapses splits left with one side
        if (s->leaf) {
            s->panes.erase(std::remove(s->panes.begin(), s->panes.end(), k), s->panes.end());
            return;
        }
        remove_pane(s->a, k);
        remove_pane(s->b, k);
        if (s->a->leaf && s->a->panes.empty()) s = std::move(s->b);
        else if (s->b->leaf && s->b->panes.empty()) s = std::move(s->a);
    }
    void rotate_split() {  // flip the axis of the split that holds the focused pane
        auto root = read_tree(ImGui::DockBuilderGetNode(dock_id_));
        prune(root);  // first, so the split being flipped is one that stays
        if (SNode* p = find_parent_of(root.get(), focused)) {
            p->axis = p->axis == ImGuiAxis_X ? ImGuiAxis_Y : ImGuiAxis_X;
            set_layout(Lay::Splits);
            rebuild_splits(std::move(root));
        }
    }
    void move_to_edge(ImGuiDir dir) {  // the focused pane spans the whole width or height at that edge
        auto root = read_tree(ImGui::DockBuilderGetNode(dock_id_));
        if (root->leaf && root->panes.size() <= 1) return;
        remove_pane(root, focused);
        auto leaf = std::make_unique<SNode>();
        leaf->panes = {focused};
        auto top = std::make_unique<SNode>();
        top->leaf = false;
        top->axis = (dir == ImGuiDir_Left || dir == ImGuiDir_Right) ? ImGuiAxis_X : ImGuiAxis_Y;
        bool first = dir == ImGuiDir_Left || dir == ImGuiDir_Up;
        top->ratio = first ? 0.3f : 0.7f;
        top->a = first ? std::move(leaf) : std::move(root);
        top->b = first ? std::move(root) : std::move(leaf);
        lay = Lay::Splits;
        rebuild_splits(std::move(top));
        ImGui::MarkIniSettingsDirty();
    }

    // --- layout switching and pane order --------------------------------------------------------
    void set_layout(Lay l) {
        if (l == lay) return;
        if (l == Lay::Stack) {
            lay_before_stack = lay;
            if (lay == Lay::Splits) saved_splits = read_tree(ImGui::DockBuilderGetNode(dock_id_));
        }
        const Lay prev = lay;
        lay = l;
        if (l == Lay::Splits) {
            if (prev == Lay::Stack && saved_splits) rebuild_splits(std::move(saved_splits));  // back from the zoom
            saved_splits.reset();                                                             // else keep the tree
        } else {
            need_layout = true;
        }
        ImGui::MarkIniSettingsDirty();
    }
    void toggle_stack() { set_layout(lay == Lay::Stack ? lay_before_stack : Lay::Stack); }
    void cycle_layout() {
        static const Lay cycle[7] = {Lay::Tall, Lay::Fat, Lay::Grid, Lay::Horizontal, Lay::Vertical, Lay::Splits, Lay::Stack};
        int i = 0;
        while (cycle[i] != lay) ++i;
        set_layout(cycle[(i + 1) % 7]);
    }
    void cycle_focus(int d) {
        auto v = visible_order();
        if (v.empty()) return;
        int i = int(std::find(v.begin(), v.end(), focused) - v.begin());
        focused = v[size_t((i + d + int(v.size())) % int(v.size()))];
        pending_focus = focused;
    }
    void move_focused(int d) {  // swap with the next / previous visible pane; d == 0 moves it to the front
        auto it = std::find(order.begin(), order.end(), focused);
        if (it == order.end()) return;
        if (d == 0) {
            order.erase(it);
            order.insert(order.begin(), focused);
        } else {
            int i = int(it - order.begin()), j = i;
            do { j += d; } while (j >= 0 && j < int(order.size()) && !visible[order[j]]);
            if (j < 0 || j >= int(order.size())) return;
            std::swap(order[i], order[j]);
        }
        if (lay != Lay::Splits) need_layout = true;
        ImGui::MarkIniSettingsDirty();
    }

    void handle_shortcuts() {  // kitty-like defaults
        auto chord = [](ImGuiKey k) { return ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | k); };
        if (chord(ImGuiKey_L)) cycle_layout();
        if (chord(ImGuiKey_Z)) toggle_stack();
        if (chord(ImGuiKey_RightBracket)) cycle_focus(+1);
        if (chord(ImGuiKey_LeftBracket)) cycle_focus(-1);
        if (chord(ImGuiKey_F)) move_focused(+1);
        if (chord(ImGuiKey_B)) move_focused(-1);
        if (chord(ImGuiKey_GraveAccent)) move_focused(0);
        if (chord(ImGuiKey_R)) rotate_split();
    }

    // Panes switched on or off since the last frame.
    void on_visibility_changes() {
        for (char k : {'v', 'g', 's', 'f'}) {
            if (visible[k] == was_visible[k]) continue;
            ImGui::MarkIniSettingsDirty();
            if (!visible[k]) {  // hidden
                if (focused == k) { auto v = visible_order(); if (!v.empty()) focused = v[0]; }
                if (lay != Lay::Splits) need_layout = true;  // in Splits its area simply collapses
                continue;
            }
            if (lay != Lay::Splits) {  // shown: insert it in the pane order, then recompute
                order.erase(std::remove(order.begin(), order.end(), k), order.end());
                auto at = std::find(order.begin(), order.end(), focused);
                if (place == 3) order.push_back(k);
                else if (place == 2 || at == order.end()) order.insert(order.begin(), k);
                else if (place == 0) order.insert(at + 1, k);
                else order.insert(at, k);
                need_layout = true;
                continue;
            }
            // Splits: back to its old area if that still exists, else split the focused pane's area
            ImGuiWindow* w = ImGui::FindWindowByName(pane_window(k));
            if (w && w->DockId && ImGui::DockBuilderGetNode(w->DockId)) continue;
            ImGuiWindow* fw = ImGui::FindWindowByName(pane_window(focused));
            ImGuiDockNode* target = (fw && visible[focused] && fw->DockNode) ? fw->DockNode : nullptr;
            if (!target) {  // fall back to the largest area
                std::function<void(ImGuiDockNode*)> visit = [&](ImGuiDockNode* n) {
                    if (!n) return;
                    if (n->IsLeafNode()) {
                        if (!target || n->Size.x * n->Size.y > target->Size.x * target->Size.y) target = n;
                        return;
                    }
                    visit(n->ChildNodes[0]);
                    visit(n->ChildNodes[1]);
                };
                visit(ImGui::DockBuilderGetNode(dock_id_));
            }
            if (!target) continue;
            bool side = split_axis == 1 || (split_axis == 0 && target->Size.x >= target->Size.y);
            ImGuiDir dir = side ? (place == 1 ? ImGuiDir_Left : ImGuiDir_Right) : (place == 1 ? ImGuiDir_Up : ImGuiDir_Down);
            ImGuiID rest = 0, fresh = ImGui::DockBuilderSplitNode(target->ID, dir, 0.5f, nullptr, &rest);
            ImGui::DockBuilderDockWindow(pane_window(k), fresh);
            ImGui::DockBuilderFinish(dock_id_);
        }
    }

    static std::string tree_signature(const ImGuiDockNode* n) {
        if (!n) return "-";
        if (n->IsSplitNode())
            return std::string("(") + (n->SplitAxis == ImGuiAxis_X ? "x" : "y") + tree_signature(n->ChildNodes[0]) +
                   tree_signature(n->ChildNodes[1]) + ")";
        std::string s = "[";
        for (const ImGuiWindow* w : n->Windows)
            if (char k = key_of(w)) s += k;
        std::sort(s.begin() + 1, s.end());
        return s + "]";
    }

    void after_panes() {
        const ImGuiDockNode* root = ImGui::DockBuilderGetNode(dock_id_);
        std::string sig = tree_signature(root);
        if (sig_pending) {
            built_sig = sig;
            sig_pending = false;
        } else if (lay != Lay::Splits && !need_layout && sig != built_sig) {
            lay = Lay::Splits;  // the user rearranged panes by hand: keep that as a Splits layout
            ImGui::MarkIniSettingsDirty();
        }
        // Tall / Fat: dragging the main divider changes the bias
        if ((lay == Lay::Tall || lay == Lay::Fat) && root && root->IsSplitNode() && !ImGui::IsPopupOpen("##layout_menu")) {
            const ImGuiAxis ax = lay == Lay::Tall ? ImGuiAxis_X : ImGuiAxis_Y;
            if (root->SplitAxis == ax && root->Size[ax] > 1) {
                const ImGuiDockNode* main = root->ChildNodes[mirrored ? 1 : 0];
                float b = std::clamp(main->Size[ax] / root->Size[ax], 0.15f, 0.85f);
                if (std::abs(b - bias) > 0.002f) { bias = b; ImGui::MarkIniSettingsDirty(); }
            }
        }
    }

    // --- layout menu (top bar) ------------------------------------------------------------------
    void layout_menu(double x, double y) {
        const float s = float(UI::scale);
        ImGui::SetNextWindowPos(P(V(x, y)));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12 * s, 10 * s));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8 * s, 6 * s));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8 * s, 4 * s));
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 8 * s);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4 * s);
        ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 3 * s);
        const std::pair<ImGuiCol, ImU32> cols[] = {
            {ImGuiCol_Header, C::ACCENT_BG}, {ImGuiCol_HeaderHovered, C::ROW_HL}, {ImGuiCol_HeaderActive, C::ACCENT_BG},
            {ImGuiCol_FrameBg, C::BUTTON}, {ImGuiCol_FrameBgHovered, C::ROW_HL}, {ImGuiCol_FrameBgActive, C::ACCENT_BG},
            {ImGuiCol_SliderGrab, C::ACCENT}, {ImGuiCol_SliderGrabActive, C::ACCENT}, {ImGuiCol_CheckMark, C::ACCENT},
            {ImGuiCol_Button, C::BUTTON}, {ImGuiCol_ButtonHovered, C::ROW_HL}, {ImGuiCol_ButtonActive, C::ACCENT_BG},
            {ImGuiCol_TextDisabled, C::MUTED}, {ImGuiCol_Separator, C::DIVIDER}, {ImGuiCol_PopupBg, C::PANEL},
            {ImGuiCol_Border, C::BORDER},
        };
        for (auto& [k, v] : cols) ImGui::PushStyleColor(k, v);
        if (ImGui::BeginPopup("##layout_menu")) {
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
            SDL* dl = window_draw_list();
            auto logical = [](ImVec2 p) {
                return V(UI::origin.x + (p.x - UI::origin.x) / UI::scale, UI::origin.y + (p.y - UI::origin.y) / UI::scale);
            };
            const int nvis = int(visible_order().size());
            ImGui::TextDisabled("LAYOUT");
            for (int i = 0; i < 7; ++i) {
                ImVec2 p = logical(ImGui::GetCursorScreenPos());
                std::string id = std::string("##lay_") + LAY_ID[i];
                if (ImGui::Selectable(id.c_str(), int(lay) == i, 0, ImVec2(250 * s, 24 * s))) set_layout(Lay(i));
                bool on = int(lay) == i;
                icon_layout(dl, p.x + 12, p.y + 12, on ? C::ACCENT : C::TEXT_2, i);
                draw_text_vc(dl, Fonts::sans, 13, p.x + 32, p.y + 12, on ? C::ACCENT_TEXT : C::TEXT, LAY_NAME[i]);
                const char* hint = i == 0 ? "ctrl+shift+Z" : "";
                if (*hint) draw_text_vc(dl, Fonts::mono, 11, p.x + 250 - 8 - text_w(Fonts::mono, 11, hint), p.y + 12, C::MUTED, hint);
            }
            ImGui::PushItemWidth(150 * s);
            if (lay == Lay::Tall || lay == Lay::Fat) {
                ImGui::Separator();
                ImGui::TextDisabled("MAIN AREA");
                float pct = bias * 100;
                if (ImGui::SliderFloat("Size", &pct, 20, 80, "%.0f %%")) { bias = pct / 100; need_layout = true; }
                int fs_max = std::max(1, nvis - 1);
                if (ImGui::SliderInt("Panes", &full_size, 1, fs_max)) need_layout = true;
                if (ImGui::Checkbox(lay == Lay::Tall ? "Main on the right" : "Main at the bottom", &mirrored)) need_layout = true;
                if (ImGui::Button("Make focused pane main")) move_focused(0);
                if (need_layout) ImGui::MarkIniSettingsDirty();
            }
            if (lay == Lay::Splits) {
                ImGui::Separator();
                ImGui::TextDisabled("SPLITS");
                if (ImGui::Button("Rotate split  (ctrl+shift+R)")) rotate_split();
                ImGui::TextUnformatted("Move focused pane to edge");
                if (ImGui::Button("Left")) move_to_edge(ImGuiDir_Left);
                ImGui::SameLine();
                if (ImGui::Button("Top")) move_to_edge(ImGuiDir_Up);
                ImGui::SameLine();
                if (ImGui::Button("Right")) move_to_edge(ImGuiDir_Right);
                ImGui::SameLine();
                if (ImGui::Button("Bottom")) move_to_edge(ImGuiDir_Down);
                ImGui::Combo("Split axis", &split_axis, "Auto (longer side)\0Side by side\0Stacked\0");
            }
            ImGui::Separator();
            ImGui::TextDisabled("SHOWN PANES GO");
            ImGui::Combo("##place", &place, "After the focused pane\0Before the focused pane\0First\0Last\0");
            ImGui::PopItemWidth();
            ImGui::Separator();
            ImGui::TextDisabled("ctrl+shift+L  next layout      ctrl+shift+] [  focus");
            ImGui::TextDisabled("ctrl+shift+F B  move pane      ctrl+shift+`  make main");
            ImGui::EndPopup();
        }
        ImGui::PopStyleColor(int(std::size(cols)));
        ImGui::PopStyleVar(7);
    }

    // ImGui keeps a dockspace's "central node" on screen even when its pane is hidden, which leaves a
    // hole. Without a central node every area behaves the same: hiding its pane gives the space to the
    // neighbours, showing it again restores it. (ImGui re-creates one when a single area remains.)
    static void drop_central_node(ImGuiID dock_id) {
        ImGuiDockNode* root = ImGui::DockBuilderGetNode(dock_id);
        if (!root || !root->IsSplitNode()) return;
        std::function<void(ImGuiDockNode*)> visit = [&](ImGuiDockNode* n) {
            if (!n) return;
            if (n->IsCentralNode()) n->SetLocalFlags(n->LocalFlags & ~ImGuiDockNodeFlags_CentralNode);
            visit(n->ChildNodes[0]);
            visit(n->ChildNodes[1]);
        };
        visit(root);
        root->CentralNode = nullptr;
    }

    void dock_pane(char key) {
        bool open = true;
        ImGui::PushStyleColor(ImGuiCol_WindowBg, key == 'v' ? C::PANEL : C::CANVAS);
        ImGui::SetNextWindowSize(V(480 * UI::scale, 360 * UI::scale), ImGuiCond_FirstUseEver);  // when floating
        if (pending_focus == key) {
            ImGui::SetNextWindowFocus();  // also brings its tab to the front (Stack)
            pending_focus = 0;
        }
        bool shown = ImGui::Begin(pane_window(key), &open,
                                  ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
                                      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::PopStyleColor();
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) focused = key;
        if (shown) {
            ImVec2 p = ImGui::GetCursorScreenPos(), a = ImGui::GetContentRegionAvail();
            // back to logical pixels
            double x = UI::origin.x + (p.x - UI::origin.x) / UI::scale, y = UI::origin.y + (p.y - UI::origin.y) / UI::scale;
            double w = a.x / UI::scale, h = a.y / UI::scale;
            if (w > 1 && h > 1) {
                SDL* dl = window_draw_list();
                if (key == 'v') viewer(dl, x, y, w, h);
                else if (key == 'g') graph_pane(dl, x, y, w, h);
                else if (key == 's') script_pane(dl, x, y, w, h);
                else profile_pane(dl, x, y, w, h);
                set_cursor(V(x, y));
                ImGui::Dummy(ImVec2(0, 0));
            }
        }
        ImGui::End();
        if (!open) visible[key] = false;  // closed from the tab's cross
    }

    double pane_header(SDL* dl, double x, double y, double w) {
        dl->AddRectFilled(V(x, y), V(x + w, y + PANE_HDR), C::PANEL);
        dl->AddLine(V(x, y + PANE_HDR - 0.5), V(x + w, y + PANE_HDR - 0.5), C::DIVIDER);
        return y + PANE_HDR / 2;
    }

    // --- viewer pane -------------------------------------------------------
    Btn chip(SDL* dl, const char* id, double x, double cy, const std::string& label, bool on, double h = 26.0) {
        double w = 20 + text_w(Fonts::mono, 12, label);
        Hit r = hit(id, x, cy - h / 2, w, h);
        ImU32 bg = on ? C::ACCENT_BG : (r.hovered ? lighten(C::BUTTON) : C::BUTTON);
        dl->AddRectFilled(V(x, cy - h / 2), V(x + w, cy + h / 2), bg, 5);
        dl->AddRect(V(x + 0.5, cy - h / 2 + 0.5), V(x + w - 0.5, cy + h / 2 - 0.5), on ? C::ACCENT : C::BORDER, 5);
        draw_text_vc(dl, Fonts::mono, 12, x + 10, cy, on ? C::ACCENT_TEXT : C::TEXT_3, label);
        return {r.clicked, w};
    }

    void viewer(SDL* dl, double x, double y, double w, double h) {
        double cy = pane_header(dl, x, y, w);
        double widths[3], total = 0;
        for (int k = 0; k < 3; ++k) { widths[k] = 20 + text_w(Fonts::mono, 12, FIELDS[k].label); total += widths[k]; }
        double tr_w = 20 + text_w(Fonts::mono, 12, "tracers");
        total += 6 * 2 + 6 + 1 + 6 + tr_w;
        double cx = x + w - 12 - total;
        for (int k = 0; k < 3; ++k) {
            std::string id = std::string("##field_") + FIELDS[k].id;
            Btn b = chip(dl, id.c_str(), cx, cy, FIELDS[k].label, field == k);
            if (b.clicked && field != k) { field = k; refresh_previews(true); }
            cx += widths[k] + 6;
        }
        dl->AddLine(V(cx, cy - 9), V(cx, cy + 9), C::BORDER);
        cx += 7;
        if (chip(dl, "##tracers", cx, cy, "tracers", show_tracers).clicked) show_tracers = !show_tracers;

        double bx = x + 16, by = y + PANE_HDR + 16, bw = w - 32, bh = h - PANE_HDR - 32;
        bool horizontal = w / std::max(h, 1.0) > 1.25;
        double img = horizontal ? std::min(bh - 30, w * 0.55) : std::min(bw, bh - 300);
        img = std::max(140.0, std::floor(img));
        view_image(dl, bx, by, img);
        colorbar(dl, bx, by + img + 10, img);
        double sx, sy, sw;
        if (horizontal) { sx = bx + img + 24; sy = by; sw = bw - img - 24; }
        else { sx = bx; sy = by + img + 10 + 14 + 18; sw = bw; }
        if (sw > 120) {
            sy = plots(dl, sx, sy, sw);
            subscriptions(dl, sx, sy + 18, sw);
        }
    }

    void view_image(SDL* dl, double x, double y, double s) {
        dl->AddRectFilled(V(x, y), V(x + s, y + s), rgba("#0c0d10"), 4);
        dl->PushClipRect(V(x, y), V(x + s, y + s), true);
        if (view3d) {
            draw_cube(dl, x, y, s);
        } else {
            dl->AddImageRounded(tex_main.ref, V(x, y), V(x + s, y + s), ImVec2(0, 0), ImVec2(1, 1), rgba("#ffffff"), 4);
            if (show_tracers)
                for (auto& p : tracer_pts) {
                    double qx = x + (p[0] + 0.5) * s, qy = y + (0.5 - p[1]) * s;
                    dl->AddRectFilled(V(qx, qy), V(qx + 2, qy + 2), rgba("#ffffff", 0.55));
                }
            dl->AddCircle(V(x + 0.648 * s, y + 0.585 * s), 4.5f, rgba("#ffffff"), 0, 1.5f);
        }


        const char* seg[2] = {"3D", "Slice"};
        double seg_w[2] = {20 + text_w(Fonts::mono, 11, seg[0]), 20 + text_w(Fonts::mono, 11, seg[1])};
        double tw = seg_w[0] + seg_w[1] + 2 * 3;
        std::string label = std::string("state.") + FIELDS[field].label + " · " + (view3d ? "cut z=0.5" : "z=0.5") +
                            " · 10 Hz";
        if (8 + 20 + text_w(Fonts::mono, 11, label) + 8 + 12 + tw + 8 > s) label = std::string("state.") + FIELDS[field].label;
        double bw_ = 8 + 6 + 6 + text_w(Fonts::mono, 11, label) + 8;
        dl->AddRectFilled(V(x + 8, y + 8), V(x + 8 + bw_, y + 28), C::CARD, 4);
        draw_live_dot(dl, x + 8 + 11, y + 18);
        draw_text_vc(dl, Fonts::mono, 11, x + 8 + 20, y + 18, C::TEXT, label);

        double sx0 = x + s - 8 - tw;
        dl->AddRectFilled(V(sx0, y + 8), V(sx0 + tw, y + 36), C::CARD, 6);
        dl->AddRect(V(sx0 + 0.5, y + 8.5), V(sx0 + tw - 0.5, y + 35.5), C::BORDER, 6);
        double bx = sx0 + 2;
        for (int i = 0; i < 2; ++i) {
            bool is3d = i == 0, on = view3d == is3d;
            std::string id = std::string("##seg_") + seg[i];
            Hit h = hit(id.c_str(), bx, y + 10, seg_w[i], 24);
            if (on) dl->AddRectFilled(V(bx, y + 10), V(bx + seg_w[i], y + 34), C::ACCENT_BG, 4);
            draw_text_vc(dl, Fonts::mono, 11, bx + 10, y + 22, on ? C::ACCENT : C::MUTED, seg[i]);
            if (h.clicked) view3d = is3d;
            bx += seg_w[i] + 2;
        }

        double val = sim.sample(field, 0.648 - 0.5, 0.5 - 0.585);
        std::string probe = std::string(FIELDS[field].label) + " = " + fmt("%.3g", val) + "  (0.65, 0.41)";
        double pw = 16 + text_w(Fonts::mono, 11, probe);
        dl->AddRectFilled(V(x + s - 8 - pw, y + s - 30), V(x + s - 8, y + s - 8), C::CARD, 4);
        dl->AddRect(V(x + s - 8 - pw, y + s - 30), V(x + s - 8, y + s - 8), C::NODE_BORDER, 4);
        draw_text_vc(dl, Fonts::mono, 11, x + s - pw, y + s - 19, C::TEXT, probe);

        // orbit with left drag (3D), double-click resets the camera. Submitted after the overlay buttons:
        // with overlapping items the first one submitted takes the hover, so the 3D / Slice switch must come first.
        set_cursor(V(x, y));
        invisible_button("##view_orbit", V(s, s));
        if (view3d && ImGui::IsItemActive() && ImGui::IsMouseDragging(0)) {
            ImVec2 d = mouse_delta();
            yaw += d.x * 0.008;
            pitch = std::max(-1.35, std::min(1.35, pitch + d.y * 0.008));
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) { yaw = -38 * PI / 180; pitch = 24 * PI / 180; }
        dl->PopClipRect();
    }

    void draw_cube(SDL* dl, double x, double y, double s) {
        const double cyw = std::cos(yaw), syw = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
        const double scale = s * 0.56, ox = x + s / 2, oy = y + s * 0.51;
        using P3 = std::array<double, 3>;
        auto rot = [&](P3 p) {
            double px = p[0] * cyw + p[2] * syw, pz = -p[0] * syw + p[2] * cyw, py = p[1];
            return P3{px, py * cp - pz * sp, py * sp + pz * cp};
        };
        auto proj = [&](P3 p) { P3 r = rot(p); return V(ox + r[0] * scale, oy - r[1] * scale); };
        const double hh = 0.5;
        double ambient = std::clamp(DemoSimulation::normalize<double>(field, sim.sample(field, 0.49, 0.49), main_lo,
                                                                      main_hi), 0.0, 1.0);
        P3 light{0.35, 0.8, 0.5};
        double ln = std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
        for (double& c : light) c /= ln;
        struct Face { P3 n; std::array<P3, 4> c; bool textured; };
        const Face faces[6] = {
            {{0, 0, 1}, {{{-hh, hh, 0}, {hh, hh, 0}, {hh, -hh, 0}, {-hh, -hh, 0}}}, true},
            {{0, 0, -1}, {{{hh, hh, -hh}, {-hh, hh, -hh}, {-hh, -hh, -hh}, {hh, -hh, -hh}}}, false},
            {{1, 0, 0}, {{{hh, hh, 0}, {hh, hh, -hh}, {hh, -hh, -hh}, {hh, -hh, 0}}}, false},
            {{-1, 0, 0}, {{{-hh, hh, -hh}, {-hh, hh, 0}, {-hh, -hh, 0}, {-hh, -hh, -hh}}}, false},
            {{0, 1, 0}, {{{-hh, hh, -hh}, {hh, hh, -hh}, {hh, hh, 0}, {-hh, hh, 0}}}, false},
            {{0, -1, 0}, {{{-hh, -hh, 0}, {hh, -hh, 0}, {hh, -hh, -hh}, {-hh, -hh, -hh}}}, false},
        };
        for (const Face& f : faces) {
            P3 rn = rot(f.n);
            if (rn[2] <= 0) continue;  // back-facing: convex box needs no sorting
            ImVec2 pts[4] = {proj(f.c[0]), proj(f.c[1]), proj(f.c[2]), proj(f.c[3])};
            if (f.textured) {
                dl->AddImageQuad(tex_main.ref, pts[0], pts[1], pts[2], pts[3]);
                dl->AddPolyline(pts, 4, rgba("#ffffff", 0.45), ImDrawFlags_Closed, 1.0f);
            } else {
                double shade = 0.62 + 0.5 * std::max(0.0, rn[0] * light[0] + rn[1] * light[1] + rn[2] * light[2]);
                dl->AddConvexPolyFilled(pts, 4, lut_color(VIRIDIS, ambient, shade));
                dl->AddPolyline(pts, 4, rgba("#ffffff", 0.3), ImDrawFlags_Closed, 1.0f);
            }
        }
        if (rot({0, 0, 1})[2] > 0) {
            if (show_tracers) {
                ImU32 col = rgba("#ffffff", 0.55);
                for (auto& p : tracer_pts) {
                    ImVec2 q = proj({p[0], p[1], 0.0});
                    dl->AddRectFilled(V(q.x - 1.0, q.y - 1.0), V(q.x + 1.0, q.y + 1.0), col);
                }
            }
            dl->AddCircle(proj({0.148, -0.085, 0.0}), 4.5f, rgba("#ffffff"), 0, 1.5f);
        }
        ImU32 ghost = rgba("#ffffff", 0.24);
        P3 front[4] = {{-hh, hh, hh}, {hh, hh, hh}, {hh, -hh, hh}, {-hh, -hh, hh}};
        for (int i = 0; i < 4; ++i) {
            ImVec2 a = proj(front[i]), b = proj(front[(i + 1) % 4]), c = proj({front[i][0], front[i][1], 0.0});
            dashed_line(dl, a.x, a.y, b.x, b.y, ghost);
            dashed_line(dl, a.x, a.y, c.x, c.y, ghost);
        }
        double gx = x + 30, gy = y + s - 22;
        struct Ax { P3 d; ImU32 col; const char* name; };
        const Ax axes[3] = {{{1, 0, 0}, rgba("#e8866a"), "x"}, {{0, 1, 0}, rgba("#8cc084"), "y"},
                            {{0, 0, 1}, rgba("#6f9be8"), "z"}};
        for (const Ax& a : axes) {
            P3 r = rot(a.d);
            double ex = gx + r[0] * 14, ey = gy - r[1] * 14;
            dl->AddLine(V(gx, gy), V(ex, ey), a.col, 1.6f);
            draw_text_vc(dl, Fonts::mono, 11, ex + r[0] * 5 - 3, ey - r[1] * 5, a.col, a.name);
        }
    }

    void colorbar(SDL* dl, double x, double y, double w) {
        auto f = [&](double v) { return FIELDS[field].log ? fmt("%.2g", v) : strip_zeros(fmt("%.2f", v)); };
        std::string lo_s = f(main_lo), hi_s = f(main_hi), name = FIELDS[field].cb_label;
        double cy = y + 7;
        draw_text_vc(dl, Fonts::mono, 11, x, cy, C::TEXT_3, lo_s);
        double bx0 = x + text_w(Fonts::mono, 11, lo_s) + 10;
        double bx1 = x + w - text_w(Fonts::mono, 11, hi_s) - 10 - text_w(Fonts::mono, 11, name) - 10;
        const int n = 24;
        for (int i = 0; i < n; ++i) {
            double a = bx0 + (bx1 - bx0) * i / n, b = bx0 + (bx1 - bx0) * (i + 1) / n;
            ImU32 ca = viridis_u32(double(i) / n), cb = viridis_u32(double(i + 1) / n);
            dl->AddRectFilledMultiColor(V(a, cy - 4), V(b + 0.5, cy + 4), ca, cb, cb, ca);
        }
        draw_text_vc(dl, Fonts::mono, 11, bx1 + 10, cy, C::TEXT_3, hi_s);
        draw_text_vc(dl, Fonts::mono, 11, bx1 + 10 + text_w(Fonts::mono, 11, hi_s) + 10, cy, C::TEXT, name);
    }

    double plots(SDL* dl, double x, double y, double w) {
        auto& hist = sim.history;
        struct Item { const char* title; std::string value; const std::deque<double>* s; ImU32 col; bool ref; };
        const Item items[3] = {
            {"Total energy", fmt("%.7f", hist["E_tot"].back()), &hist["E_tot"], C::TEAL, true},
            {"Timestep dt", fmt("%.2e", hist["dt"].back()), &hist["dt"], C::ACCENT, false},
            {"Speed", fmt("%.1f", hist["speed"].back() / 1e9) + " Gcell/s", &hist["speed"], C::BLUE, false},
        };
        const double gap = 10.0, cw = (w - gap * 2) / 3;
        for (int i = 0; i < 3; ++i) {
            const Item& it = items[i];
            double cx = x + i * (cw + gap);
            draw_text(dl, Fonts::sans, 12, cx, y, C::TEXT_2, it.title);
            draw_text(dl, Fonts::mono, 12, cx, y + 18, C::TEXT, it.value);
            double by0 = y + 38, by1 = y + 38 + 44;
            dl->AddRectFilled(V(cx, by0), V(cx + cw, by1), C::CANVAS, 4);
            for (double f : {0.25, 0.5, 0.75}) {
                double gy = by0 + (by1 - by0) * f;
                dl->AddLine(V(cx, gy), V(cx + cw, gy), rgba("#26282d"));
            }
            if (it.ref) {
                auto [mn, mx] = std::minmax_element(it.s->begin(), it.s->end());
                double span = (*mx - *mn) != 0 ? *mx - *mn : 1.0;
                double ry = by1 - (1.0 - (*mn - span * 0.12)) / (span * 1.24) * (by1 - by0);
                if (by0 < ry && ry < by1) dashed_line(dl, cx, ry, cx + cw, ry, rgba("#4a4c52"), 3, 3);
            }
            sparkline(dl, cx + 1, by0 + 3, cx + cw - 1, by1 - 3, *it.s, it.col, 1.6);
        }
        return y + 82;
    }

    void subscriptions(SDL* dl, double x, double y, double w) {
        draw_text(dl, Fonts::sans, 12, x, y, C::TEXT_2, "Preview subscriptions");
        std::string rate = fmt("%.1f", bytes_per_s / 1e6) + " MB/s";
        draw_text(dl, Fonts::mono, 11, x + w - text_w(Fonts::mono, 11, rate), y + 1, C::MUTED, rate);
        bool on = graph.previews_on;
        struct R { std::string c[4]; bool active; };
        const bool pf = visible['f'] && prof_live;
        const R rows[6] = {
            {{std::string("state.") + FIELDS[field].label, "main", "512²", "10 Hz"}, true},
            {{"state.ρ", "card", "128²", on ? "4 Hz" : "off"}, on},
            {{"tracers", "card", "128²", on ? "4 Hz" : "off"}, on},
            {{"diagnostics", "card", "series", on ? "1 Hz" : "off"}, on},
            {{"profile", "pane", "scopes", pf ? "2 Hz" : "off"}, pf},
            {{"mesh", "card", "—", "off"}, false},
        };
        const double col_x[4] = {0.0, 0.4, 0.6, 0.8};  // first column is the widest (edge names)
        double ry = y + 22;
        for (const R& r : rows) {
            for (int i = 0; i < 4; ++i) {
                ImU32 col = r.active ? (i == 0 ? C::TEXT : C::TEXT_3) : C::DIM;
                draw_text(dl, Fonts::mono, 11, x + col_x[i] * w, ry, col, r.c[i]);
            }
            ry += 17;
        }
    }

    // --- profile pane (live flamegraph) ----------------------------------------
    static ImU32 cat_color(DemoSimulation::Cat c, const char* name) {
        uint32_t hsh = 2166136261u;  // small per-scope shade variation, stable across frames
        for (const char* p = name; *p; ++p) hsh = (hsh ^ uint8_t(*p)) * 16777619u;
        double k = 0.88 + 0.24 * double(hsh % 1000) / 1000.0;
        auto shade = [&](int r, int g, int b) {
            return rgb_u32(std::min(255, int(r * k)), std::min(255, int(g * k)), std::min(255, int(b * k)));
        };
        switch (c) {
            case DemoSimulation::Cat::Gpu: return shade(0xd9, 0x8f, 0x3a);
            case DemoSimulation::Cat::Mpi: return shade(0x6f, 0x9b, 0xe8);
            case DemoSimulation::Cat::Host: return shade(0x9a, 0x96, 0x8e);
            case DemoSimulation::Cat::Io: return shade(0x4f, 0xb3, 0xa9);
            default: return rgba("#3a3c42");
        }
    }

    void profile_pane(SDL* dl, double x, double y, double w, double h) {
        const auto& prof = sim.prof;
        if (prof_snap.size() != prof.size()) {
            prof_snap.resize(prof.size());
            for (size_t i = 0; i < prof.size(); ++i) prof_snap[i] = prof[i].ema;
        }
        double cy = pane_header(dl, x, y, w);
        std::string info = "rank 0 · running mean · " + fmt("%.2f", prof_snap[0]) + " ms / step";
        // right: Live toggle, Reset zoom when zoomed
        std::string live = prof_live ? "Live · 2 Hz" : "Paused";
        double live_w = 20 + (prof_live ? 12 : 0) + text_w(Fonts::sans, 12, live);
        double bx = x + w - 12 - live_w;
        if (prof_focus != 0) {
            double rw = 20 + text_w(Fonts::sans, 12, "Reset zoom");
            if (text_button(dl, "##prof_reset", bx - 6 - rw, cy, "Reset zoom").clicked) prof_focus = 0;
        }
        if (text_button(dl, "##prof_live", bx, cy, live, prof_live, prof_live).clicked) prof_live = !prof_live;
        double info_max = bx - 12 - (prof_focus != 0 ? 20 + text_w(Fonts::sans, 12, "Reset zoom") + 6 : 0);
        if (x + 12 + text_w(Fonts::mono, 12, info) <= info_max) draw_text_vc(dl, Fonts::mono, 12, x + 12, cy, C::MUTED, info);

        // frames: ancestors of the focused scope (dimmed, full width), the focus, then its subtree
        const double fx = x + 12, fw = w - 24, gap = 1;
        double top = y + PANE_HDR + 10;
        std::vector<int> chain;
        for (int i = prof_focus; i >= 0; i = prof[i].parent) chain.insert(chain.begin(), i);
        std::function<int(int)> sub_depth = [&](int i) {
            int d = 1;
            for (size_t c = 0; c < prof.size(); ++c)
                if (prof[c].parent == i && prof_snap[c] > 1e-4) d = std::max(d, 1 + sub_depth(int(c)));
            return d;
        };
        const int rows = int(chain.size()) - 1 + sub_depth(prof_focus);
        // rows shrink (down to 14 px) before anything is cut; the legend goes first when space is short
        double avail = y + h - top - 8;
        bool show_legend = avail - 26 >= rows * (16 + gap);
        if (show_legend) avail -= 26;
        const double row_h = std::clamp(avail / rows - gap, 14.0, 22.0);
        const double bottom = top + avail;
        int depth = 0;
        auto frame_box = [&](int i, double bx0, double bw, int d, bool dim) {
            double by0 = top + d * (row_h + gap);
            if (by0 + row_h > bottom + 0.5 || bw < 1.0) return;
            ImU32 fill = dim ? rgba("#2a2b30") : cat_color(prof[i].cat, prof[i].name);
            std::string id = std::string("##prof_") + std::to_string(i) + (dim ? "a" : "");
            Hit hv = hit(id.c_str(), bx0, by0, std::max(bw - 1, 1.0), row_h);
            dl->AddRectFilled(V(bx0, by0), V(bx0 + std::max(bw - 1, 1.0), by0 + row_h), hv.hovered ? lighten(fill, 22) : fill, 2);
            if (hv.hovered) dl->AddRect(V(bx0, by0), V(bx0 + std::max(bw - 1, 1.0), by0 + row_h), C::TEXT, 2, 0, 1.0f);
            bool dark_text = !dim && prof[i].cat != DemoSimulation::Cat::Root;
            ImU32 tc = dark_text ? rgba("#16140f") : C::TEXT_2;
            std::string ms = fmt("%.2f ms", prof_snap[i]);
            std::string label = prof[i].name;
            double tw = text_w(Fonts::mono, 11, label), mw = text_w(Fonts::mono, 11, ms);
            if (bw > tw + mw + 24 && row_h >= 16) {
                draw_text_vc(dl, Fonts::mono, 11, bx0 + 6, by0 + row_h / 2, tc, label);
                draw_text_vc(dl, Fonts::mono, 11, bx0 + bw - 7 - mw, by0 + row_h / 2, tc, ms);
            } else if (bw > tw + 12) {
                draw_text_vc(dl, Fonts::mono, 11, bx0 + 6, by0 + row_h / 2, tc, label);
            } else if (bw > 28) {
                std::string cut = label.substr(0, std::max<size_t>(1, size_t((bw - 16) / 6.6))) + "…";
                draw_text_vc(dl, Fonts::mono, 11, bx0 + 5, by0 + row_h / 2, tc, cut);
            }
            if (hv.hovered) {
                double step = std::max(prof_snap[0], 1e-9);
                double par = prof[i].parent >= 0 ? std::max(prof_snap[prof[i].parent], 1e-9) : step;
                const char* cat = prof[i].cat == DemoSimulation::Cat::Gpu ? "GPU kernel"
                                  : prof[i].cat == DemoSimulation::Cat::Mpi ? "MPI / communication"
                                  : prof[i].cat == DemoSimulation::Cat::Io  ? "I/O" : "host";
                tooltip((label + "\n" + fmt("%.3f ms", prof_snap[i]) + "   " + fmt("%.1f", 100 * prof_snap[i] / step) +
                         "% of step   " + fmt("%.1f", 100 * prof_snap[i] / par) + "% of parent\n" + cat +
                         (prof[i].leaf ? "" : "   ·   click to zoom")).c_str());
            }
            if (hv.clicked && (dim || !prof[i].leaf)) prof_focus = i;  // zoom in, or back out via an ancestor
        };
        for (size_t k = 0; k + 1 < chain.size(); ++k) frame_box(chain[k], fx, fw, depth++, true);
        std::function<void(int, double, double, int)> draw_sub = [&](int i, double bx0, double bw, int d) {
            frame_box(i, bx0, bw, d, false);
            double total = std::max(prof_snap[i], 1e-9), cx = bx0;
            for (size_t c = 0; c < prof.size(); ++c) {
                if (prof[c].parent != i || prof_snap[c] <= 1e-4) continue;
                double cw = bw * prof_snap[c] / total;
                draw_sub(int(c), cx, cw, d + 1);
                cx += cw;
            }
        };
        draw_sub(prof_focus, fx, fw, depth);

        // legend
        struct L { DemoSimulation::Cat c; const char* t; };
        const L legend[4] = {{DemoSimulation::Cat::Gpu, "GPU kernel"}, {DemoSimulation::Cat::Mpi, "MPI / comm"},
                             {DemoSimulation::Cat::Host, "host"}, {DemoSimulation::Cat::Io, "I/O"}};
        if (!show_legend) return;
        double lx = x + 12, ly = y + h - 16;
        for (const L& l : legend) {
            dl->AddRectFilled(V(lx, ly - 5), V(lx + 10, ly + 5), cat_color(l.c, ""), 2);
            draw_text_vc(dl, Fonts::sans, 11, lx + 16, ly, C::MUTED, l.t);
            lx += 16 + text_w(Fonts::sans, 11, l.t) + 16;
        }
        std::string hint = "click a frame to zoom";
        double hw = text_w(Fonts::sans, 11, hint);
        if (lx + 20 + hw < x + w - 12) draw_text_vc(dl, Fonts::sans, 11, x + w - 12 - hw, ly, C::DIM, hint);
    }

    // --- graph pane --------------------------------------------------------
    void graph_pane(SDL* dl, double x, double y, double w, double h) {
        double cy = pane_header(dl, x, y, w);
        int n_c = 0;
        for (auto& n : graph.nodes) n_c += n.compute;
        int n_e = int(graph.nodes.size()) - n_c;
        std::string counts = std::to_string(n_c) + " nodes · " + std::to_string(n_e) + " edges · " +
                             std::to_string(graph.links.size()) + " links";
        double counts_x = x + 12;
        std::string prev_label = graph.previews_on ? "Previews · 4 Hz" : "Previews off";
        double widths[3] = {20 + 12 + text_w(Fonts::sans, 12, prev_label), 20 + text_w(Fonts::sans, 12, "Auto-layout"),
                            20 + text_w(Fonts::sans, 12, "Fit")};
        double bx = x + w - 12 - (widths[0] + widths[1] + widths[2]) - 6 * 2;
        std::string zoom = std::to_string(int(std::nearbyint(graph.zoom * 100))) + "%";
        double zx = bx - 8 - text_w(Fonts::mono, 11, zoom);
        draw_text_vc(dl, Fonts::mono, 11, zx, cy, C::MUTED, zoom);
        if (counts_x + text_w(Fonts::mono, 12, counts) + 12 <= zx) draw_text_vc(dl, Fonts::mono, 12, counts_x, cy, C::MUTED, counts);
        Btn b = text_button(dl, "##fit", bx, cy, "Fit");
        if (b.clicked) { graph.user_view = false; graph.fit(w, h - PANE_HDR); }
        bx += b.w + 6;
        b = text_button(dl, "##autolayout", bx, cy, "Auto-layout");
        if (b.clicked) {
            auto fresh = build_demo_graph().first;
            for (auto& n : graph.nodes)
                for (auto& f : fresh)
                    if (f.id == n.id) { n.x = f.x; n.y = f.y; }
            graph.user_view = false;
            graph.fit(w, h - PANE_HDR);
        }
        bx += b.w + 6;
        if (text_button(dl, "##previews", bx, cy, prev_label, graph.previews_on, graph.previews_on).clicked)
            graph.previews_on = !graph.previews_on;
        graph.draw(*this, x, y + PANE_HDR, w, h - PANE_HDR);
    }

    // --- script pane -------------------------------------------------------
    void script_pane(SDL* dl, double x, double y, double w, double h) {
        dl->AddRectFilled(V(x, y), V(x + w, y + 36), C::PANEL);
        dl->AddLine(V(x, y + 35.5), V(x + w, y + 35.5), C::DIVIDER);
        double tx = x;
        struct Tab { const char* label; bool active; };
        const Tab tabs[3] = {{"run_sedov.py", true}, {"Log", false}, {"Problems", false}};
        for (const Tab& t : tabs) {
            ImFont* font = t.active ? Fonts::mono : Fonts::sans;
            bool problems = !std::strcmp(t.label, "Problems");
            double tw = 28 + text_w(font, 12, t.label) + (t.active ? 14 : 0) + (problems ? 22 : 0);
            if (t.active) dl->AddRectFilled(V(tx, y), V(tx + tw, y + 36), C::CANVAS);
            std::string id = std::string("##tab_") + t.label;
            hit(id.c_str(), tx, y, tw, 36);
            draw_text_vc(dl, font, 12, tx + 14, y + 18, t.active ? C::TEXT : C::MUTED, t.label);
            double lw = text_w(font, 12, t.label);
            if (t.active) draw_live_dot(dl, tx + 14 + lw + 10, y + 18);
            if (problems) {
                double bx = tx + 14 + lw + 6;
                dl->AddRectFilled(V(bx, y + 10), V(bx + 16, y + 26), C::ROW_HL, 8);
                draw_text_vc(dl, Fonts::sans, 11, bx + 5, y + 18, C::TEXT_2, "0");
            }
            dl->AddLine(V(tx + tw - 0.5, y), V(tx + tw - 0.5, y + 36), C::DIVIDER);
            tx += tw;
        }
        double ay = y + 36;
        dl->AddRectFilled(V(x, ay), V(x + w, ay + 38), C::CANVAS);
        dl->AddLine(V(x, ay + 37.5), V(x + w, ay + 37.5), rgba("#25272b"));
        double acy = ay + 19;
        double apply_w = 20 + text_w(Fonts::sans, 12, "Apply at next step"), dry_w = 20 + text_w(Fonts::sans, 12, "Dry run");
        double bx = x + w - 12 - apply_w - 6 - dry_w;
        dl->AddCircleFilled(V(x + 15, acy), 3, C::TEAL);
        std::string sync;
        for (const char* s : {"in sync with graph", "in sync", ""}) {
            sync = s;
            if (x + 24 + text_w(Fonts::sans, 12, sync) + 12 <= bx) break;
        }
        draw_text_vc(dl, Fonts::sans, 12, x + 24, acy, C::TEAL_TEXT, sync);
        text_button(dl, "##dry", bx, acy, "Dry run");
        text_button(dl, "##apply", bx + dry_w + 6, acy, "Apply at next step", true);
        double ey = ay + 38;
        set_cursor(V(x, ey + 6));
        ImGui::PushFont(Fonts::mono, S(13));
        editor.Render("##run_script", V(w * UI::scale, (h - 38 - 36 - 6) * UI::scale));
        ImGui::PopFont();
    }

    // --- status bar --------------------------------------------------------
    void status_bar(SDL* dl, double X, double Y, double W) {
        dl->AddRectFilled(V(X, Y), V(X + W, Y + STATUS_H), C::PANEL);
        dl->AddLine(V(X, Y + 0.5), V(X + W, Y + 0.5), C::DIVIDER);
        double cy = Y + STATUS_H / 2, wob = std::sin(now() * 0.7);
        std::string left[3] = {"8 MPI ranks · control on rank 0", "GPU util " + fmt("%.0f", 87 + 2 * wob) + "%",
                               "GPU mem 61 / 80 GB"};
        std::string right[3] = {"step " + fmt("%.1f", sim.step_ms) + " ms", "preview extract 0.3 ms",
                                "link latency " + fmt("%.0f", 38 + 3 * wob) + " ms"};
        double tx = X + 16;
        for (auto& t : left) {
            draw_text_vc(dl, Fonts::mono, 11, tx, cy, C::TEXT_3, t);
            tx += text_w(Fonts::mono, 11, t) + 20;
        }
        tx = X + W - 16;
        for (int i = 2; i >= 0; --i) {
            tx -= text_w(Fonts::mono, 11, right[i]);
            draw_text_vc(dl, Fonts::mono, 11, tx, cy, C::TEXT_3, right[i]);
            tx -= 20;
        }
    }
};

// Layout settings in the .ini file (next to ImGui's own dock tree) ---------------------------------
static App* g_app = nullptr;
static void* ini_open(ImGuiContext*, ImGuiSettingsHandler*, const char* name) {
    return std::strcmp(name, "Layout") == 0 ? (void*)1 : nullptr;
}
static void ini_line(ImGuiContext*, ImGuiSettingsHandler*, void*, const char* line) {
    App& a = *g_app;
    char buf[64];
    float f;
    int i;
    if (std::sscanf(line, "Layout=%63s", buf) == 1 && !a.layout_from_cli) a.lay = parse_lay(buf);
    else if (std::sscanf(line, "Bias=%f", &f) == 1) a.bias = std::clamp(f, 0.15f, 0.85f);
    else if (std::sscanf(line, "Main=%d", &i) == 1) a.full_size = std::max(1, i);
    else if (std::sscanf(line, "Mirrored=%d", &i) == 1) a.mirrored = i != 0;
    else if (std::sscanf(line, "Place=%d", &i) == 1) a.place = std::clamp(i, 0, 3);
    else if (std::sscanf(line, "Axis=%d", &i) == 1) a.split_axis = std::clamp(i, 0, 2);
    else if (std::sscanf(line, "Order=%63s", buf) == 1 && std::strlen(buf) == 4) a.order.assign(buf, buf + 4);
    else if (std::sscanf(line, "Visible=%63s", buf) == 1 && !a.profile_from_cli)
        for (char k : {'v', 'g', 's', 'f'}) a.visible[k] = a.was_visible[k] = std::strchr(buf, k) != nullptr;
    else if (std::strncmp(line, "Visible=", 8) == 0 && !a.profile_from_cli)
        for (char k : {'v', 'g', 's', 'f'}) a.visible[k] = a.was_visible[k] = false;
}
static void ini_write(ImGuiContext*, ImGuiSettingsHandler* h, ImGuiTextBuffer* out) {
    App& a = *g_app;
    std::string vis, ord(a.order.begin(), a.order.end());
    for (char k : {'v', 'g', 's', 'f'})
        if (a.visible[k]) vis += k;
    out->appendf("[%s][Layout]\nLayout=%s\nBias=%.3f\nMain=%d\nMirrored=%d\nPlace=%d\nAxis=%d\nOrder=%s\nVisible=%s\n\n",
                 h->TypeName, LAY_ID[int(a.lay)], a.bias, a.full_size, a.mirrored ? 1 : 0, a.place, a.split_axis,
                 ord.c_str(), vis.c_str());
}

// GraphView members that need App -------------------------------------------
void GraphView::draw(App& app, double x, double y, double w, double h) {
    SDL* dl = window_draw_list();
    const double origin[2] = {x, y};
    if (!user_view && (std::abs(last_size[0] - w) > 0.5 || std::abs(last_size[1] - h) > 0.5)) fit(w, h);
    last_size[0] = w; last_size[1] = h;
    dl->AddRectFilled(V(x, y), V(x + w, y + h), C::CANVAS);
    draw_grid(dl, x, y, w, h);
    interact(x, y, w, h, origin);
    dl->PushClipRect(V(x, y), V(x + w, y + h), true);
    for (auto& ln : links) draw_link(dl, origin, ln);
    for (auto& n : nodes) {
        if (n.compute) draw_compute(dl, origin, n);
        else draw_edge(dl, origin, n, app);
    }
    dl->PopClipRect();
    draw_legend(dl, x, y, w, h);
}

void GraphView::draw_edge(SDL* dl, const double* origin, const Node& n, App& app) {
    const double z = zoom;
    auto p0 = to_screen(origin, n.x, n.y);
    double x0 = p0[0], y0 = p0[1], x1 = x0 + n.w * z, y1 = y0 + n.h() * z;
    dl->AddRectFilled(V(x0, y0), V(x1, y1), C::CARD, float(10 * z));
    dl->AddRect(V(x0, y0), V(x1, y1), selected == n.id ? C::ACCENT : C::NODE_BORDER, float(10 * z), 0, 1.0f);
    dl->AddLine(V(x0 + 1, y0 + 30 * z), V(x1 - 1, y0 + 30 * z), rgba("#26282d"));
    double hc = y0 + 15 * z;
    dl->AddRectFilled(V(x0 + 12 * z, hc - 4 * z), V(x0 + 20 * z, hc + 4 * z), n.color, float(2 * z));
    draw_text_vc(dl, Fonts::medium, 13 * z, x0 + 27 * z, hc, C::TEXT, n.title);
    double mw = text_w(Fonts::mono, 11 * z, n.meta);
    draw_text_vc(dl, Fonts::mono, 11 * z, x1 - 10 * z - mw, hc, C::MUTED, n.meta);
    port(dl, x0, y0 + 16 * z, n.color);
    port(dl, x1, y0 + 16 * z, n.color);

    double bx0 = x0 + 7 * z, by0 = y0 + 37 * z;
    if (n.preview == Preview::None) {
        draw_text(dl, Fonts::mono, 11 * z, x0 + 12 * z, y0 + 38 * z, C::TEXT_3, "256³ · 8 patches");
        draw_text(dl, Fonts::mono, 11 * z, x0 + 12 * z, y0 + 54 * z, C::DIM, "no preview");
        return;
    }
    double img_h = n.preview == Preview::Series ? 56 : 88;
    double bx1 = bx0 + 136 * z, by1 = by0 + img_h * z;
    if (!previews_on) {
        dl->AddRectFilled(V(bx0, by0), V(bx1, by1), C::DARK, float(4 * z));
        draw_text_vc(dl, Fonts::mono, 11 * z, bx0 + 8 * z, (by0 + by1) / 2, C::DIM, "preview paused");
    } else if (n.preview == Preview::Slice) {
        dl->AddImageRounded(app.tex_card_state.ref, V(bx0, by0), V(bx1, by1), ImVec2(0, 0), ImVec2(1, 1),
                            rgba("#ffffff"), float(4 * z));
    } else if (n.preview == Preview::Tracers) {
        dl->AddImageRounded(app.tex_card_tracers.ref, V(bx0, by0), V(bx1, by1), ImVec2(0, 0), ImVec2(1, 1),
                            rgba("#ffffff"), float(4 * z));
    } else {
        dl->AddRectFilled(V(bx0, by0), V(bx1, by1), C::DARK, float(4 * z));
        sparkline(dl, bx0, by0 + 3 * z, bx1, by1 - 3 * z, app.sim.history["dt"], C::ACCENT, 1.5);
        sparkline(dl, bx0, by0 + 3 * z, bx1, by1 - 3 * z, app.sim.history["E_tot"], C::TEAL, 1.2, 0.35);
    }
    double fy = by1 + 11 * z;
    draw_text_vc(dl, Fonts::mono, 11 * z, bx0, fy, C::TEXT_3, n.footer);
    if (previews_on) draw_live_dot(dl, bx1 - 3 * z, fy - 1 * z, 3 * z);
}

// ════════════════════════════════════════════════════════════════════════════
//  Entry point
// ════════════════════════════════════════════════════════════════════════════
static void print_bench(const App& app, int warmup) {
    std::printf("BENCH {\"impl\": \"cpp\"");
    for (const char* k : {"update", "ui", "frame"}) {
        std::vector<double> a(app.timings.at(k).begin() + std::min<size_t>(warmup, app.timings.at(k).size()),
                              app.timings.at(k).end());
        for (double& v : a) v *= 1e3;
        std::sort(a.begin(), a.end());
        double mean = a.empty() ? 0 : std::accumulate(a.begin(), a.end(), 0.0) / a.size();
        auto pct = [&](double p) {  // numpy's default (linear) percentile
            if (a.empty()) return 0.0;
            double idx = p / 100.0 * (a.size() - 1);
            size_t lo = size_t(idx), hi = std::min(lo + 1, a.size() - 1);
            return a[lo] + (a[hi] - a[lo]) * (idx - lo);
        };
        std::printf(", \"%s\": {\"mean_ms\": %.3f, \"median_ms\": %.3f, \"p95_ms\": %.3f}", k, mean, pct(50), pct(95));
    }
    std::printf("}\n");
}

int main(int argc, char** argv) {
    std::string layout = "tall", screenshot;
    int frames = 45, bench = 0;
    double ui_scale = 1.0;
    bool layout_from_cli = false, show_profile = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--layout") { layout = next(); layout_from_cli = true; }
        else if (a == "--screenshot") screenshot = next();
        else if (a == "--frames") frames = std::stoi(next());
        else if (a == "--bench") bench = std::stoi(next());
        else if (a == "--ui-scale") ui_scale = std::stod(next());
        else if (a == "--profile") show_profile = true;
        else if (a == "--assets") g_assets = next();
        else {
            std::printf("usage: %s [--layout stack|tall|fat|grid|horizontal|vertical|splits] [--ui-scale 1.5] [--profile] [--screenshot out.png] [--frames N] "
                        "[--bench N] [--assets DIR]\n", argv[0]);
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }
    if (!fs::exists(g_assets / "fonts")) g_assets = fs::path(argv[0]).parent_path() / "assets";
    if (bench) frames = bench + 30;

    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow* window = glfwCreateWindow(1440, 960, "Shamrock", nullptr, nullptr);
    if (!window) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(bench || !screenshot.empty() ? 0 : 1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // interactive runs remember the arrangement; screenshots and benchmarks always start from the preset
    io.IniFilename = (bench || !screenshot.empty()) ? nullptr : "shamrock_gui_layout.ini";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    load_fonts();
    App::setup_style();

    App app(layout, !screenshot.empty(), frames, bench);
    app.layout_from_cli = layout_from_cli;
    app.profile_from_cli = show_profile;
    app.visible['f'] = app.was_visible['f'] = show_profile;
    g_app = &app;
    {  // layout options live in the same .ini as the dock tree
        ImGuiSettingsHandler h;
        h.TypeName = "Shamrock";
        h.TypeHash = ImHashStr("Shamrock");
        h.ReadOpenFn = ini_open;
        h.ReadLineFn = ini_line;
        h.WriteAllFn = ini_write;
        ImGui::AddSettingsHandler(&h);
    }
    app.ui_scale = std::min(UI::MAX, std::max(UI::MIN, ui_scale));
    app.post_init();

    int fbw = 0, fbh = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        app.pre_frame();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        app.gui();
        ImGui::Render();
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (app.want_exit && !screenshot.empty()) {
            std::vector<uint8_t> px(size_t(fbw) * fbh * 4), flipped(px.size());
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, fbw, fbh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            for (int j = 0; j < fbh; ++j)
                std::memcpy(&flipped[size_t(j) * fbw * 4], &px[size_t(fbh - 1 - j) * fbw * 4], size_t(fbw) * 4);
            for (size_t k = 3; k < flipped.size(); k += 4) flipped[k] = 255;
            stbi_write_png(screenshot.c_str(), fbw, fbh, 4, flipped.data(), fbw * 4);
            std::printf("saved %s\n", screenshot.c_str());
        }
        glfwSwapBuffers(window);
        if (app.want_exit) break;
    }
    if (bench) print_bench(app, 30);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
