# Regenerates docs/catalog-banner.png (run from docs/): python scripts/make_banner.py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke

W, H, DPI = 1280, 640, 100
BG = np.array([7, 11, 20]) / 255
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor(BG)
rng = np.random.default_rng(7)
# background spectrogram, full width
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
t = np.linspace(-40, 40, 1280)
f = np.linspace(0, 1000, 400)
c = 343.0


def fr(f0, tc, v, d):
    x = v * (t - tc)
    return f0 * c / (c + v * x / np.sqrt(x**2 + d**2))


S = rng.gamma(1.0, 0.12, (f.size, t.size))
for k, f0 in enumerate([110, 220, 330, 440, 550, 660, 770, 880]):
    S += (1.3 - 0.1 * k) * np.exp(
        -(((f[:, None] - fr(f0, -8, 80, 900)[None, :]) / 3.5) ** 2)
    )
for f0 in [170, 340, 510, 680, 850]:
    S += 0.45 * np.exp(-(((f[:, None] - fr(f0, 14, 120, 2200)[None, :]) / 4.0) ** 2))
S = np.log1p(S)
cm = LinearSegmentedColormap.from_list(
    "d", [BG, "#0a2033", "#0d4660", "#1aa6c3", "#9af4ff"]
)
ax.imshow(
    S,
    origin="lower",
    aspect="auto",
    cmap=cm,
    extent=[0, 1, 0, 1],
    vmin=0.05,
    vmax=1.05,
    interpolation="bilinear",
)
xx0 = np.linspace(0, 1, t.size)
m = xx0 < 0.62
ax.plot(xx0[m], fr(330, -8, 80, 900)[m] / 1000, color="#ff8a3d", lw=2.4, alpha=0.95)
# left-to-right darkening so text stays readable, and a right vignette behind the map
xg = np.linspace(0, 1, 512)
a = np.clip(0.92 - 1.25 * xg, 0, 0.92)[None, :] ** 0.8
ax.imshow(
    np.dstack([np.broadcast_to(BG, (1, 512, 3)), a[..., None]]),
    extent=[0, 1, 0, 1],
    aspect="auto",
)
yy, xx = np.mgrid[0:1:200j, 0:1:400j]
r = np.hypot((xx - 0.78) * 2.0, (yy - 0.5) * 1.0)
ax.imshow(
    np.dstack(
        [
            np.broadcast_to(BG, (200, 400, 3)),
            np.clip(1.15 - 1.25 * r, 0, 0.88)[..., None],
        ]
    ),
    extent=[0, 1, 0, 1],
    aspect="auto",
    origin="lower",
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# map panel
am = fig.add_axes([0.60, 0.06, 0.38, 0.88])
am.set_axis_off()
R = 9.5
clip = Circle((0, 0), R, transform=am.transData)
for r in np.arange(1.6, R, 1.6):
    am.add_patch(Circle((0, 0), r, fill=False, color="#1a3a55", lw=0.8))
am.add_patch(Circle((0, 0), R, fill=False, color="#2b5878", lw=1.3))
for ang in np.arange(0, 180, 30):
    th = np.radians(ang)
    (l,) = am.plot(
        [-R * np.cos(th), R * np.cos(th)],
        [-R * np.sin(th), R * np.sin(th)],
        color="#132d44",
        lw=0.7,
    )
x = np.linspace(-R, R, 400)
X, Y = np.meshgrid(x, x)
cx, cy = 1.7, -0.8
th = 0.55
u = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
w = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
P = np.exp(-(u**2 / 2.2 + w**2 / 0.5))
hm = LinearSegmentedColormap.from_list(
    "h",
    [(1, 0.54, 0.24, 0), (1, 0.5, 0.2, 0.35), (1, 0.72, 0.3, 0.8), (1, 0.97, 0.85, 1)],
)
im = am.imshow(P, extent=[-R, R, -R, R], origin="lower", cmap=hm, zorder=2)
im.set_clip_path(clip)
cs = am.contour(X, Y, P, levels=[0.2], colors="#ffb36b", linewidths=1.3, zorder=3)
s = np.linspace(-R * 1.2, R * 1.2, 300)
for ang, off, col, lw in [
    (-0.32, 0.95, "#9af4ff", 2.4),
    (1.2, -2.1, "#4cc9e6", 1.5),
    (2.55, 4.6, "#2a8fb0", 1.2),
]:
    xs = s * np.cos(ang) - off * np.sin(ang)
    ys = s * np.sin(ang) + off * np.cos(ang)
    (ln,) = am.plot(xs, ys, color=col, lw=lw, zorder=4, alpha=0.95)
    ln.set_clip_path(clip)
    i = 205
    ar = am.annotate(
        "",
        xy=(xs[i + 8], ys[i + 8]),
        xytext=(xs[i], ys[i]),
        arrowprops={
            "arrowstyle": "-|>,head_width=0.35,head_length=0.7",
            "color": col,
            "lw": lw,
        },
        zorder=5,
    )
am.plot(cx, cy, "o", ms=9, mfc="white", mec="#ff8a3d", mew=2.6, zorder=7)
am.add_patch(
    Circle((cx, cy), 0.75, fill=False, color="#ff8a3d", lw=1.1, alpha=0.7, zorder=7)
)
am.text(
    cx + 0.95,
    cy - 1.25,
    "recording",
    color="#ffcf9e",
    fontsize=11,
    zorder=8,
    path_effects=[withStroke(linewidth=4, foreground=BG)],
)
am.set_xlim(-R - 0.3, R + 0.3)
am.set_ylim(-R - 0.3, R + 0.3)
am.set_aspect("equal")
pe = [withStroke(linewidth=6, foreground=BG)]
fig.text(
    0.045,
    0.74,
    "Aircraft Doppler",
    fontsize=52,
    fontweight="bold",
    color="white",
    path_effects=pe,
)
fig.text(
    0.045,
    0.625,
    "Geolocation",
    fontsize=52,
    fontweight="bold",
    color="#9af4ff",
    path_effects=pe,
)
fig.text(
    0.047,
    0.53,
    "Find where a recording was made from the Doppler",
    fontsize=17,
    color="#d2deea",
    path_effects=pe,
)
fig.text(
    0.047,
    0.48,
    "of aircraft overhead, matched against ADS-B tracks.",
    fontsize=17,
    color="#d2deea",
    path_effects=pe,
)
fig.text(
    0.047,
    0.08,
    "Hermes plugin  ·  tonal + broadband detection  ·  95% credible regions",
    fontsize=13,
    color="#8ea3b8",
    path_effects=pe,
)
fig.savefig("catalog-banner.png", dpi=DPI, facecolor=BG)
