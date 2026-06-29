# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 4点をつないだ、スコップの軌道の可視化コード


FILE = "clean1.csv"

# マーカー対応
POINTS = {
    "tip": "scop - 1",      # スコップ先端
    "left": "scop - 2",     # 柄に近い側の左端
    "handle": "scop - 3",   # 柄との接続部あたり
    "right": "scop - 4",    # 柄に近い側の右端
}

# スコップ形状としてつなぐ線
EDGES = [
    ("tip", "left"),
    ("tip", "right"),
    ("left", "right"),
    ("handle", "left"),
    ("handle", "right"),
]

STEP = 5          # 何フレームおきに描画するか
TAIL = 80         # tip軌跡を何点分残すか
INTERVAL = 50     # アニメーション更新間隔[ms]


def read_point(df, name):
    x = pd.to_numeric(df[f"{name}_x"], errors="coerce")
    y = pd.to_numeric(df[f"{name}_y"], errors="coerce")
    z = pd.to_numeric(df[f"{name}_z"], errors="coerce")
    return np.stack([x, y, z], axis=1)


def set_axes_equal(ax, points):
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2
    max_range = np.max(maxs - mins)

    if max_range == 0:
        max_range = 1.0

    ax.set_xlim(centers[0] - max_range / 2, centers[0] + max_range / 2)
    ax.set_ylim(centers[1] - max_range / 2, centers[1] + max_range / 2)
    ax.set_zlim(centers[2] - max_range / 2, centers[2] + max_range / 2)


# =====================
# CSV読み込み
# =====================
df = pd.read_csv(FILE)

time = pd.to_numeric(df["time"], errors="coerce").to_numpy()

markers = {}
for role, name in POINTS.items():
    markers[role] = read_point(df, name)

# 有効な行だけ使う
valid = np.isfinite(time)
for role in markers:
    valid &= np.all(np.isfinite(markers[role]), axis=1)

time = time[valid]
for role in markers:
    markers[role] = markers[role][valid]

# 間引き
time = time[::STEP]
for role in markers:
    markers[role] = markers[role][::STEP]

print("frames:", len(time))

all_points = np.concatenate(list(markers.values()), axis=0)


# =====================
# Figure
# =====================
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

set_axes_equal(ax, all_points)

ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.set_title("Animated 4-marker motion of scop")


# =====================
# 現在位置の点
# =====================
scatters = {}

scatters["tip"] = ax.scatter([], [], [], s=60, label="tip")
scatters["left"] = ax.scatter([], [], [], s=60, label="left")
scatters["handle"] = ax.scatter([], [], [], s=60, label="handle")
scatters["right"] = ax.scatter([], [], [], s=60, label="right")


# =====================
# スコップ形状の線
# =====================
edge_lines = {}

for a, b in EDGES:
    line, = ax.plot([], [], [], lw=2.5, label=f"{a}-{b}")
    edge_lines[(a, b)] = line


# =====================
# tip の軌跡
# =====================
tip_trail, = ax.plot([], [], [], lw=2, alpha=0.8, label="tip trail")


# =====================
# 時刻表示
# =====================
time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

ax.legend(loc="upper right")


# =====================
# アニメーション更新関数
# =====================
def update(frame):
    # 各マーカーの現在位置を更新
    for role, scatter in scatters.items():
        p = markers[role][frame]
        scatter._offsets3d = ([p[0]], [p[1]], [p[2]])

    # スコップ形状の線を更新
    for (a, b), line in edge_lines.items():
        pa = markers[a][frame]
        pb = markers[b][frame]

        line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
        line.set_3d_properties([pa[2], pb[2]])

    # tip の過去軌跡を更新
    start = max(0, frame - TAIL)
    trail = markers["tip"][start:frame + 1]

    tip_trail.set_data(trail[:, 0], trail[:, 1])
    tip_trail.set_3d_properties(trail[:, 2])

    # 時刻表示を更新
    time_text.set_text(f"time = {time[frame]:.2f} s   frame = {frame}")

    return (
        *scatters.values(),
        *edge_lines.values(),
        tip_trail,
        time_text
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(time),
    interval=INTERVAL,
    blit=False
)

plt.tight_layout()
plt.show()