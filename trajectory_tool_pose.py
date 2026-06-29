# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 全時刻の軌跡を一度に重ねた静止画の可視化


FILE = "clean1.csv"

POINTS = {
    "tip": "scop - 1",
    "handle": "scop - 2",
    "left": "scop - 3",
    "right": "scop - 4",
}


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

    ax.set_xlim(centers[0] - max_range / 2, centers[0] + max_range / 2)
    ax.set_ylim(centers[1] - max_range / 2, centers[1] + max_range / 2)
    ax.set_zlim(centers[2] - max_range / 2, centers[2] + max_range / 2)


# =====================
# CSV読み込み
# =====================
df = pd.read_csv(FILE)

print("columns:")
print(df.columns.tolist())
print("data shape:", df.shape)

time = pd.to_numeric(df["time"], errors="coerce").to_numpy()

markers = {}
for role, name in POINTS.items():
    markers[role] = read_point(df, name)

# finiteな行だけ使う
valid = np.isfinite(time)
for role in markers:
    valid &= np.all(np.isfinite(markers[role]), axis=1)

time = time[valid]
for role in markers:
    markers[role] = markers[role][valid]

print("valid frames:", len(time))


# =====================
# 3D可視化
# =====================
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# 各点の軌跡を描画
for role, pts in markers.items():
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        label=role,
        linewidth=1.5
    )

# 開始時刻の4点を線でつなぐ
start_pts = np.stack([
    markers["tip"][0],
    markers["handle"][0],
    markers["left"][0],
    markers["right"][0],
])

ax.scatter(
    start_pts[:, 0],
    start_pts[:, 1],
    start_pts[:, 2],
    s=60,
    label="start markers"
)

# スコップっぽく線でつなぐ
p_tip = markers["tip"][0]
p_handle = markers["handle"][0]
p_left = markers["left"][0]
p_right = markers["right"][0]

ax.plot(
    [p_tip[0], p_handle[0]],
    [p_tip[1], p_handle[1]],
    [p_tip[2], p_handle[2]],
    linewidth=3,
    label="tip-handle at start"
)

ax.plot(
    [p_left[0], p_right[0]],
    [p_left[1], p_right[1]],
    [p_left[2], p_right[2]],
    linewidth=3,
    label="left-right at start"
)

all_points = np.concatenate(list(markers.values()), axis=0)
set_axes_equal(ax, all_points)

ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")
ax.set_title("4 marker trajectories of scop")
ax.legend()

plt.tight_layout()
plt.show()