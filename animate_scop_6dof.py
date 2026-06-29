# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 6DoF姿勢推定 + 3Dアニメーション可視化コード
# ============================================================

FILE = "clean1.csv"

POINTS = {
    "tip": "scop - 1",
    "left": "scop - 2",
    "handle": "scop - 3",
    "right": "scop - 4",
}

EDGES = [
    ("tip", "left"),
    ("tip", "right"),
    ("left", "right"),
    ("handle", "left"),
    ("handle", "right"),
]


# ============================================================
# 可視化する時間範囲
# Noneなら最初から最後まで
#
# 例:
# START_TIME = 10.0
# END_TIME = 12.8
# ============================================================
START_TIME = 10
END_TIME = 12.8

# 元CSV上のフレーム番号で指定したい場合
# 基本は None でOK。時間指定の方が使いやすい。
START_ORIGINAL_FRAME = None
END_ORIGINAL_FRAME = None


# ============================================================
# 可視化設定
# ============================================================
STEP = 5
TAIL = 80
INTERVAL = 50

AXIS_LEN = 120.0

# "window": 指定区間にズーム
# "full"  : 全体スケールの中で指定区間を表示
AXIS_SCOPE = "window"

SAVE_POSE_CSV = True
POSE_CSV = "scop_6dof_pose.csv"

SHOW_ROTATION_CHECK = True
ROTATION_CHECK_STEP = 100


# ============================================================
# Utility
# ============================================================
def normalize(v, eps=1e-9):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


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


def set_line_3d(line, p0, p1):
    line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    line.set_3d_properties([p0[2], p1[2]])


# ============================================================
# 6DoF姿勢推定
# ============================================================
def estimate_tool_pose(tip, left, right):
    """
    3点からスコップの6DoF姿勢を推定する。

    origin:
        tip

    x_axis:
        left-right の中心から tip へ向かう方向

    y_axis:
        left から right へ向かう方向

    z_axis:
        x_axis × y_axis
    """
    back_center = (left + right) / 2.0

    x_axis = normalize(tip - back_center)
    y_raw = normalize(right - left)

    if x_axis is None or y_raw is None:
        return None, None

    z_axis = normalize(np.cross(x_axis, y_raw))

    if z_axis is None:
        return None, None

    # 数値誤差で x と y が完全直交しないことがあるので y を作り直す
    y_axis = normalize(np.cross(z_axis, x_axis))

    if y_axis is None:
        return None, None

    R = np.column_stack([x_axis, y_axis, z_axis])
    origin = tip

    return origin, R


def check_rotation_matrix(R):
    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]

    return {
        "norm_x": np.linalg.norm(x),
        "norm_y": np.linalg.norm(y),
        "norm_z": np.linalg.norm(z),
        "dot_xy": np.dot(x, y),
        "dot_yz": np.dot(y, z),
        "dot_zx": np.dot(z, x),
        "det_R": np.linalg.det(R),
    }


def save_pose_csv(time, markers, original_indices, out_path):
    rows = []

    for i in range(len(time)):
        tip = markers["tip"][i]
        left = markers["left"][i]
        right = markers["right"][i]

        origin, R = estimate_tool_pose(tip, left, right)

        if origin is None:
            row = {
                "frame": int(original_indices[i]),
                "time": time[i],
                "origin_x": np.nan,
                "origin_y": np.nan,
                "origin_z": np.nan,
            }

            for name in ["x_axis", "y_axis", "z_axis"]:
                for c in ["x", "y", "z"]:
                    row[f"{name}_{c}"] = np.nan

            for r in range(3):
                for c in range(3):
                    row[f"R{r}{c}"] = np.nan

            rows.append(row)
            continue

        row = {
            "frame": int(original_indices[i]),
            "time": time[i],

            "origin_x": origin[0],
            "origin_y": origin[1],
            "origin_z": origin[2],

            "x_axis_x": R[0, 0],
            "x_axis_y": R[1, 0],
            "x_axis_z": R[2, 0],

            "y_axis_x": R[0, 1],
            "y_axis_y": R[1, 1],
            "y_axis_z": R[2, 1],

            "z_axis_x": R[0, 2],
            "z_axis_y": R[1, 2],
            "z_axis_z": R[2, 2],

            "R00": R[0, 0],
            "R01": R[0, 1],
            "R02": R[0, 2],
            "R10": R[1, 0],
            "R11": R[1, 1],
            "R12": R[1, 2],
            "R20": R[2, 0],
            "R21": R[2, 1],
            "R22": R[2, 2],
        }

        rows.append(row)

    pose_df = pd.DataFrame(rows)
    pose_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved: {out_path}")


# ============================================================
# CSV読み込み
# ============================================================
df = pd.read_csv(FILE)

time = pd.to_numeric(df["time"], errors="coerce").to_numpy()
original_indices = np.arange(len(df))

markers = {}
for role, name in POINTS.items():
    markers[role] = read_point(df, name)

valid = np.isfinite(time)
for role in markers:
    valid &= np.all(np.isfinite(markers[role]), axis=1)

time = time[valid]
original_indices = original_indices[valid]

for role in markers:
    markers[role] = markers[role][valid]

# 全体軸範囲用
all_points_full = np.concatenate(list(markers.values()), axis=0)

# 6DoF pose をCSV保存する場合は、切り出し前・間引き前の全データで保存
if SAVE_POSE_CSV:
    save_pose_csv(time, markers, original_indices, POSE_CSV)


# ============================================================
# 可視化範囲の切り出し
# ============================================================
view_mask = np.ones(len(time), dtype=bool)

if START_TIME is not None:
    view_mask &= time >= START_TIME

if END_TIME is not None:
    view_mask &= time <= END_TIME

if START_ORIGINAL_FRAME is not None:
    view_mask &= original_indices >= START_ORIGINAL_FRAME

if END_ORIGINAL_FRAME is not None:
    view_mask &= original_indices <= END_ORIGINAL_FRAME

time = time[view_mask]
original_indices = original_indices[view_mask]

for role in markers:
    markers[role] = markers[role][view_mask]

if len(time) == 0:
    raise ValueError(
        "指定した START_TIME / END_TIME / START_ORIGINAL_FRAME / END_ORIGINAL_FRAME に該当するデータがありません。"
    )

print("selected time range:", f"{time[0]:.3f}", "to", f"{time[-1]:.3f}")
print("selected original frame range:", int(original_indices[0]), "to", int(original_indices[-1]))


# ============================================================
# 可視化用に間引き
# ============================================================
anim_indices = np.arange(len(time))[::STEP]

time = time[anim_indices]
original_indices = original_indices[anim_indices]

for role in markers:
    markers[role] = markers[role][anim_indices]

print("animation frames:", len(time))

if AXIS_SCOPE == "full":
    all_points = all_points_full
else:
    all_points = np.concatenate(list(markers.values()), axis=0)


# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

set_axes_equal(ax, all_points)

ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Z [mm]")

if START_TIME is None and END_TIME is None:
    title_range = "full range"
else:
    title_range = f"{time[0]:.2f} - {time[-1]:.2f} s"

ax.set_title(f"6DoF pose estimation of scop ({title_range})")


# ============================================================
# 現在位置の点
# ============================================================
scatters = {}
for role in ["tip", "left", "handle", "right"]:
    scatters[role] = ax.scatter([], [], [], s=60, label=role)


# ============================================================
# スコップ形状の線
# ============================================================
edge_lines = {}

for a, b in EDGES:
    line, = ax.plot([], [], [], lw=2.5, label=f"{a}-{b}")
    edge_lines[(a, b)] = line


# ============================================================
# tip の軌跡
# ============================================================
tip_trail, = ax.plot([], [], [], lw=2, alpha=0.8, label="tip trail")


# ============================================================
# 6DoF姿勢軸
# x: red, y: green, z: blue
# ============================================================
x_axis_line, = ax.plot([], [], [], lw=3, color="red", label="tool x-axis")
y_axis_line, = ax.plot([], [], [], lw=3, color="green", label="tool y-axis")
z_axis_line, = ax.plot([], [], [], lw=3, color="blue", label="tool z-axis")


time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

ax.legend(loc="upper right")


# ============================================================
# アニメーション更新
# ============================================================
def update(frame):
    # 各マーカーの現在位置
    for role, scatter in scatters.items():
        p = markers[role][frame]
        scatter._offsets3d = ([p[0]], [p[1]], [p[2]])

    # スコップ形状
    for (a, b), line in edge_lines.items():
        pa = markers[a][frame]
        pb = markers[b][frame]
        set_line_3d(line, pa, pb)

    # tip の過去軌跡
    start = max(0, frame - TAIL)
    trail = markers["tip"][start:frame + 1]

    tip_trail.set_data(trail[:, 0], trail[:, 1])
    tip_trail.set_3d_properties(trail[:, 2])

    # 6DoF姿勢推定
    tip = markers["tip"][frame]
    left = markers["left"][frame]
    right = markers["right"][frame]

    origin, R = estimate_tool_pose(tip, left, right)

    # 回転行列の値チェック
    if SHOW_ROTATION_CHECK and frame % ROTATION_CHECK_STEP == 0 and origin is not None:
        check = check_rotation_matrix(R)
        print(
            f"animation frame={frame}, "
            f"original frame={int(original_indices[frame])}, "
            f"time={time[frame]:.3f}, "
            f"check={check}"
        )

    if origin is not None:
        x_end = origin + R[:, 0] * AXIS_LEN
        y_end = origin + R[:, 1] * AXIS_LEN
        z_end = origin + R[:, 2] * AXIS_LEN

        set_line_3d(x_axis_line, origin, x_end)
        set_line_3d(y_axis_line, origin, y_end)
        set_line_3d(z_axis_line, origin, z_end)
    else:
        empty = np.array([np.nan, np.nan, np.nan])
        set_line_3d(x_axis_line, empty, empty)
        set_line_3d(y_axis_line, empty, empty)
        set_line_3d(z_axis_line, empty, empty)

    total_frames = len(time)
    original_frame = int(original_indices[frame])

    time_text.set_text(
        f"time = {time[frame]:.2f} s\n"
        f"animation frame = {frame} / {total_frames - 1}\n"
        f"original frame = {original_frame}"
    )

    return (
        *scatters.values(),
        *edge_lines.values(),
        tip_trail,
        x_axis_line,
        y_axis_line,
        z_axis_line,
        time_text,
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