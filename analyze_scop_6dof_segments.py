# -*- coding: utf-8 -*-
"""
analyze_scop_6dof_segments.py

目的:
    scop_6dof_pose.csv から
    - スコップ先端位置
    - tip速度
    - 回転行列Rの変化から角速度
    - 姿勢軸の時間変化
    を可視化し、基本動作区間を切るための材料を作る。

入力:
    scop_6dof_pose.csv

出力:
    scop_motion_features.csv
    scop_auto_candidate_segments.csv
    scop_motion_features.png

使い方:
    python analyze_scop_6dof_segments.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 設定
# =========================
INPUT_CSV = "scop_6dof_pose.csv"

FEATURE_CSV = "scop_motion_features.csv"
AUTO_SEGMENT_CSV = "scop_auto_candidate_segments.csv"
PLOT_PNG = "scop_motion_features.png"

# 表示範囲を絞りたい場合は秒で指定
# 例: START_TIME = 10.0, END_TIME = 12.0
START_TIME = 7
END_TIME = 12.8

# 急変候補を出すためのしきい値
# 上位何%を「速い」とみなすか
SPEED_PERCENTILE = 90
ANGULAR_SPEED_PERCENTILE = 90

# 自動候補区間の結合条件
MIN_SEGMENT_DURATION = 0.10  # 秒未満の短すぎる区間は無視
MERGE_GAP = 0.15             # 秒以内の隣接候補は結合

# 手動で区間を決めたら、ここに書く
# 最初は空でOK。グラフを見ながら後で埋める。
MANUAL_SEGMENTS = [
    # ("approach", 0.00, 4.40),
    # ("insert",   4.40, 5.30),
    # ("scrape",   5.30, 10.80),
    # ("dump",    10.80, 11.40),
    # ("retract", 11.40, 13.00),
]

# MANUAL_SEGMENTS = [
#     ("scrape_or_scoop", 7.0, 9.0),
#     ("lift",            9.0, 10.0),
#     ("dump",           10.0, 11.0),
#     ("retract",        11.0, 12.8),
# ]

MANUAL_SEGMENTS = [
    ("scrape_or_scoop", 7.0, 9.0),
    ("lift",            9.0, 10.9),
    ("dump_main",      10.9, 11.45),
    ("retract",        11.45, 12.8),
]


# =========================
# 計算関数
# =========================
def rotation_angle_from_R(R_rel):
    """
    相対回転行列から回転角[rad]を求める。
    angle = arccos((trace(R)-1)/2)
    """
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def make_rotation_matrices(df):
    """
    CSVの R00 ... R22 から Nx3x3 の回転行列列を作る。
    """
    Rs = []

    for _, row in df.iterrows():
        R = np.array([
            [row["R00"], row["R01"], row["R02"]],
            [row["R10"], row["R11"], row["R12"]],
            [row["R20"], row["R21"], row["R22"]],
        ], dtype=float)
        Rs.append(R)

    return np.stack(Rs, axis=0)


def compute_motion_features(df):
    """
    位置速度・角速度・姿勢軸の変化を計算する。
    """
    time = df["time"].to_numpy(dtype=float)

    pos = df[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=float)
    Rs = make_rotation_matrices(df)

    n = len(df)

    speed = np.full(n, np.nan)
    angular_speed = np.full(n, np.nan)
    frame_rotation_deg = np.full(n, np.nan)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vz = np.full(n, np.nan)

    for i in range(1, n):
        dt = time[i] - time[i - 1]
        if dt <= 0:
            continue

        dp = pos[i] - pos[i - 1]

        vx[i] = dp[0] / dt
        vy[i] = dp[1] / dt
        vz[i] = dp[2] / dt
        speed[i] = np.linalg.norm(dp) / dt

        # R_prev から R_curr への相対回転
        R_prev = Rs[i - 1]
        R_curr = Rs[i]
        R_rel = R_prev.T @ R_curr

        angle_rad = rotation_angle_from_R(R_rel)
        frame_rotation_deg[i] = np.degrees(angle_rad)
        angular_speed[i] = np.degrees(angle_rad / dt)

    out = df.copy()

    out["tip_speed_mm_s"] = speed
    out["tip_vx_mm_s"] = vx
    out["tip_vy_mm_s"] = vy
    out["tip_vz_mm_s"] = vz

    out["frame_rotation_deg"] = frame_rotation_deg
    out["angular_speed_deg_s"] = angular_speed

    # 姿勢軸の時間変化確認用
    # 連続フレームで各軸がどれくらい向きを変えたか
    for axis_name in ["x_axis", "y_axis", "z_axis"]:
        axis = df[[f"{axis_name}_x", f"{axis_name}_y", f"{axis_name}_z"]].to_numpy(dtype=float)
        axis_change_deg = np.full(n, np.nan)

        for i in range(1, n):
            dot = np.dot(axis[i - 1], axis[i])
            dot = np.clip(dot, -1.0, 1.0)
            axis_change_deg[i] = np.degrees(np.arccos(dot))

        out[f"{axis_name}_change_deg"] = axis_change_deg

    return out


def boolean_to_segments(time, mask, min_duration=0.1, merge_gap=0.15):
    """
    True/False のmaskから候補区間を作る。
    """
    time = np.asarray(time)
    mask = np.asarray(mask, dtype=bool)

    raw_segments = []
    in_seg = False
    start_idx = None

    for i, flag in enumerate(mask):
        if flag and not in_seg:
            in_seg = True
            start_idx = i
        elif not flag and in_seg:
            end_idx = i - 1
            raw_segments.append([start_idx, end_idx])
            in_seg = False

    if in_seg:
        raw_segments.append([start_idx, len(mask) - 1])

    if not raw_segments:
        return []

    # 近い候補を結合
    merged = [raw_segments[0]]
    for s, e in raw_segments[1:]:
        prev_s, prev_e = merged[-1]
        gap = time[s] - time[prev_e]

        if gap <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # 短すぎる候補を除外
    final_segments = []
    for s, e in merged:
        duration = time[e] - time[s]
        if duration >= min_duration:
            final_segments.append((s, e))

    return final_segments


def make_auto_segments(features):
    """
    tip速度または角速度が大きい場所を急変候補として抽出する。
    """
    speed = features["tip_speed_mm_s"].to_numpy(dtype=float)
    angular_speed = features["angular_speed_deg_s"].to_numpy(dtype=float)
    time = features["time"].to_numpy(dtype=float)

    speed_thr = np.nanpercentile(speed, SPEED_PERCENTILE)
    angular_thr = np.nanpercentile(angular_speed, ANGULAR_SPEED_PERCENTILE)

    rapid_mask = (speed >= speed_thr) | (angular_speed >= angular_thr)

    segments = boolean_to_segments(
        time,
        rapid_mask,
        min_duration=MIN_SEGMENT_DURATION,
        merge_gap=MERGE_GAP,
    )

    rows = []

    for segment_id, (s, e) in enumerate(segments):
        part = features.iloc[s:e + 1]

        rows.append({
            "segment_id": segment_id,
            "kind": "rapid_candidate",
            "start_frame": int(s),
            "end_frame": int(e),
            "start_time": float(features.iloc[s]["time"]),
            "end_time": float(features.iloc[e]["time"]),
            "duration": float(features.iloc[e]["time"] - features.iloc[s]["time"]),
            "max_tip_speed_mm_s": float(np.nanmax(part["tip_speed_mm_s"])),
            "mean_tip_speed_mm_s": float(np.nanmean(part["tip_speed_mm_s"])),
            "max_angular_speed_deg_s": float(np.nanmax(part["angular_speed_deg_s"])),
            "mean_angular_speed_deg_s": float(np.nanmean(part["angular_speed_deg_s"])),
        })

    return pd.DataFrame(rows), speed_thr, angular_thr


def export_manual_segments(features):
    """
    MANUAL_SEGMENTS に書いた区間をCSVとして保存する。
    """
    if len(MANUAL_SEGMENTS) == 0:
        return None

    rows = []

    for segment_id, (primitive, start_time, end_time) in enumerate(MANUAL_SEGMENTS):
        part = features[(features["time"] >= start_time) & (features["time"] <= end_time)]

        if len(part) == 0:
            continue

        rows.append({
            "segment_id": segment_id,
            "primitive": primitive,
            "start_frame": int(part.index[0]),
            "end_frame": int(part.index[-1]),
            "start_time": float(part["time"].iloc[0]),
            "end_time": float(part["time"].iloc[-1]),
            "duration": float(part["time"].iloc[-1] - part["time"].iloc[0]),
            "max_tip_speed_mm_s": float(np.nanmax(part["tip_speed_mm_s"])),
            "mean_tip_speed_mm_s": float(np.nanmean(part["tip_speed_mm_s"])),
            "max_angular_speed_deg_s": float(np.nanmax(part["angular_speed_deg_s"])),
            "mean_angular_speed_deg_s": float(np.nanmean(part["angular_speed_deg_s"])),
            "start_origin_x": float(part["origin_x"].iloc[0]),
            "start_origin_y": float(part["origin_y"].iloc[0]),
            "start_origin_z": float(part["origin_z"].iloc[0]),
            "end_origin_x": float(part["origin_x"].iloc[-1]),
            "end_origin_y": float(part["origin_y"].iloc[-1]),
            "end_origin_z": float(part["origin_z"].iloc[-1]),
        })

    manual_df = pd.DataFrame(rows)
    manual_df.to_csv("scop_manual_segments.csv", index=False, encoding="utf-8-sig")
    print("saved: scop_manual_segments.csv")

    return manual_df



def get_plot_time_range(features):
    """
    START_TIME / END_TIME が None の場合は全区間。
    指定がある場合はその区間を返す。
    """
    t_min = float(features["time"].min())
    t_max = float(features["time"].max())

    plot_start = START_TIME if START_TIME is not None else t_min
    plot_end = END_TIME if END_TIME is not None else t_max

    if plot_start > plot_end:
        plot_start, plot_end = plot_end, plot_start

    return plot_start, plot_end


def filter_auto_segments_for_plot(auto_segments, plot_start, plot_end):
    """
    auto_segments のうち、表示範囲に重なるものだけ残す。
    表示範囲からはみ出す部分はクリップする。
    """
    if auto_segments is None or len(auto_segments) == 0:
        return auto_segments

    rows = []

    for _, row in auto_segments.iterrows():
        seg_start = float(row["start_time"])
        seg_end = float(row["end_time"])

        # 表示範囲と全く重ならない場合は除外
        if seg_end < plot_start or seg_start > plot_end:
            continue

        new_row = row.copy()
        new_row["start_time"] = max(seg_start, plot_start)
        new_row["end_time"] = min(seg_end, plot_end)
        rows.append(new_row)

    if len(rows) == 0:
        return pd.DataFrame(columns=auto_segments.columns)

    return pd.DataFrame(rows)


def filter_manual_segments_for_plot(plot_start, plot_end):
    """
    MANUAL_SEGMENTS のうち、表示範囲に重なるものだけ返す。
    """
    filtered = []

    for primitive, start_time, end_time in MANUAL_SEGMENTS:
        if end_time < plot_start or start_time > plot_end:
            continue

        filtered.append((
            primitive,
            max(start_time, plot_start),
            min(end_time, plot_end),
        ))

    return filtered


def shade_segments(ax, auto_segments=None, manual_segments=None):
    """
    グラフ上に候補区間・手動区間を薄く表示する。
    すでに表示範囲でフィルタ済みのものだけを描く。
    """
    if auto_segments is not None and len(auto_segments) > 0:
        for _, row in auto_segments.iterrows():
            ax.axvspan(
                row["start_time"],
                row["end_time"],
                alpha=0.12,
                color="tab:blue"
            )

    if manual_segments is not None and len(manual_segments) > 0:
        for primitive, start_time, end_time in manual_segments:
            ax.axvspan(
                start_time,
                end_time,
                alpha=0.22,
                color="tab:orange"
            )

            mid = (start_time + end_time) / 2
            ymin, ymax = ax.get_ylim()

            ax.text(
                mid,
                ymax,
                primitive,
                ha="center",
                va="top",
                fontsize=9,
                rotation=0,
            )


def plot_features(features, auto_segments, speed_thr, angular_thr):
    """
    区間分割のための特徴量グラフを表示・保存する。

    START_TIME / END_TIME が None:
        全区間を表示

    START_TIME / END_TIME が指定あり:
        その区間だけを拡大表示
    """
    plot_start, plot_end = get_plot_time_range(features)

    # 表示データを切り出す
    data = features[
        (features["time"] >= plot_start) &
        (features["time"] <= plot_end)
    ].copy()

    if len(data) == 0:
        raise ValueError(
            f"指定した範囲にデータがありません: {plot_start} - {plot_end} s"
        )

    # 表示範囲に重なるsegmentだけ残す
    auto_segments_plot = filter_auto_segments_for_plot(
        auto_segments,
        plot_start,
        plot_end,
    )

    manual_segments_plot = filter_manual_segments_for_plot(
        plot_start,
        plot_end,
    )

    t = data["time"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    # 1. 先端位置
    axes[0].plot(t, data["origin_x"], label="origin_x")
    axes[0].plot(t, data["origin_y"], label="origin_y")
    axes[0].plot(t, data["origin_z"], label="origin_z")
    axes[0].set_ylabel("Position [mm]")
    axes[0].set_title("Tool tip position")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # 2. 先端速度
    axes[1].plot(t, data["tip_speed_mm_s"], label="tip speed")
    axes[1].axhline(
        speed_thr,
        linestyle="--",
        label=f"speed threshold p{SPEED_PERCENTILE}"
    )
    axes[1].set_ylabel("Speed [mm/s]")
    axes[1].set_title("Tool tip speed")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # 3. 角速度
    axes[2].plot(t, data["angular_speed_deg_s"], label="angular speed")
    axes[2].axhline(
        angular_thr,
        linestyle="--",
        label=f"angular threshold p{ANGULAR_SPEED_PERCENTILE}"
    )
    axes[2].set_ylabel("Angular speed [deg/s]")
    axes[2].set_title("Orientation change speed")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    # 4. 姿勢軸の連続変化
    axes[3].plot(t, data["x_axis_change_deg"], label="x_axis change")
    axes[3].plot(t, data["y_axis_change_deg"], label="y_axis change")
    axes[3].plot(t, data["z_axis_change_deg"], label="z_axis change")
    axes[3].set_ylabel("Axis change [deg/frame]")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_title("Frame-to-frame axis direction change")
    axes[3].grid(True)
    axes[3].legend(loc="upper right")

    # 区間表示
    for ax in axes:
        shade_segments(
            ax,
            auto_segments=auto_segments_plot,
            manual_segments=manual_segments_plot,
        )

        # ここが重要：x軸を明示的に固定
        ax.set_xlim(plot_start, plot_end)

    if START_TIME is None and END_TIME is None:
        fig.suptitle("Motion features: full range", fontsize=14)
        output_png = PLOT_PNG
    else:
        fig.suptitle(
            f"Motion features: {plot_start:.2f} - {plot_end:.2f} s",
            fontsize=14
        )
        output_png = f"scop_motion_features_{plot_start:.2f}_{plot_end:.2f}s.png"

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"saved: {output_png}")

    plt.show()

# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    # time順に並べてindexを振り直す
    df = df.sort_values("time").reset_index(drop=True)

    required_cols = [
        "time",
        "origin_x", "origin_y", "origin_z",
        "R00", "R01", "R02",
        "R10", "R11", "R12",
        "R20", "R21", "R22",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")

    features = compute_motion_features(df)
    features.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {FEATURE_CSV}")

    auto_segments, speed_thr, angular_thr = make_auto_segments(features)
    auto_segments.to_csv(AUTO_SEGMENT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {AUTO_SEGMENT_CSV}")

    print("\n=== thresholds ===")
    print(f"speed threshold: {speed_thr:.2f} mm/s")
    print(f"angular speed threshold: {angular_thr:.2f} deg/s")

    print("\n=== rapid candidate segments ===")
    if len(auto_segments) == 0:
        print("no candidate segments")
    else:
        print(auto_segments[[
            "segment_id",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "duration",
            "max_tip_speed_mm_s",
            "max_angular_speed_deg_s",
        ]].to_string(index=False))

    export_manual_segments(features)

    plot_features(features, auto_segments, speed_thr, angular_thr)


if __name__ == "__main__":
    main()