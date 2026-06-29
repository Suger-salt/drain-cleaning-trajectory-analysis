# -*- coding: utf-8 -*-
"""
extract_primitive_parameters.py

目的:
    scop_motion_features.csv と scop_manual_segments.csv から、
    手動で区切った基本動作ごとの特徴量をまとめる。

入力:
    scop_motion_features.csv
    scop_manual_segments.csv

出力:
    scop_primitive_features.csv
    primitive_parameters.json

実行:
    python extract_primitive_parameters.py
"""

import json
import os
import numpy as np
import pandas as pd


# =========================
# 入出力ファイル
# =========================
FEATURE_CSV = "scop_motion_features.csv"
SEGMENT_CSV = "scop_manual_segments.csv"

OUT_FEATURE_CSV = "scop_primitive_features.csv"
OUT_JSON = "primitive_parameters.json"


# =========================
# scop_manual_segments.csv が無い場合の予備
# =========================
MANUAL_SEGMENTS = [
    ("scrape_or_scoop", 7.0, 9.0),
    ("lift",            9.0, 10.9),
    ("dump_main",      10.9, 11.45),
    ("retract",        11.45, 12.8),
]


# =========================
# 今回の作業座標の解釈
# =========================
TASK_AXIS_LABELS = {
    "x": "longitudinal",      # 側溝長手方向
    "y": "width_or_depth",    # 幅方向 / 奥行き方向
    "z": "height",            # 高さ方向
}


# =========================
# Utility
# =========================
def safe_float(x):
    if pd.isna(x):
        return None
    return float(x)


def safe_int(x):
    if pd.isna(x):
        return None
    return int(x)


def unit_vector(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n < eps:
        return np.array([0.0, 0.0, 0.0])

    return v / n


def path_length_mm(pos):
    """
    軌跡長を計算する。
    """
    if len(pos) < 2:
        return 0.0

    diff = np.diff(pos, axis=0)
    length = np.sum(np.linalg.norm(diff, axis=1))
    return float(length)


def positive_ratio(values):
    """
    値が増加している割合。
    例:
        x_positive_ratio が高い
        → x方向へ継続的に進んでいる
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    d = np.diff(values)
    return float(np.mean(d > 0))


def negative_ratio(values):
    """
    値が減少している割合。
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    d = np.diff(values)
    return float(np.mean(d < 0))


def make_R_from_row(row):
    """
    R00 ... R22 から回転行列を作る。
    """
    return np.array([
        [row["R00"], row["R01"], row["R02"]],
        [row["R10"], row["R11"], row["R12"]],
        [row["R20"], row["R21"], row["R22"]],
    ], dtype=float)


def rotation_angle_deg(R_start, R_end):
    """
    R_start から R_end までの姿勢変化量[deg]を計算する。
    """
    R_rel = R_start.T @ R_end

    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def load_segments():
    """
    scop_manual_segments.csv があればそれを使う。
    なければ MANUAL_SEGMENTS を使う。
    """
    if os.path.exists(SEGMENT_CSV):
        seg_df = pd.read_csv(SEGMENT_CSV)

        required = ["primitive", "start_time", "end_time"]
        for col in required:
            if col not in seg_df.columns:
                raise ValueError(f"{SEGMENT_CSV} に {col} 列がありません。")

        segments = []

        for i, row in seg_df.iterrows():
            segments.append({
                "segment_id": safe_int(row["segment_id"]) if "segment_id" in seg_df.columns else i,
                "primitive": str(row["primitive"]),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
            })

        print(f"loaded: {SEGMENT_CSV}")
        return segments

    print(f"{SEGMENT_CSV} が見つからないため、コード内の MANUAL_SEGMENTS を使います。")

    segments = []
    for i, (primitive, start_time, end_time) in enumerate(MANUAL_SEGMENTS):
        segments.append({
            "segment_id": i,
            "primitive": primitive,
            "start_time": start_time,
            "end_time": end_time,
        })

    return segments


# =========================
# ロボット翻訳の初期ルール
# =========================
def make_robot_translation_hint(primitive, feature):
    """
    人間動作をロボット用primitiveへ変換するためのヒント。
    ここではまだロボット制御はしない。
    """

    dx = feature["delta_x"]
    dy = feature["delta_y"]
    dz = feature["delta_z"]

    if primitive == "scrape_or_scoop":
        return {
            "robot_primitive": "guarded_scrape_or_scoop",
            "human_motion_usage": "partial",
            "translation_policy": "人間動作から作業方向・距離・姿勢を抽出し、ロボットでは低速・力制限付きで再生成する。",
            "task_direction": feature["dominant_axis_signed"],
            "suggested_distance_mm": min(abs(dx), 300.0),
            "suggested_speed": "slow",
            "force_policy": "force_limited",
            "safety_rule": "底面へ強く押し込まない。力が急増したら停止または後退する。"
        }

    if primitive == "lift":
        return {
            "robot_primitive": "lift",
            "human_motion_usage": "partial",
            "translation_policy": "人間動作から持ち上げ意図と高さ変化を抽出し、ロボットでは安定姿勢で上方へ移動する。",
            "suggested_lift_height_mm": max(dz, 0.0),
            "suggested_speed": "moderate",
            "orientation_policy": "keep_or_stabilize_tool_orientation",
            "safety_rule": "持ち上げ中に泥を落とさないよう姿勢変化を抑える。"
        }

    if primitive == "dump_main":
        return {
            "robot_primitive": "safe_dump",
            "human_motion_usage": "intent_only",
            "translation_policy": "人間の高速投げ捨て軌道は直接模倣せず、排出意図として扱い、安全な低速傾け動作へ置換する。",
            "human_motion_character": "high_speed_throwing",
            "suggested_dump_strategy": "move_to_dump_area_then_slow_tilt",
            "suggested_speed": "slow_tilt",
            "safety_rule": "高速な角速度を避け、周囲へ泥を飛散させない。"
        }

    if primitive == "retract":
        return {
            "robot_primitive": "safe_retract",
            "human_motion_usage": "partial",
            "translation_policy": "人間動作から復帰意図を抽出し、ロボットでは安全姿勢または次の開始姿勢へ戻る。",
            "suggested_target": "safe_pose_or_next_start_pose",
            "suggested_speed": "moderate_or_slow",
            "safety_rule": "側溝壁面・底面・未知物体との衝突を避ける。"
        }

    return {
        "robot_primitive": "unknown",
        "human_motion_usage": "unknown",
        "translation_policy": "このprimitiveに対する変換ルールは未定義。"
    }


# =========================
# 特徴量抽出
# =========================
def extract_segment_features(features_df, segment):
    primitive = segment["primitive"]
    start_time = segment["start_time"]
    end_time = segment["end_time"]

    part = features_df[
        (features_df["time"] >= start_time) &
        (features_df["time"] <= end_time)
    ].copy()

    if len(part) == 0:
        raise ValueError(
            f"指定区間にデータがありません: {primitive}, {start_time} - {end_time}"
        )

    start_row = part.iloc[0]
    end_row = part.iloc[-1]

    if "frame" in part.columns:
        start_frame = int(start_row["frame"])
        end_frame = int(end_row["frame"])
    else:
        start_frame = int(part.index[0])
        end_frame = int(part.index[-1])

    # 位置
    pos = part[["origin_x", "origin_y", "origin_z"]].to_numpy(dtype=float)
    p_start = pos[0]
    p_end = pos[-1]

    delta = p_end - p_start
    dx, dy, dz = delta

    displacement = float(np.linalg.norm(delta))
    path_len = path_length_mm(pos)

    if path_len > 1e-9:
        straightness = displacement / path_len
    else:
        straightness = np.nan

    # 支配的な移動軸
    axis_names = ["x", "y", "z"]
    dominant_axis_idx = int(np.argmax(np.abs(delta)))
    dominant_axis = axis_names[dominant_axis_idx]

    dominant_sign = "+" if delta[dominant_axis_idx] >= 0 else "-"
    dominant_axis_signed = f"{dominant_sign}{dominant_axis}"

    direction = unit_vector(delta)

    # 速度
    speed = part["tip_speed_mm_s"].to_numpy(dtype=float)
    angular_speed = part["angular_speed_deg_s"].to_numpy(dtype=float)

    # 姿勢軸変化
    x_axis_change = part["x_axis_change_deg"].to_numpy(dtype=float)
    y_axis_change = part["y_axis_change_deg"].to_numpy(dtype=float)
    z_axis_change = part["z_axis_change_deg"].to_numpy(dtype=float)

    # 回転行列
    R_start = make_R_from_row(start_row)
    R_end = make_R_from_row(end_row)
    total_rotation = rotation_angle_deg(R_start, R_end)

    # 単調性
    x_pos_ratio = positive_ratio(part["origin_x"])
    y_pos_ratio = positive_ratio(part["origin_y"])
    z_pos_ratio = positive_ratio(part["origin_z"])

    x_neg_ratio = negative_ratio(part["origin_x"])
    y_neg_ratio = negative_ratio(part["origin_y"])
    z_neg_ratio = negative_ratio(part["origin_z"])

    feature = {
        "segment_id": int(segment["segment_id"]),
        "primitive": primitive,

        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time": float(start_row["time"]),
        "end_time": float(end_row["time"]),
        "duration_s": float(end_row["time"] - start_row["time"]),

        # start/end position
        "start_x": float(p_start[0]),
        "start_y": float(p_start[1]),
        "start_z": float(p_start[2]),
        "end_x": float(p_end[0]),
        "end_y": float(p_end[1]),
        "end_z": float(p_end[2]),

        # displacement
        "delta_x": float(dx),
        "delta_y": float(dy),
        "delta_z": float(dz),
        "displacement_mm": displacement,
        "path_length_mm": path_len,
        "straightness": safe_float(straightness),

        # direction
        "dominant_axis": dominant_axis,
        "dominant_axis_role": TASK_AXIS_LABELS[dominant_axis],
        "dominant_axis_signed": dominant_axis_signed,
        "direction_unit_x": float(direction[0]),
        "direction_unit_y": float(direction[1]),
        "direction_unit_z": float(direction[2]),

        # min/max position
        "min_x": float(np.nanmin(part["origin_x"])),
        "max_x": float(np.nanmax(part["origin_x"])),
        "min_y": float(np.nanmin(part["origin_y"])),
        "max_y": float(np.nanmax(part["origin_y"])),
        "min_z": float(np.nanmin(part["origin_z"])),
        "max_z": float(np.nanmax(part["origin_z"])),

        # monotonicity
        "x_positive_ratio": safe_float(x_pos_ratio),
        "y_positive_ratio": safe_float(y_pos_ratio),
        "z_positive_ratio": safe_float(z_pos_ratio),
        "x_negative_ratio": safe_float(x_neg_ratio),
        "y_negative_ratio": safe_float(y_neg_ratio),
        "z_negative_ratio": safe_float(z_neg_ratio),

        # speed stats
        "mean_tip_speed_mm_s": float(np.nanmean(speed)),
        "max_tip_speed_mm_s": float(np.nanmax(speed)),
        "min_tip_speed_mm_s": float(np.nanmin(speed)),
        "std_tip_speed_mm_s": float(np.nanstd(speed)),

        # angular speed stats
        "mean_angular_speed_deg_s": float(np.nanmean(angular_speed)),
        "max_angular_speed_deg_s": float(np.nanmax(angular_speed)),
        "min_angular_speed_deg_s": float(np.nanmin(angular_speed)),
        "std_angular_speed_deg_s": float(np.nanstd(angular_speed)),

        # axis change
        "mean_x_axis_change_deg": float(np.nanmean(x_axis_change)),
        "mean_y_axis_change_deg": float(np.nanmean(y_axis_change)),
        "mean_z_axis_change_deg": float(np.nanmean(z_axis_change)),
        "max_x_axis_change_deg": float(np.nanmax(x_axis_change)),
        "max_y_axis_change_deg": float(np.nanmax(y_axis_change)),
        "max_z_axis_change_deg": float(np.nanmax(z_axis_change)),

        # total orientation change
        "total_rotation_deg": total_rotation,

        # rotation matrix
        "R_start": R_start.tolist(),
        "R_end": R_end.tolist(),
    }

    robot_hint = make_robot_translation_hint(primitive, feature)

    return feature, robot_hint


# =========================
# Main
# =========================
def main():
    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(
            f"{FEATURE_CSV} が見つかりません。先に analyze_scop_6dof_segments.py を実行してください。"
        )

    features_df = pd.read_csv(FEATURE_CSV)
    features_df = features_df.sort_values("time").reset_index(drop=True)

    segments = load_segments()

    csv_rows = []
    json_rows = []

    for segment in segments:
        feature, robot_hint = extract_segment_features(features_df, segment)

        csv_row = feature.copy()

        # CSVでは回転行列を文字列化
        csv_row["R_start"] = json.dumps(feature["R_start"], ensure_ascii=False)
        csv_row["R_end"] = json.dumps(feature["R_end"], ensure_ascii=False)

        csv_rows.append(csv_row)

        json_rows.append({
            "segment_id": feature["segment_id"],
            "primitive": feature["primitive"],
            "human_interval": {
                "start_time": feature["start_time"],
                "end_time": feature["end_time"],
                "duration_s": feature["duration_s"],
            },
            "motion_features": feature,
            "robot_translation_hint": robot_hint,
        })

    out_df = pd.DataFrame(csv_rows)
    out_df.to_csv(OUT_FEATURE_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_FEATURE_CSV}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, ensure_ascii=False)

    print(f"saved: {OUT_JSON}")

    # 見やすい要約を表示
    print("\n=== primitive summary ===")
    summary_cols = [
        "primitive",
        "start_time",
        "end_time",
        "duration_s",
        "delta_x",
        "delta_y",
        "delta_z",
        "dominant_axis_signed",
        "max_tip_speed_mm_s",
        "max_angular_speed_deg_s",
        "total_rotation_deg",
    ]

    print(out_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()