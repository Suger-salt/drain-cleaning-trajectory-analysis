# -*- coding: utf-8 -*-
"""
generate_robot_motion_plan.py

目的:
    primitive_parameters.json から、
    ロボット用の動作命令 robot_motion_plan.json を生成する。

入力:
    primitive_parameters.json

出力:
    robot_motion_plan.json
    robot_motion_plan_summary.csv

実行:
    python generate_robot_motion_plan.py
"""

import json
import os
import pandas as pd


# =========================
# 入出力
# =========================
INPUT_JSON = "primitive_parameters.json"

OUT_JSON = "robot_motion_plan.json"
OUT_CSV = "robot_motion_plan_summary.csv"


# =========================
# 安全側のパラメータ
# 単位: mm, mm/s, deg, deg/s, N
# =========================
SAFETY_CONFIG = {
    # miniature workspace
    "workspace_x_mm": 200.0,
    "workspace_y_mm": 200.0,
    "workspace_z_mm": 120.0,
    "workspace_margin_mm": 20.0,

    # scrape
    "scrape_max_distance_mm": 80.0,
    "scrape_speed_mm_s": 20.0,
    "scrape_force_limit_N": 3.0,

    # lift
    "lift_max_height_mm": 50.0,
    "lift_speed_mm_s": 20.0,

    # dump
    "dump_lateral_offset_mm": 60.0,
    "dump_height_offset_mm": 30.0,
    "dump_tilt_angle_deg": 65.0,
    "dump_tilt_speed_deg_s": 10.0,

    # retract
    "retract_speed_mm_s": 20.0,
}


# =========================
# Utility
# =========================
def load_primitive_parameters(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} が見つかりません。先に extract_primitive_parameters.py を実行してください。"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sign_label(value, axis):
    if value >= 0:
        return f"+{axis}"
    return f"-{axis}"


def clamp_abs(value, max_abs):
    """
    符号は保ったまま、絶対値を max_abs 以下に制限する。
    """
    if value >= 0:
        return min(value, max_abs)
    return -min(abs(value), max_abs)


def get_features(item):
    return item["motion_features"]


def get_start_position(f):
    return [f["start_x"], f["start_y"], f["start_z"]]


def get_end_position(f):
    return [f["end_x"], f["end_y"], f["end_z"]]


# =========================
# primitiveごとの変換
# =========================
def translate_scrape_or_scoop(item):
    f = get_features(item)

    dx = f["delta_x"]
    dy = f["delta_y"]
    dz = f["delta_z"]

    # 人間は +x 方向に約305mm進んでいる。
    # ただしロボットでは安全のため距離を上限で制限する。
    distance_mm = min(abs(dx), SAFETY_CONFIG["scrape_max_distance_mm"])
    direction = sign_label(dx, "x")

    return {
        "step_id": f["segment_id"],
        "source_primitive": f["primitive"],
        "robot_primitive": "guarded_scrape_or_scoop",

        "intent": "側溝長手方向に沿って泥や堆積物をかく・すくう",

        "human_reference": {
            "start_time": f["start_time"],
            "end_time": f["end_time"],
            "delta_mm": [dx, dy, dz],
            "max_tip_speed_mm_s": f["max_tip_speed_mm_s"],
            "max_angular_speed_deg_s": f["max_angular_speed_deg_s"],
            "total_rotation_deg": f["total_rotation_deg"],
        },

        "robot_parameters": {
            "task_frame": "mocap_task_frame",
            "motion_type": "linear_contact_motion",
            "direction": direction,
            "distance_mm": distance_mm,
            "speed_mm_s": SAFETY_CONFIG["scrape_speed_mm_s"],
            "force_limit_N": SAFETY_CONFIG["scrape_force_limit_N"],
            "orientation_policy": "keep_average_tool_orientation",
            "contact_policy": "guarded",
        },

        "safety_policy": {
            "do_not_push_bottom_strongly": True,
            "stop_if_force_exceeds_limit": True,
            "use_low_speed_near_unknown_region": True,
        },

        "translation_note": (
            "人間のscrape_or_scoopでは+x方向への進行が支配的であった。"
            "ロボットでは同じ方向性のみを使い、距離・速度・力を安全側に制限する。"
        )
    }


def translate_lift(item):
    f = get_features(item)

    dx = f["delta_x"]
    dy = f["delta_y"]
    dz = f["delta_z"]

    # 人間のliftは672mm上がっているが、実機では大きすぎる可能性がある。
    # まずは安全な高さに制限する。
    lift_height_mm = min(max(dz, 0.0), SAFETY_CONFIG["lift_max_height_mm"])

    return {
        "step_id": f["segment_id"],
        "source_primitive": f["primitive"],
        "robot_primitive": "lift",

        "intent": "すくった泥や堆積物を保持したまま持ち上げる",

        "human_reference": {
            "start_time": f["start_time"],
            "end_time": f["end_time"],
            "delta_mm": [dx, dy, dz],
            "max_tip_speed_mm_s": f["max_tip_speed_mm_s"],
            "max_angular_speed_deg_s": f["max_angular_speed_deg_s"],
            "total_rotation_deg": f["total_rotation_deg"],
        },

        "robot_parameters": {
            "task_frame": "mocap_task_frame",
            "motion_type": "vertical_lift",
            "direction": "+z",
            "height_mm": lift_height_mm,
            "speed_mm_s": SAFETY_CONFIG["lift_speed_mm_s"],
            "orientation_policy": "keep_tool_orientation",

            # 人間動作にはx,y方向の移動も含まれるが、
            # ロボットでは安全のため、まず上方向への持ち上げとして扱う
            "human_dominant_direction": f["dominant_axis_signed"],
            "preserve_lateral_motion": False,
        },

        "safety_policy": {
            "avoid_sudden_acceleration": True,
            "keep_payload_stable": True,
        },

        "translation_note": (
            "人間のliftでは+z方向の変化が支配的であった。"
            "ロボットでは持ち上げ意図を使い、安全な高さまで上昇する動作に変換する。"
        )
    }


def translate_dump_main(item):
    f = get_features(item)

    dx = f["delta_x"]
    dy = f["delta_y"]
    dz = f["delta_z"]

    return {
        "step_id": f["segment_id"],
        "source_primitive": f["primitive"],
        "robot_primitive": "safe_dump",

        "intent": "すくった泥や堆積物を排出する",

        "human_reference": {
            "start_time": f["start_time"],
            "end_time": f["end_time"],
            "delta_mm": [dx, dy, dz],
            "max_tip_speed_mm_s": f["max_tip_speed_mm_s"],
            "max_angular_speed_deg_s": f["max_angular_speed_deg_s"],
            "total_rotation_deg": f["total_rotation_deg"],
            "human_motion_character": "high_speed_throwing",
        },

        "robot_parameters": {
            "task_frame": "mocap_task_frame",
            "motion_type": "move_then_tilt",
            "dump_position_policy": "predefined_safe_dump_area",
            "lateral_offset_mm": SAFETY_CONFIG["dump_lateral_offset_mm"],
            "height_offset_mm": SAFETY_CONFIG["dump_height_offset_mm"],
            "tilt_angle_deg": SAFETY_CONFIG["dump_tilt_angle_deg"],
            "tilt_speed_deg_s": SAFETY_CONFIG["dump_tilt_speed_deg_s"],
            "do_not_throw": True,
        },

        "safety_policy": {
            "do_not_imitate_high_speed_throwing": True,
            "limit_angular_velocity": True,
            "avoid_scattering_sediment": True,
            "avoid_collision_with_wall": True,
        },

        "translation_note": (
            "人間のdump_mainでは先端速度・角速度・姿勢変化が最大となり、"
            "高速な投げ捨て動作であることが確認された。"
            "ロボットではこの軌道を直接模倣せず、排出意図のみを利用して"
            "安全な低速傾け動作へ置換する。"
        )
    }


def translate_retract(item):
    f = get_features(item)

    dx = f["delta_x"]
    dy = f["delta_y"]
    dz = f["delta_z"]

    return {
        "step_id": f["segment_id"],
        "source_primitive": f["primitive"],
        "robot_primitive": "safe_retract",

        "intent": "排出後にスコップを次の作業開始姿勢へ戻す",

        "human_reference": {
            "start_time": f["start_time"],
            "end_time": f["end_time"],
            "delta_mm": [dx, dy, dz],
            "max_tip_speed_mm_s": f["max_tip_speed_mm_s"],
            "max_angular_speed_deg_s": f["max_angular_speed_deg_s"],
            "total_rotation_deg": f["total_rotation_deg"],
        },

        "robot_parameters": {
            "task_frame": "mocap_task_frame",
            "motion_type": "return_to_ready_pose",
            "target_policy": "next_scrape_start_pose_or_safe_pose",
            "speed_mm_s": SAFETY_CONFIG["retract_speed_mm_s"],
            "orientation_policy": "restore_scrape_orientation",
        },

        "safety_policy": {
            "collision_avoidance": True,
            "avoid_bottom_and_wall_contact": True,
            "move_through_safe_height_if_needed": True,
        },

        "translation_note": (
            "人間のretractではdump後に位置と姿勢が大きく復帰していた。"
            "ロボットではこの戻り軌道を直接再現するのではなく、"
            "安全姿勢または次のscrape開始姿勢へ戻る動作として生成する。"
        )
    }


def translate_unknown(item):
    f = get_features(item)

    return {
        "step_id": f["segment_id"],
        "source_primitive": f["primitive"],
        "robot_primitive": "unknown",
        "intent": "未定義",
        "human_reference": {
            "start_time": f["start_time"],
            "end_time": f["end_time"],
        },
        "robot_parameters": {},
        "safety_policy": {},
        "translation_note": "このprimitiveに対応する変換ルールはまだ定義されていない。"
    }


def translate_item(item):
    primitive = item["primitive"]

    if primitive == "scrape_or_scoop":
        return translate_scrape_or_scoop(item)

    if primitive == "lift":
        return translate_lift(item)

    if primitive == "dump_main":
        return translate_dump_main(item)

    if primitive == "retract":
        return translate_retract(item)

    return translate_unknown(item)


# =========================
# summary作成
# =========================
def make_summary_rows(robot_plan):
    rows = []

    for step in robot_plan:
        params = step["robot_parameters"]
        ref = step["human_reference"]

        row = {
            "step_id": step["step_id"],
            "source_primitive": step["source_primitive"],
            "robot_primitive": step["robot_primitive"],
            "intent": step["intent"],

            "start_time": ref.get("start_time"),
            "end_time": ref.get("end_time"),

            "human_delta": ref.get("delta_mm"),
            "human_max_tip_speed_mm_s": ref.get("max_tip_speed_mm_s"),
            "human_max_angular_speed_deg_s": ref.get("max_angular_speed_deg_s"),
            "human_total_rotation_deg": ref.get("total_rotation_deg"),

            "motion_type": params.get("motion_type"),
            "direction": params.get("direction"),
            "distance_mm": params.get("distance_mm"),
            "height_mm": params.get("height_mm"),
            "speed_mm_s": params.get("speed_mm_s"),
            "force_limit_N": params.get("force_limit_N"),
            "tilt_angle_deg": params.get("tilt_angle_deg"),
            "tilt_speed_deg_s": params.get("tilt_speed_deg_s"),
            "do_not_throw": params.get("do_not_throw"),
        }

        rows.append(row)

    return rows


# =========================
# Main
# =========================
def main():
    primitive_items = load_primitive_parameters(INPUT_JSON)

    robot_plan = []
    for item in primitive_items:
        robot_step = translate_item(item)
        robot_plan.append(robot_step)

    output = {
        "plan_name": "drain_cleaning_robot_motion_plan",
        "description": (
            "Human shovel motion primitives were translated into robot-executable "
            "motion primitives. Human trajectories are not directly replayed; "
            "they are used to infer task intent and safe robot parameters."
        ),
        "coordinate_assumption": {
            "task_frame": "mocap_task_frame",
            "x": "longitudinal direction of gutter",
            "y": "width/depth direction",
            "z": "height direction",
            "note": (
                "This is a provisional task-frame interpretation based on motion observation. "
                "Before real robot execution, conversion to robot base frame is required."
            )
        },
        "safety_config": SAFETY_CONFIG,
        "steps": robot_plan,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"saved: {OUT_JSON}")

    summary_rows = make_summary_rows(robot_plan)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"saved: {OUT_CSV}")

    print("\n=== robot motion plan summary ===")
    show_cols = [
        "source_primitive",
        "robot_primitive",
        "motion_type",
        "direction",
        "distance_mm",
        "height_mm",
        "speed_mm_s",
        "force_limit_N",
        "tilt_angle_deg",
        "do_not_throw",
    ]

    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()