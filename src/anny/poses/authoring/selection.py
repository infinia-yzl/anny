# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""Poses chosen from the MakeHuman community pose packs (CC0), imported with import_poses.py.
Ported from the legacy 3D Model build (build/poses_mh.py).

Each entry: key, file name (without .bvh), label in the viewer, group in the Pose panel. The packs are the MakeHuman
system poses and the community packs poses01 to poses05 (import_poses.pack_dir()).

Left out: poses built around a prop that the viewer does not have (a bicycle, a guitar, a glass, a counter, a bow, a
gun, a hammer), poses that rest a hand on another person, poses in the air or in water, fashion-model poses, and
poses that repeat a pose of the library.
"""

from . import import_poses as IP

SELECTION = [
    ("mh_step", "standing01", "Mid-step", "Standing"),
    ("mh_hand_hip", "standing04", "Hand on hip", "Standing"),
    ("mh_attention", "callharvey3d_standingatattention", "At attention", "Standing"),
    ("mh_at_ease", "callharvey3d_standingatease", "At ease", "Standing"),
    ("mh_thinking", "callharvey3d_curious", "Thinking", "Gestures"),
    ("mh_salute", "callharvey3d_salute", "Salute", "Gestures"),
    ("mh_peace", "gpedroso_peace_and_love", "Peace sign", "Gestures"),
    ("mh_pointing", "standing05", "Pointing", "Gestures"),
    ("mh_goodbye", "punkduck_train_departure", "Waving goodbye", "Gestures"),
    ("mh_cheer", "spreadcore_arms_up_pose_001", "Cheering", "Gestures"),
    ("mh_hero", "elvs_super_him_standing_pose_1", "Hero stance", "Gestures"),
    ("mh_cast", "culturalibre_wizard_with_sceptre_1", "Casting a spell", "Action"),
    ("mh_cast_wide", "culturalibre_wizard_with_sceptre_2", "Arms spread", "Action"),
    ("mh_crouch_low", "gpedroso_spider_man", "Low crouch", "Action"),
    ("mh_fight", "fight01", "Fighting stance", "Martial arts"),
    ("mh_guard", "fight03", "Low guard", "Martial arts"),
    ("mh_kick", "fight04", "High kick", "Martial arts"),
    ("mh_block", "culturalibre_cl_fight_01", "Block and strike", "Martial arts"),
    ("mh_boxing", "gpedroso_boxing", "Boxing guard", "Martial arts"),
    ("mh_palm", "gpedroso_hadouken", "Palm strike", "Martial arts"),
    ("mh_focus", "gpedroso_ninja_focus", "Focus", "Martial arts"),
    ("mh_sprint", "run01", "Sprint", "Sports and fitness"),
    ("mh_serve", "joachip_tennis_serve", "Tennis serve", "Sports and fitness"),
    (
        "mh_backhand",
        "punkduck_tennis_two-handed_backhand",
        "Backhand",
        "Sports and fitness",
    ),
    ("mh_star", "elvs_yoga_star_pose_1", "Star pose", "Sports and fitness"),
    ("mh_triangle", "elvs_yoga_triangle_pose_1", "Triangle pose", "Sports and fitness"),
    (
        "mh_toe_touch",
        "callharvey3d_sittingfloorstretch2",
        "Toe-touch stretch",
        "Sports and fitness",
    ),
    ("mh_cobra", "elvs_yoga_cobra_pose_1", "Cobra stretch", "Sports and fitness"),
    ("mh_pushup", "elvs_pushups_1", "Push-up", "Sports and fitness"),
    ("mh_splits", "elvs_gymnastic_pose_1", "Splits", "Sports and fitness"),
    ("mh_handstand", "gym01", "Handstand", "Sports and fitness"),
    ("mh_sit_think", "anrico_sitting04", "Chin on hand", "Sitting"),
    ("mh_sit_talk", "anrico_sitting10", "Talking with hands", "Sitting"),
    ("mh_sit_explain", "anrico_sitting03", "Explaining", "Sitting"),
    ("mh_sit_ankle", "callharvey3d_sittinglegscrossed", "Ankle on knee", "Sitting"),
    ("mh_floor_back", "sit01", "Leaning back", "On the floor"),
    ("mh_cross_legged", "callharvey3d_lotus", "Cross-legged", "On the floor"),
    ("mh_hug_knees", "callharvey3d_sittingfloor2", "Hugging knees", "On the floor"),
    ("mh_side_sit", "wolgade_sit_on_ground_01", "Side sit", "On the floor"),
    ("mh_head_in_hand", "elvs_what_have_i_done", "Head in hand", "On the floor"),
]

_FILES = None
_CACHE = {}


def path(stem):
    """(zip path, member name) of a pose file"""
    global _FILES
    if _FILES is None:
        _FILES = IP.pose_files()
    return _FILES[stem]


def build(key):
    if key not in _CACHE:
        stem = next(e[1] for e in SELECTION if e[0] == key)
        _CACHE[key] = IP.load_pose(path(stem))
    return _CACHE[key]


def credits():
    """author, pack and license of each chosen pose (from the pack descriptions, else the .meta files)"""
    import os

    packs = IP.pack_descriptions()
    out = []
    for key, stem, label, group in SELECTION:
        f = path(stem)
        m = IP.meta(f)
        info = packs.get(stem, {})
        pack = os.path.basename(f[0])[: -len(".zip")].replace("_cc0", "")
        author = info.get("author") or m.get("author") or stem.split("_")[0]
        out.append(
            dict(
                key=key,
                label=label,
                file=stem,
                pack=pack,
                author=author,
                license=info.get("license") or m.get("license") or "CC0",
                source=info.get("source", ""),
            )
        )
    return out
