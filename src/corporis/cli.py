# Corporis
# Apache License, Version 2.0
"""
The ``corporis`` command.

    corporis character > me.json                    # a character card with the default settings
    corporis export me.glb --character me.json --animation walk --animation wave
    corporis export me.glb --morph-targets all      # every facial action and face shape
    corporis names poses                            # the poses and clips of the library
    corporis viewer build                           # the web viewer (python -m anny.viewer build)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from corporis.character import Character


def _load_character(path: str | None) -> Character:
    if path is None:
        return Character()
    return Character.from_dict(json.loads(pathlib.Path(path).read_text()))


def _export(args) -> int:
    from corporis.export.gltf import export_glb

    character = _load_character(args.character)
    model, morph_targets = None, None  # export_glb builds the model it needs
    if args.morph_targets == "none":
        morph_targets = []
    elif args.morph_targets == "all":
        model = character.build_model(face_shapes=True)
        morph_targets = list(model.facial_action_labels) + list(model.face_shape_labels)
    elif args.morph_targets not in (None, "facial"):
        morph_targets = [n.strip() for n in args.morph_targets.split(",") if n.strip()]
    summary = export_glb(
        args.output,
        character,
        model,
        animations=args.animation or (),
        morph_targets=morph_targets,
        max_influences=args.influences,
        ground=not args.no_ground,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _names(args) -> int:
    import anny

    if args.kind == "poses":
        import anny.poses

        names = anny.poses.names()
    elif args.kind == "local-changes":
        names = anny.Anny(local_changes="all").local_change_labels
    else:
        model = Character().build_model(face_shapes=args.kind == "face-shapes")
        attribute = {
            "phenotypes": "phenotype_labels",
            "facial-actions": "facial_action_labels",
            "face-shapes": "face_shape_labels",
        }[args.kind]
        names = getattr(model, attribute)
    print("\n".join(names))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="corporis", description="Characters of the Anny body model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("character", help="print a character card with the default settings")

    e = sub.add_parser(
        "export", help="export a character as a glTF 2.0 binary file (.glb)"
    )
    e.add_argument("output", help="the .glb file to write")
    e.add_argument(
        "--character",
        help="a character card (JSON); the default character when omitted",
    )
    e.add_argument(
        "--animation",
        action="append",
        help="a pose or clip of the library; repeat for more",
    )
    e.add_argument(
        "--morph-targets",
        help="'facial' (the default: the 52 facial actions), 'all' (with the face shapes), "
        "'none', or a comma-separated list of names",
    )
    e.add_argument(
        "--influences",
        type=int,
        choices=(4, 8),
        default=4,
        help="bones per vertex (default 4)",
    )
    e.add_argument(
        "--no-ground", action="store_true", help="keep the rest pose at Anny's origin"
    )

    n = sub.add_parser("names", help="list parameter, pose and clip names")
    n.add_argument(
        "kind",
        choices=(
            "poses",
            "phenotypes",
            "local-changes",
            "facial-actions",
            "face-shapes",
        ),
    )

    v = sub.add_parser(
        "viewer", help="build the web viewer (see python -m anny.viewer)"
    )
    v.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "character":
        print(json.dumps(Character().to_dict(), indent=2))
        return 0
    if args.command == "export":
        return _export(args)
    if args.command == "names":
        return _names(args)
    if args.command == "viewer":
        from anny.viewer.__main__ import main as viewer_main

        return viewer_main(args.rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())
