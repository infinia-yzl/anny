# OpenSculptBoy
# Apache License, Version 2.0
"""
The ``opensculptboy`` command.

    opensculptboy character > me.json                # a character card with the default settings
    opensculptboy export me.glb --character me.json --animation walk --animation wave
    opensculptboy export me.glb --morph-targets all  # every facial action and face shape
    opensculptboy names poses                        # the poses and clips of the library
    opensculptboy pose photo.jpg --card pose.json --glb pose.glb --svg pose.svg --png pose.png
    opensculptboy viewer build                       # the web viewer (python -m anny.viewer build)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from opensculptboy.character import Character


def _load_character(path: str | None) -> Character:
    if path is None:
        return Character()
    return Character.from_dict(json.loads(pathlib.Path(path).read_text()))


def _export(args) -> int:
    from opensculptboy.export.gltf import export_glb

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


def _pose(args) -> int:
    from opensculptboy.posing.picture import flat_view, pose_from_image, posed_mesh
    from opensculptboy.render.flat import outline_svg, shaded_png, silhouette_svg

    if not any((args.card, args.glb, args.svg, args.outline, args.png)):
        args.card = "-"
    character = _load_character(args.character)
    model = character.build_model()
    character = pose_from_image(
        args.image,
        character,
        model,
        refine=not args.no_refine,
        hands=not args.no_hands,
        face=not args.no_face,
        head=dict(turn=args.head_turn, up=args.head_up, tilt=args.head_tilt),
    )
    written = []
    if args.card:
        text = json.dumps(character.to_dict(), indent=2)
        if args.card == "-":
            print(text)
        else:
            pathlib.Path(args.card).write_text(text + "\n")
            written.append(args.card)
    if args.glb:
        from opensculptboy.export.gltf import export_glb

        export_glb(args.glb, character, model)
        written.append(args.glb)
    if args.svg or args.outline or args.png:
        vertices, faces = posed_mesh(character, model)
        view = flat_view(character, args.view)
        size = (args.size, args.size)
        desc = f"{character.name}, posed from {pathlib.Path(args.image).name}"
        if args.svg:
            text = silhouette_svg(vertices, faces, view, size=size, desc=desc)
            pathlib.Path(args.svg).write_text(text)
            written.append(args.svg)
        if args.outline:
            text = outline_svg(vertices, faces, view, size=size, desc=desc)
            pathlib.Path(args.outline).write_text(text)
            written.append(args.outline)
        if args.png:
            shaded_png(vertices, faces, view, size=size).save(args.png)
            written.append(args.png)
    for path in written:
        print(f"wrote {path}", file=sys.stderr)
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
        prog="opensculptboy", description="Characters of the Anny body model."
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

    p = sub.add_parser(
        "pose",
        help="pose a character as the figure in a picture, and draw it flat "
        "(needs the pose extra)",
    )
    p.add_argument("image", help="the picture (JPEG, PNG, WebP)")
    p.add_argument(
        "--character",
        help="a character card (JSON) to pose; the default character when omitted",
    )
    p.add_argument(
        "--card", help="write the posed character card (JSON); '-' prints it"
    )
    p.add_argument("--glb", help="write a glTF 2.0 binary file with the pose")
    p.add_argument("--svg", help="write the silhouette (SVG)")
    p.add_argument("--outline", help="write the line drawing (SVG)")
    p.add_argument("--png", help="write the flat-shaded picture (PNG)")
    p.add_argument(
        "--view",
        choices=("image", "front", "three-quarter"),
        default="image",
        help="the camera of the flat drawings (default: the picture's)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=512,
        help="the drawings' size in pixels (default 512)",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the refinement against the picture",
    )
    p.add_argument("--no-hands", action="store_true", help="leave the fingers at rest")
    p.add_argument("--no-face", action="store_true", help="leave the face neutral")
    for name, what in (
        ("turn", "toward the figure's right (negative: its left)"),
        ("up", "to raise the face (negative: lower it)"),
        ("tilt", "toward the figure's right shoulder (negative: the left)"),
    ):
        p.add_argument(
            f"--head-{name}",
            type=float,
            default=0.0,
            metavar="DEG",
            help=f"turn the head further {what}, when the picture leaves it unclear",
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
    if args.command == "pose":
        return _pose(args)
    if args.command == "viewer":
        from anny.viewer.__main__ import main as viewer_main

        return viewer_main(args.rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())
