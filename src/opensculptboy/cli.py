# OpenSculptBoy
# Apache License, Version 2.0
"""
The ``opensculptboy`` command.

    opensculptboy character > me.json                # a character card with the default settings
    opensculptboy export me.glb --character me.json --animation walk --animation wave
    opensculptboy export me.glb --morph-targets all  # every facial action and face shape
    opensculptboy export me.vrm --character me.json --author "Me"   # a VRM 1.0 avatar
    opensculptboy export me0.vrm --author "Me" --vrm-version 0 --thumbnail auto   # VRM 0.x
    opensculptboy names poses                        # the poses and clips of the library
    opensculptboy pose photo.jpg --card pose.json --glb pose.glb --svg pose.svg --png pose.png
    opensculptboy viewer build                       # the web viewer (python -m anny.viewer build)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

from opensculptboy.character import Character

# The options of ``export`` that only one of the two formats takes.
_GLB_ONLY = ("animation", "morph_targets", "influences", "no_ground")
_VRM_ONLY = (
    "author",
    "name",
    "meta",
    "vrm_version",
    "thumbnail",
    "twist",
    "bind",
    "keep_leg_spread",
    "budget",
    "anny_vertex",
)


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _load_character(path: str | None, parser=None) -> Character:
    """
    The character card at ``path``, or the default character; with ``parser``, a card that
    cannot be read is a usage error.
    """
    if path is None:
        return Character()
    try:
        return Character.from_dict(json.loads(pathlib.Path(path).read_text()))
    except (OSError, ValueError, TypeError) as error:
        if parser is None:
            raise
        parser.error(f"--character {path}: {_one_line(error)}")


def _one_line(error: BaseException) -> str:
    """The message of an error on one line."""
    return " ".join(str(error).split()) or type(error).__name__


def _export(args, parser) -> int:
    is_vrm = pathlib.Path(args.output).suffix.lower() == ".vrm"
    wrong = _GLB_ONLY if is_vrm else _VRM_ONLY
    given = [_flag(d) for d in wrong if getattr(args, d) not in (None, False)]
    if given:
        kind, other = (".vrm", ".glb") if is_vrm else (".glb", ".vrm")
        parser.error(
            f"{', '.join(given)}: only for {other} files, not for the {kind} file "
            f"{args.output}."
        )
    if is_vrm:
        return _export_vrm(args, parser)
    from opensculptboy.export.gltf import export_glb

    character = _load_character(args.character, parser)
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
        max_influences=args.influences or 4,
        ground=not args.no_ground,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _export_vrm(args, parser) -> int:
    from opensculptboy.export.budget import BudgetError
    from opensculptboy.export.vrm import DEFAULT_BIND, VrmMeta, export_vrm

    meta = None
    if args.meta:
        try:
            meta = VrmMeta.from_dict(json.loads(pathlib.Path(args.meta).read_text()))
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--meta {args.meta}: {_one_line(error)}")
    if not args.author and not (meta and meta.authors):
        parser.error(
            "a VRM file needs an author: pass --author NAME, or --meta FILE with "
            '"authors": ["NAME"].'
        )
    character = _load_character(args.character, parser)
    # A file over a strict budget fails (exit 1); a character, an option or a thumbnail
    # that a VRM file cannot carry is a usage error (exit 2). Each prints one line, after
    # the warnings of the export.
    failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            summary = export_vrm(
                args.output,
                character,
                version={"1": "1.0", "0": "0.x"}[args.vrm_version or "1"],
                meta=meta,
                author=args.author,
                name=args.name,
                thumbnail=args.thumbnail,
                bare=args.bare,
                twist=args.twist,
                bind=args.bind or DEFAULT_BIND,
                keep_leg_spread=args.keep_leg_spread,
                budget=args.budget or "warn",
                keep_anny_vertex=args.anny_vertex,
            )
        except BudgetError as error:
            failure = (1, _one_line(error))
        except (OSError, ValueError, TypeError) as error:
            failure = (2, _one_line(error))
        finally:
            for warning in caught:
                print(f"warning: {warning.message}", file=sys.stderr)
    if failure is not None:
        code, message = failure
        if code == 2:
            parser.error(message)
        print(f"opensculptboy export: {message}", file=sys.stderr)
        return code
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
        "export",
        help="export a character as a glTF 2.0 binary file (.glb) or a VRM avatar (.vrm)",
        description="Export a character. The suffix of the output picks the format: .vrm "
        "writes a VRM avatar for VTuber apps (VRM 1.0, or VRM 0.x with --vrm-version 0), "
        "any other suffix a glTF 2.0 binary file.",
    )
    e.add_argument("output", help="the .glb or .vrm file to write")
    e.add_argument(
        "--character",
        help="a character card (JSON); the default character when omitted",
    )
    e.add_argument(
        "--bare",
        action="store_true",
        help="the body alone: no outfit, hair cards, teeth or lashes",
    )
    g = e.add_argument_group("glTF (.glb) options")
    g.add_argument(
        "--animation",
        action="append",
        help="a pose or clip of the library; repeat for more",
    )
    g.add_argument(
        "--morph-targets",
        help="'facial' (the default: the 52 facial actions), 'all' (with the face shapes), "
        "'none', or a comma-separated list of names",
    )
    g.add_argument(
        "--influences",
        type=int,
        choices=(4, 8),
        help="bones per vertex (default 4)",
    )
    g.add_argument(
        "--no-ground", action="store_true", help="keep the rest pose at Anny's origin"
    )
    r = e.add_argument_group("VRM (.vrm) options")
    r.add_argument(
        "--author",
        help="the avatar's author (required, unless --meta names the authors)",
    )
    r.add_argument("--name", help="the avatar's name (default: the character's name)")
    r.add_argument(
        "--meta",
        metavar="FILE",
        help="licence metadata as JSON: the fields of opensculptboy.export.vrm.VrmMeta "
        '(for example {"authors": ["Me"], "commercial_usage": "personalProfit"})',
    )
    r.add_argument(
        "--vrm-version",
        choices=("1", "0"),
        help="1 for VRM 1.0 (the default: Warudo, VMagicMirror, three-vrm), 0 for VRM 0.x "
        "(VSeeFace, 3tene)",
    )
    r.add_argument(
        "--thumbnail",
        metavar="PATH|auto",
        help="the thumbnail: a square PNG or JPEG file, or 'auto' for a rendered portrait",
    )
    r.add_argument(
        "--twist",
        choices=("constraint", "merge"),
        help="the forearm twist bones: roll constraints from the wrists (VRM 1.0 only, and "
        "its default) or their weights merged into the forearm bones (the VRM 0.x default); "
        "the upper arm, thigh and shin twist weights always merge",
    )
    r.add_argument(
        "--bind",
        choices=("forward", "inverse"),
        help="how the mesh moves into the T-pose (default: inverse, closest to the "
        "character's own skinning with the arms down)",
    )
    r.add_argument(
        "--keep-leg-spread",
        action="store_true",
        help="keep the rig's leg spread in the T-pose (default: legs vertical)",
    )
    r.add_argument(
        "--budget",
        choices=("strict", "warn", "off"),
        help="a file over the budget of its VRM version fails, warns (the default) or "
        "passes",
    )
    r.add_argument(
        "--anny-vertex",
        action="store_true",
        help="keep the _ANNY_VERTEX attribute (the Anny vertex of each file vertex)",
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
        return _export(args, e)
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
