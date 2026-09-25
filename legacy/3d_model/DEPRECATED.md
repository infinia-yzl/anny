# Deprecated: the 3D Model experiment

This folder is deprecated. It holds the standalone 3D Model experiment, which the commit `b10538d` added at the root of the repository as `3D Model/`. The experiment had its own MakeHuman loader, its own morph code, its own skeleton builder and a three.js viewer, and every stage baked one fixed character.

anny now carries the parts of this experiment that anny did not have. Each part follows anny's sliders, and anny is the only source of the shape, the skeleton and the skin weights.

| Part of the experiment | New home |
| --- | --- |
| Catmull-Clark subdivision with a finer head (`build/subdiv.py`, `build/build_body.py`) | `src/anny/utils/subdivision.py` |
| Poses, clips, shoulder girdle rule, IK and grounding (`build/posing.py`, `build/poselib.py`, `build/poses.py`, `build/anims.py`, `build/import_poses.py`) | `src/anny/poses/` |
| Soft-tissue correctives and their simulation (`build/correctives.py`, `build/sim.py`) | `src/anny/correctives/` |
| Hair, brows and lashes (`build/hair2.py`, `build/brows_lashes.py`) | `src/anny/hair/` |
| Bakes, skin regions, eyes and detail layers (`build/bake.py`, `build/regions.py`, `build/eyes.py`, `build/relief.py`) | `src/anny/viewer/` |
| The viewer page (`web/app.js`, `web/shell.html`) | `viewer/`, with the shaders in `viewer/src/shading.ts` |

The command below builds the new viewer page into `viewer/dist/anny_viewer.html`:

```bash
uv sync --extra viewer
uv run python -m anny.viewer build
```

The wearables (trunks, shorts and T-shirt) and the fixed character of `build/character.py` stay in this folder for now. The build scripts sit in `baseline_body/source.zip` and `baseline_head/source.zip`, and the READMEs of `baseline_body` and `baseline_head` describe the experiment as it was.
