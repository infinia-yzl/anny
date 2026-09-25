# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**Anny** is a differentiable human body mesh model in PyTorch that covers all ages (infants to elders) with a common topology and parameter space. Based on MakeHuman assets, it provides full-body, hand, and face models.

## Commands

### Setup
```bash
uv sync --extra examples  # full install with demo dependencies
```

### Testing
```bash
uv run python -m unittest discover     # run all tests
uv run python -m unittest test.test_various  # run a single test file
```

### Documentation
```bash
bash build_doc.bash  # build HTML docs from the jupytext py:percent tutorials in tutorials/*.py
```

### Face calibration
```bash
uv sync --extra viewer --extra faces
uv run python -m anny.faces.authoring.sources       # download the data sources into ANNY_CACHE_DIR/faces
uv run python -m anny.faces.authoring.landmarks     # data/keypoints/craniofacial.json
uv run python -m anny.faces.authoring.fit_3d        # fit anny's face shapes to ICT-FaceKit identities
uv run python -m anny.faces.authoring.detail        # data/faces/detail_shapes.safetensors, from the fit residuals
uv run python -m anny.faces.authoring.mediapipe_map # data/keypoints/mediapipe.json
uv run python -m anny.faces.authoring.calibrate     # data/shape_calibration/face_prior.safetensors
uv run python -m anny.faces.authoring.benchmark     # benchmark.html in ANNY_CACHE_DIR/faces
```

### Web viewer
```bash
uv sync --extra viewer                # scipy, tetgen, embreex for the build
uv run python -m anny.viewer build    # data (cached under ANNY_CACHE_DIR/viewer) and the page viewer/dist/anny_viewer.html
cd viewer && npx tsc --noEmit         # type-check the page
```

## Architecture

### Entry Points

`src/anny/models/__init__.py` exports the public API:
- `Anny(...)` — the full-body model. `Anny` is a class (cf. `anny.SMPLX`); calling
  `anny.Anny(...)` builds a model and `isinstance(model, Anny)` holds for any Anny model.
  Accepts `rig`, `topology`, `pose_parameterization`, `phenotypes`, and skinning options.
- `create_fullbody_model(...)` — deprecated legacy full-body factory. It preserves the old
  default rig preset (`rig="default"`) and old full-body defaults; prefer `Anny(...)`.
- `create_hand_model()` / `create_head_model()` — isolated part models

### Core Class Hierarchy

- **`RiggedModelWithLinearBlendShapes`** (`models/rigged_model.py`) — base class; holds template vertices/faces/blend shapes, implements forward kinematics and LBS. The `model_type` parameter (`"tail"` or `"procrustes"`) selects bone orientation strategy internally.
- **`Anny`** (`models/phenotype.py`) — inherits directly from `RiggedModelWithLinearBlendShapes`; adds the 9 phenotype dimensions (gender, age, muscle, weight, height, proportions, race, cupsize, firmness) and computes blend shape coefficients from these semantic scalars. 
- **`SMPL`** / **`SMPLX`** (`models/smpl.py`) — first-class model types that also inherit directly from `RiggedModelWithLinearBlendShapes`; wrap the `smplx` library and follow the same initialization pattern as `Anny`, but accept `betas` + pose parameters instead of phenotype dimensions. Require the optional `smplx` package (`uv sync --extra smpl`).

### Rigs & Topologies

**Rigs** (`anny`, `makehuman`, `cmu_mb`, `game_engine`, `mixamo`, `soma`): bone hierarchies defined as JSON in `src/anny/data/mpfb2/rigs/`. `anny` is the pruned procrustes Anny default, equivalent to the MakeHuman source rig with `notongue`, `noexpression`, and `pruned` modifiers. `default` is a legacy preset accepted only by `create_fullbody_model(...)` and preserves the old full MakeHuman rig defaults. `makehuman` is the full MakeHuman rig with tail/blender orientation and `root_identity_orientation=True`. Rig orientation is part of rig resolution; public constructors do not accept a separate `bone_orientation` argument.

**Topologies** (`default`/`makehuman` ≈16K verts, `smplx` 6890 verts, `soma`): alternative meshes are produced by retopology matrices in `src/anny/data/topology/`. SMPL-X is non-commercial only.

### Key Subsystems

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| Forward kinematics | `utils/kinematics.py` | Tree traversal with parallel propagation fronts |
| Skinning | `skinning/skinning.py` | LBS and dual-quaternion skinning |
| GPU skinning | `skinning/warp_skinning.py` | `warp-lang` accelerated variant (optional) |
| Collision | `utils/collision.py` | Self-intersection detection; warp-accelerated when available |
| Model data | `models/model_data.py` | `ModelData` / `ModelMetadata` dataclasses; bundle template mesh, blend shapes, and rig data; safetensors serialization for caching |
| Model transforms | `models/model_transforms.py` | `ModelData` → `ModelData` operations: retopology (from a mesh, or from linear combinations of template vertices), bone orientation conversion, mesh/skinning cleanups |
| Parameter regression | `anny_inverter.py` | `AnnyInverter`: iterative pose+shape fitting to a target mesh |
| Anthropometry | `anthropometry.py` | Computes body measurements (height, volume, mass) from mesh |
| Subdivision | `utils/subdivision.py` | Catmull-Clark as sparse linear operators; `MixedSubdivision` adds one level on a region (the head) |
| Pose library | `poses/` | 50 poses and 7 clips as `local-ref` parameters (`data/poses/`), grounding, stool; `poses/authoring/` builds the library on the authoring rig |
| Correctives | `correctives/` | `SoftTissueCorrectives` (hinge and cone drivers, shapes scaled with the local size; `data/correctives/`); `correctives/authoring/` holds the simulation, the fit and `evaluate` |
| Hair | `hair/` | `StrandBinding` ties strands to the skin at roots and tips; `hair/authoring/` grows the groom, brows and lashes |
| Viewer data | `viewer/` | `python -m anny.viewer build` writes the page data to `viewer/build/` and runs the node build of `viewer/` |
| Face shapes | `models/face_shapes.py`, `faces/` | 103 named, symmetric face-shape parameters and 10 detail shapes from ICT-FaceKit (`Anny(face_shapes=...)`, `face_shape_kwargs`), scaled per group with the size of the head; craniofacial landmarks and the measurements of 3D Facial Norms and ANSUR II (`faces/measurements.py`); a face-shape distribution calibrated against measured faces (`faces/distribution.py`); `faces/authoring/` fetches the sources, fits the ICT-FaceKit identity space, calibrates the distribution and benchmarks against FairFace photos |

### Phenotype System

Phenotypes are blended linearly between discrete anchor states defined in `src/anny/data/mpfb2/targets/`. Default mode omits race, cupsize, and firmness; pass `phenotypes="all"` to enable them. Blend shape data is computed at model creation and cached in `~/.cache/anny/`. Set the `ANNY_CACHE_DIR` environment variable to use a different location.

### Face Shapes

`Anny(face_shapes="all")` (or a list of names) adds the face-shape block: each parameter of
`data/faces/face_shapes.json` sums the left and right MakeHuman targets of the head, forehead, brows,
eyes, nose, cheeks, mouth, chin and ears into rows `face_shape:{name}.pos` and `.neg`; +1 applies the
positive targets, -1 the negative ones, and the head archetypes and `chin-triangle` run from 0 to 1.
The `detail` parameters (`source: "ict"` in the spec) come from `data/faces/detail_shapes.safetensors`:
the symmetric principal components of the residuals of the ICT-FaceKit fits, written by
`python -m anny.faces.authoring.detail`. The model cache key carries a digest of both files
(`face_shape_data_digest`), and the calibration widens the slider ranges to hold the calibrated
distribution. The rows of a group scale with the size of that part of the head (`Anny.face_shape_scales`),
measured on the craniofacial landmarks of `data/keypoints/craniofacial.json`, which `ModelData`
stores for the template and every blend shape. The `anny` and `soma` rig caches carry the face rows;
`scripts/precompute_rig_caches.py --append` adds rows for new blend shapes and leaves the others
bit for bit. `anny.faces.distribution.FaceShapeDistribution` samples face values for given phenotypes
from `data/shape_calibration/face_prior.safetensors`, built by `python -m anny.faces.authoring.calibrate`
(sources and licences in `data/faces/SOURCES.md`): the covariance of the ICT fits with one variance
factor per group, and means fitted by MAP around anny's default face (prior SD 0.3 per named shape)
to the anthropometric targets of each anchor. Check changes to the calibration by rendering random
faces in the viewer as well as by the measurement check: fits that meet the numbers can still give
implausible faces.

### Pose Parameterization

Five built-in variants: `local-ref` (the `Anny()` default), `local-bone`, `local-bone-world`, `world`, `world-orient`. Selected via the `pose_parameterization` argument to `Anny()`. The deprecated `create_fullbody_model(...)` preserves the old `local-bone` default.

### Authoring Rig and Viewer Frame

The pose, corrective and hair authoring code (`*/authoring/`) works on the authoring rig of `poses/authoring/rig.py`: anny's default body in the frame of the legacy pose library (metres, Y up, X toward the figure's left, Z forward, uniform scale 0.8916 so that the left eye stands at a fixed height). The viewer page draws in the same frame. `authoring_rig(phenotype)` builds it for other slider values, and the `ANNY_AUTHORING_PHENOTYPE` environment variable (a JSON object) sets the body of the authoring tools, e.g. for `python -m anny.correctives.authoring.evaluate`.

### Web Viewer

`viewer/` is a node project (TypeScript, three.js, esbuild). `src/anny_shape.ts` and `src/subdivision.ts` repeat anny's coefficient maths and the subdivision, and `test/test_viewer_parity.py` checks them against Python. `src/body.ts` rebuilds the fine body for any slider setting, `src/shading.ts` holds the shaders, and `src/main.ts` holds the renderer and the panels. `npm run build` writes the single-file page `viewer/dist/anny_viewer.html`; `npx tsc --noEmit` type-checks.

### Optional Dependencies

- `smplx` — required for `SMPL` and `SMPLX` model classes; install via `uv sync --extra smpl`
- `trimesh`, `gradio`, `jsonargparse`, `requests` — needed only for examples and parameter regression tests
- `scipy`, `tetgen`, `embreex` — needed for the viewer build and the authoring tools; install via `uv sync --extra viewer` (the page build also needs node 22 or later)
- `mediapipe`, `pyarrow`, `playwright` — needed for the face calibration and the photo benchmark (`python -m anny.faces.authoring.*`); install via `uv sync --extra faces` (MediaPipe needs the system libraries `libegl1` and `libgles2`)
