# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**Anny** is a differentiable, parametric body mesh model in PyTorch with a common topology and parameter space. Based on MakeHuman assets, it provides full-body, hand, and face models.

This repository is **OpenSculptBoy**, an independent fork of Anny (NAVER LABS Europe) that adds the web viewer, face shapes, hair, poses and correctives. The README states that AI agents write and maintain the additions under one human author, and it carries the credits and the no-warranty disclaimer; keep those sections accurate when you change the project. The Python distribution is `opensculptboy` (`pyproject.toml`) and holds two packages: `anny` (`src/anny`, the body model, kept close to upstream so that merges stay clean: `git fetch upstream main`, remote `upstream` is `https://github.com/naver/anny`) and `opensculptboy` (`src/opensculptboy`, this project's additions on top of the model: `Character`, the glTF and VRM export and the `opensculptboy` command). Put new features that only use the model's public API in `opensculptboy`, and change `src/anny` only where the model itself must change. Use "OpenSculptBoy" for the project and "Anny" for the model and the `anny` package, and never present the fork as an official NAVER release. In the README, the viewer and other user-facing text, present the models as humanoid 3D characters for free, unlimited creativity: never as humans, and never by human age. The `age` parameter keeps its name in the API and appears as "Form" in the viewer, with the ends "Compact" and "Elongated".

### Conventions

- Write Markdown docs with one line per paragraph or list item. Never hard-wrap prose; editors and GitHub wrap it for display.
- Bring upstream updates in their own pull request: a branch from `main` with `git merge upstream/main`, merged with a merge commit (never squashed or rebased), so that git keeps tracking which upstream commits the fork has. Keep upstream commits out of feature pull requests.

## Commands

### Setup
```bash
uv sync --extra examples  # full install with demo dependencies
```

### Testing

The checks come in three tiers, and `scripts/check.sh` runs each of them. GitHub Actions (`.github/workflows/lint.yml`, `viewer.yml` and `tests.yml`) calls the same script, so a local run of a tier matches CI.

| Tier | Where | Command | Runs |
|---|---|---|---|
| quick | locally, while working | `scripts/check.sh quick test.test_gltf_export ...` | lint (`uv lock --check`, ruff at the version pinned in the `dev` extra, copyright headers), then the test modules you name, with the CI settings |
| ci | GitHub Actions on every pull request and push to `main`; locally before a push | `scripts/check.sh ci` | lint; the viewer type-check and stale-page check; the whole suite with `OPENSCULPTBOY_CI=1` and `OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1`; the wheel installed into a fresh environment with CPU torch |
| full | locally only | `scripts/check.sh full` | the ci tier, with the local-only tests and the non-commercial data tests where their tools and data are present |

Rules for agents:
- While working, run `scripts/check.sh quick` with the test modules your change touches. Before a push, run `scripts/check.sh ci`, or at least the jobs your change touches (`scripts/check.sh lint|viewer|tests|package`).
- `OPENSCULPTBOY_CI=1` skips the tests marked `@local_only(reason)` (`test/markers.py`). Mark a new test `local_only` when it needs something CI lacks: Playwright with the built page (`test_hair_page`), the viewer data build (`test_viewer_body`, `test_hair_parity`, `TestPageSolver`), MediaPipe's models (`TestPoseFromImage`), the viewer's node packages three-vrm and gltf-validator with Playwright and Chromium (`test_vrm_three`, with the `faces` extra and after `cd viewer && npm ci`), or licensed files (`test_smpl`).
- `OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1` makes the download of the non-commercial SMPL and SMPL-X data raise `unittest.SkipTest` (`test/__init__.py`), so CI never fetches it. Put each topology of a loop in its own `subTest`, so that the smpl and smplx cases skip alone.
- `lint.yml` runs on every pull request and on every push to `main`, without path filters. The other workflows run on the same events only when their files change (GitHub `paths:` filters): `tests.yml` (the suite and the package) for changes to `src/`, `test/`, `viewer/src/`, `viewer/test/`, `pyproject.toml`, `uv.lock`, `MANIFEST.in`, `scripts/check.sh` or the workflow; `viewer.yml` for changes to `viewer/`. Draft pull requests skip the tests and the package; marking a pull request ready for review runs them. A change to the docs alone runs lint only. The repository is public, so GitHub-hosted runners cost nothing; the filters save runner time. When you add a directory that the tests read, add it to the paths of `tests.yml`.
- `test_vrm_schema` validates the VRM extensions of the shared exports against the VRM schemas and the glTF base schemas. The vrm-c/vrm-specification repository has no licence, so never vendor its schemas: the test downloads them at the commits pinned in `VRM_SPEC_COMMIT` and `GLTF_COMMIT` into `ANNY_CACHE_DIR/vrm_schema/<commit>/` and skips offline. Its structural checks (`check_vrm1`, `check_vrm0`) need no download and are the CI guard for both versions.
- `test/vrm_fixtures.py` caches the shared VRM exports of one character (`vrm_export`, `vrm_json`, `vrm_build`, `vrm_glb`); every VRM test module uses them instead of exporting again. `test/gltf_reader.py` evaluates a GLB or VRM file as an engine does (`evaluate_nodes` skins every primitive of every skinned mesh node; `structure_errors` lists dangling indices and undeclared extensions).
- `TestGltfReference` (`test/test_gltf_export.py`) guards the GLB output against `test/data/gltf_reference.json` (format `opensculptboy/gltf-reference@2`): the structure of seven exports (among them the float32 export of the command, a body with every phenotype and local changes, and a head part model with dense morph targets), with a SHA-256 of each integer accessor and order-sensitive sums of each float accessor, so that moved rows, flipped triangles and wrong indices fail it. The float32 case allows for the rounding of other CPUs, so the test also compares it value by value with a float64 export of the same character. The exports are byte-identical to those of the exporter before the VRM work, except for the `target` of dense morph-target buffer views. After a deliberate change of the GLB output, record it again with `uv run python -c "from test.test_gltf_export import write_reference; write_reference()"`.
- A change to `viewer/src` or `viewer/shell.html` needs the rebuilt page committed (`cd viewer && node build.mjs --reuse-data`); the viewer job fails on a stale `viewer/dist/anny_viewer.html`.
- NAVER's cluster workflow (`.github/workflows/tests.yml`, `.github/scripts/ci-install.sh`) is deleted in this fork. When an upstream merge touches those files, keep them deleted.

```bash
uv run --extra dev --extra examples python -m unittest test.test_various  # a single test file, as in the quick tier
```

### Documentation
```bash
bash build_doc.bash  # build HTML docs from the jupytext py:percent tutorials in tutorials/*.py
```

### Export (OpenSculptBoy)
```bash
uv run opensculptboy character > card.json                                  # a character card with the default settings
uv run opensculptboy export out.glb --character card.json --animation walk  # glTF 2.0 binary; --morph-targets all, --influences 8
uv run python -m unittest test.test_gltf_export                             # evaluates the file as an engine does and compares it with Anny
uv run opensculptboy export out.vrm --character card.json --author me       # VRM 1.0; --vrm-version 0, --thumbnail auto, --meta meta.json, --bind forward, --budget strict
uv run python -m unittest test.test_vrm_export test.test_vrm_tpose          # the VRM files against the T-pose rebind, both versions
uv run --extra dev --extra faces python -m unittest test.test_vrm_schema test.test_vrm_three  # VRM schemas, three-vrm and the Khronos validator (the last two need cd viewer && npm ci)
node viewer/test/vrm_page/validate.mjs out.vrm                              # the Khronos glTF validator on a GLB or VRM file (JSON report)
uv run --extra pose opensculptboy pose photo.jpg --card pose.json --svg pose.svg --png pose.png   # a character posed from a picture, drawn flat
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
uv run python -m anny.faces.authoring.review        # review.png in ANNY_CACHE_DIR/faces: random faces to judge by eye
```

### Hair
```bash
uv sync --extra viewer
uv run python -m anny.hair.authoring.layout                 # data/hair/scalp_layout.safetensors
uv run python -m anny.hair.authoring.presets                # grow every style: data/hair/styles/*.json and styles.safetensors
uv run python -m anny.hair.authoring.presets --only lob     # grow some styles
uv run python -m anny.hair.authoring.presets --specs-only   # rewrite the specs alone (render, physics, controls)
uv run python -m anny.hair.authoring.benchmark              # load, slider and render times of the built page
uv run python -m anny.hair.authoring.review                 # review.png in ANNY_CACHE_DIR/hair: every style at three lengths on three bodies
uv run python -m anny.hair.authoring.photos                 # photos.html in ANNY_CACHE_DIR/hair: the hairline of every style against FairFace photos (needs --extra faces)
```

### Web viewer
```bash
uv sync --extra viewer                # scipy, tetgen, embreex for the build
uv run python -m anny.viewer build    # data (cached under ANNY_CACHE_DIR/viewer) and the page viewer/dist/anny_viewer.html
cd viewer && node build.mjs --reuse-data   # the page alone, with the model data of the last build (for changes to src/ or shell.html)
cd viewer && npx tsc --noEmit         # type-check the page
uv run --with playwright python -m anny.viewer.benchmark   # uploads, draws and vertices of a moving frame
```

## Architecture

### Entry Points

`src/anny/__init__.py` exports the public API:
- `Anny(...)` — the model class, for full-body and part models alike; calling `anny.Anny(...)` builds a model and `isinstance(model, Anny)` holds for any Anny model. Accepts `rig`, `topology`, `local_changes`, `facial_actions`, `face_shapes`, `phenotypes`, `extrapolate_phenotypes`, `pose_parameterization`, and `skinning_method`.
- `AnnyInverter`, `KeypointsRegressor`, `Anthropometry` — see Key Subsystems.
- `create_fullbody_model(...)` — deprecated legacy full-body factory. It preserves the old default rig preset (`rig="default"`) and old full-body defaults; prefer `Anny(...)`.
- `create_hand_model()` / `create_head_model()` — deprecated part-model factories; use `Anny(rig="anny-hand.R", topology="hand.R")` or `Anny(rig="makehuman-head", topology="head")` instead.

### Core Class Hierarchy

- **`RiggedModelWithLinearBlendShapes`** (`models/rigged_model.py`) — base class; holds template vertices/faces/blend shapes, implements forward kinematics and LBS. The `bone_orientation` parameter (`"blender"`, `"procrustes"`, or `"cached"`) selects how rest bone orientations are computed: `"blender"` from bone tails, `"procrustes"` by aligning each bone to the vertices it skins, and `"cached"` by the same alignment from precomputed cross-covariance data (`data/cached/{anny,soma}.pth`, generated by `scripts/precompute_rig_caches.py`).
- **`Anny`** (`models/phenotype.py`) — inherits directly from `RiggedModelWithLinearBlendShapes`; adds the 9 phenotype dimensions (gender, age, muscle, weight, height, proportions, race, cupsize, firmness) and computes blend shape coefficients from these semantic scalars.
- **`SMPL`** / **`SMPLX`** (`models/smpl.py`) — first-class model types that also inherit directly from `RiggedModelWithLinearBlendShapes`; wrap the `smplx` library and follow the same initialization pattern as `Anny`, but accept `betas` + pose parameters instead of phenotype dimensions. Require the optional `smplx` package (`uv sync --extra smpl`).

### Rigs & Topologies

**Rigs** (`anny`, `makehuman`, `cmu_mb`, `game_engine`, `mixamo`, `soma`): MakeHuman-derived bone hierarchies are defined as JSON in `src/anny/data/mpfb2/rigs/standard/`; the `soma` rig lives in `src/anny/data/soma/`. `anny` is the default: the MakeHuman source rig with the `notongue`, `nobreasts`, `nofacialexpression`, and `pruned` modifiers, and `"cached"` bone orientations. Rig specs accept `-`-separated modifiers, including the subtree selectors `head`, `hand.L`, and `hand.R` (for `anny` and `makehuman` only). `default` is a legacy preset accepted only by `create_fullbody_model(...)` and preserves the old full MakeHuman rig defaults. `makehuman` is the full MakeHuman rig with tail/blender orientation and `root_identity_orientation=True`. Rig orientation is part of rig resolution; public constructors do not accept a separate `bone_orientation` argument.

**Topologies**: `anny` (the default MakeHuman mesh, with nudity edits), `makehuman` (the unedited MakeHuman mesh, quads and unattached vertices kept), the part meshes `head`, `hand.L`, and `hand.R`, and the alternative topologies `smplx`, `smpl`, `soma`, `anny_from_soma`, `notoes*`, and `legacy_default`. `default` is accepted only by `create_fullbody_model(...)`. Alternative topologies are built by retopology (`models/retopology.py`) onto target meshes stored in `src/anny/data/topology/` and `src/anny/data/soma/`. The SMPL and SMPL-X topologies are non-commercial only.

### Key Subsystems

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| Forward kinematics | `utils/kinematics.py` | Tree traversal with parallel propagation fronts |
| Skinning | `skinning/skinning.py` | LBS and dual-quaternion skinning (`skinning_method="lbs"` / `"dqs"`) |
| GPU skinning | `skinning/warp_skinning.py` | `warp-lang` accelerated LBS (`skinning_method="warp_lbs"`) |
| Collision | `utils/collision.py` | Self-intersection detection, warp-accelerated |
| Model data | `models/model_data.py` | `ModelData` / `ModelMetadata` dataclasses; bundle template mesh, blend shapes, and rig data; safetensors serialization for caching |
| Model transforms | `models/model_transforms.py` | `ModelData` → `ModelData` operations: retopology (from a mesh, or from linear combinations of template vertices), bone orientation conversion, mesh/skinning cleanups |
| Parameter regression | `anny_inverter.py` | `AnnyInverter`: iterative pose+shape fitting to a target mesh |
| Keypoints | `keypoints.py` | `KeypointsRegressor`: keypoints as linear combinations of mesh vertices (e.g. COCO, `data/keypoints/`) |
| Facial actions | `models/facial_actions.py` | Facial expression blend shapes from the MakeHuman face units (`data/faceunits01/`) |
| Anthropometry | `anthropometry.py` | `Anthropometry`: body measurements (height, volume, mass) from mesh |
| Subdivision | `utils/subdivision.py` | Catmull-Clark as sparse linear operators; `MixedSubdivision` adds one level on a region (the head) |
| Pose library | `poses/` | 50 poses and 7 clips as `local-ref` parameters (`data/poses/`), grounding, stool; `poses/authoring/` builds the library on the authoring rig |
| Correctives | `correctives/` | `SoftTissueCorrectives` (hinge and cone drivers, shapes scaled with the local size; `data/correctives/`); `correctives/authoring/` holds the simulation, the fit and `evaluate` |
| Hair | `hair/` | `StrandBinding` ties strands to the skin at roots and tips; `chart.py` holds the scalp chart (hairline, fade) and the 8-bit curve codec; `layout.py` loads the scalp layout (`data/hair/scalp_layout.safetensors`: guide and render roots in progressive order, their weights, the simulated guides); `styles.py` loads the 25 styles (`data/hair/styles/*.json` and `styles.safetensors`) and repeats the page's passes A and B and its density volume in NumPy; `dynamics.py` repeats the page's solver; `hair/authoring/` builds the layout, grows the styles (`groom.py`, `presets.py`), the brows and lashes, benchmarks the page, renders the review grid and compares the hairline with photos (`photos.py`); `anatomy.py` gives the landmarks and the outer ears that place the hairline |
| glTF export | `opensculptboy/export/gltf.py`, `document.py`, `body.py` | `export_glb(path, character, model, animations, morph_targets, max_influences)` builds its file with `GltfDocument` (`document.py`: several meshes, primitives and skins, embedded PNG images and samplers, `pbr_material` and `unlit_material`, every extension named under `extensions` listed in `extensionsUsed`) and the body data of `body.py`, which the VRM export shares (`morph_target_rows`, `rest_rows`, `build_body`, `body_primitive`): the rest mesh split at UV seams (`_ANNY_VERTEX` maps back to Anny's vertices), normals from the welded mesh, the strongest 4 or 8 skin weights, and morph targets as exact differences of rest meshes (facial actions; face shapes as `.pos`/`.neg`; sparse when few vertices move). The file also holds the rig as joints with inverse bind matrices, and the `anny.poses` entries as animations. Anny's Z-up, -Y-facing frame maps to glTF's Y-up, +Z-facing frame by C = (x, y, z) -> (x, z, -y): positions C p, joint matrices C M C^T. The `Character` settings travel in the scene extras, and `read_character` reads them back from `.glb` and `.vrm` files |
| VRM export | `opensculptboy/export/vrm.py`, `budget.py` | `vrm_spec(character, model, version, ...)` builds a `VrmSpec` (nodes, skins, meshes with targets, materials, humanoid map, expressions with binds per mesh, look-at, first person, meta and thumbnail) in the frame of its version: VRM 1.0 maps (x, y, z) -> (x, z, -y) and faces +Z; VRM 0.x maps (x, y, z) -> (-x, z, y) and faces -Z. `write_vrm(spec, path, budget)` writes VRM 1.0 (`VRMC_vrm`, `VRMC_materials_mtoon`, `VRMC_node_constraint`, sparse targets) or VRM 0.x (`VRM`, dense targets, `materialProperties`) and checks it against `budget.BUDGETS`; `export_vrm` runs both. Before any model is built, `vrm_spec` checks the topology (`check_topology`: the MakeHuman body mesh with its eyes, so that SMPL data is never downloaded), the rig, the twist mode of the version, the metadata (`VrmMeta.check_fields`: real booleans for the permissions, strings for the text) and a given thumbnail (a square PNG or JPEG file). The command reports these errors in one line with exit status 2, and a file over a strict budget with exit status 1. The joints are normalised: identity rotations, translations P_j - P_parent and inverse bind matrices translate(-P_j). Face shapes are baked; the facial actions are the targets. The card, the options and the counts go in the scene extras |
| VRM T-pose | `opensculptboy/export/tpose.py` | `vrm_t_pose(model, B)`: the world matrices T_j of an exact VRM T-pose (arms, hands and fingers along +-X with the palms down; thumbs level at 45 degrees toward -Y, the last thumb bone rolled so that the nail faces outward as VRM T-pose definition 1.8 asks (`THUMB_ROLL`: a roll of the other thumb bones folds the web of the thumb, which `test_thumb_web` and `test_base_thumb_bones_take_no_roll` guard); hip to ankle vertical unless `keep_leg_spread`; feet along -Y; spine, neck, head, eyes, clavicles and shoulder01 keep B_j), placed by world-orient FK. `rebind(...)`: the inverse bind (exact at Anny's rest pose with the file's 4 weights, the default) or the forward bind (Anny's skinning into T), the targets (delta' = A_i delta), the normals (for the inverse bind, Anny's rest normals mapped as the targets, exact at rest), the joints and the centring `offset`. `file_pose` gives the joint matrices of a posed file; `bind_error` gives the error of a bind on library poses with the file's weights (`file_skin_weights`), and `humanoid_error` the error when an app drives only the humanoid bones (with the roll constraints, and with the other turns folded into the humanoid bones; `poses`, `constraints` and `region` test a pronated forearm with and without the constraints). `test_bind_error` holds both binds within 0.9 to 1.1 times the measured values (`MEASURED_BIND_ERROR`); measure again when a change moves them |
| VRM tables | `opensculptboy/export/vrm_tables.py`, `data/vrm/expressions.json` | The humanoid bone maps of VRM 1.0 and 0.x for the `anny` rig; roll constraints on the forearm twist bones in VRM 1.0 (`file_parents` hangs the wrist from lowerarm01; the T-posed shin runs off the vertical, by 5 degrees on the default body and 3 to 8 degrees over the phenotype corners (8, and 5 to 14, with `keep_leg_spread`), so the shin has no roll constraint); the VRM skin weights (the eyelid weight of the eye bones moves to head, the upper arm, thigh and shin twist weights merge into their mapped bones, and the forearm twist weights too with `--twist merge`); `eyeball_vertices` finds the eyeballs by their MakeHuman base-mesh indices and raises ValueError on a mesh without them; the expressions as ARKit mixes (14 presets, 52 PascalCase perfect-sync customs, and the 10 VSeeFace visemes in VRM 0.x only); lid-only `eyeLook*` targets; look-at ranges from a Kabsch fit of the eyeballs. `test.test_vrm_expressions` measures the visemes on the mesh: keep its bands when you change the mixes, and review the faces by eye |
| MToon materials | `opensculptboy/export/mtoon.py` | `ToonMaterial` (linear colours, MToon 1.0 factors); `to_vrm1` (a glTF material with `VRMC_materials_mtoon`) and `to_vrm0` (the glTF material and the 0.x `materialProperties` entry with every MToon 0.x property: colours in the exact sRGB encoding that MToon 0.x in Unity decodes (three-vrm reads them within 0.009), the toony/shift pair as the exact inverse of the mapping of three-vrm and UniVRM, keywords, render queues and tags); `representable` projects what VRM 0.x cannot hold and notes what its readers show differently, such as the unmasked matcap; `v0_to_v1` ports three-vrm's 0.x reader for the tests |
| Viewer data | `viewer/` | `python -m anny.viewer build` writes the page data to `viewer/build/` and runs the node build of `viewer/` |
| Pose from a picture | `opensculptboy/posing/`, `opensculptboy/render/` | `landmarks.detect` runs MediaPipe's pose, hand and face landmarkers (models cached in `ANNY_CACHE_DIR/opensculptboy/models`) and returns `Landmarks` in Anny's frame; `skeleton.Skeleton` poses Anny by world rotations (`params` gives `local-ref` parameters, `hinge` keeps elbows, knees and fingers in one plane); `retarget.AnnyLandmarks` places MediaPipe's landmarks on Anny's joints and vertices, and `retarget.Retargeter` turns landmarks into rotations; `head.py` fits the head robustly (every triple of points proposes a rotation, the points that agree refine it, and a free scale allows a larger drawn head), and `Retargeter.head_fit` takes the first believable source: the face mesh, the 3D head points when 7 of 11 agree, their picture positions alone, or the chest; `plausible_hand` keeps the rest fingers for a hand with impossible proportions; `adjust_head` turns the head further by hand (turn, up, tilt in degrees, spread over the neck), for the CLI's `--head-*` options and the viewer's head sliders; `refine.refine` turns the torso and the limbs until Anny's landmarks meet the picture's, with a robust loss and without the rejected head points; `picture.pose_from_image` fills `Character.pose` and `facial_actions`; `render/flat.py` draws a posed mesh as a silhouette SVG, an outline SVG or a toon PNG (`shaded_png`; with `smooth=True`, which the VRM thumbnail uses, each pixel takes the band of its interpolated normal). The viewer repeats the retarget in `viewer/src/pose_from_image.ts` (checked by `test/test_viewer_pose_parity.py`) and loads MediaPipe from its CDN (`viewer/src/picture.ts`) |
| Face shapes | `models/face_shapes.py`, `faces/` | 103 named, symmetric face-shape parameters and 10 detail shapes from ICT-FaceKit (`Anny(face_shapes=...)`, `face_shape_kwargs`), scaled per group with the size of the head; craniofacial landmarks and the measurements of 3D Facial Norms and ANSUR II (`faces/measurements.py`); a face-shape distribution calibrated against measured faces (`faces/distribution.py`); `faces/authoring/` fetches the sources, fits the ICT-FaceKit identity space, calibrates the distribution and benchmarks against FairFace photos |

### Phenotype System

Phenotypes are blended linearly between discrete anchor states defined in `src/anny/data/mpfb2/targets/`. Default mode omits race, cupsize, and firmness; pass `phenotypes="all"` to enable them. Blend shape data is computed at model creation and cached in `~/.cache/anny/`. Set the `ANNY_CACHE_DIR` environment variable to use a different location.

### Face Shapes

`Anny(face_shapes="all")` (or a list of names) adds the face-shape block: each parameter of `data/faces/face_shapes.json` sums the left and right MakeHuman targets of the head, forehead, brows, eyes, nose, cheeks, mouth, chin and ears into rows `face_shape:{name}.pos` and `.neg`; +1 applies the positive targets, -1 the negative ones, and the head archetypes and `chin-triangle` run from 0 to 1. The `detail` parameters (`source: "ict"` in the spec) come from `data/faces/detail_shapes.safetensors`: the symmetric principal components of the residuals of the ICT-FaceKit fits, written by `python -m anny.faces.authoring.detail`. The model cache key carries a digest of both files (`face_shape_data_digest`), and the calibration widens the slider ranges to hold the calibrated distribution. The rows of a group scale with the size of that part of the head (`Anny.face_shape_scales`), measured on the craniofacial landmarks of `data/keypoints/craniofacial.json`, which `ModelData` stores for the template and every blend shape. The `anny` and `soma` rig caches carry the face rows; `scripts/precompute_rig_caches.py --append` adds rows for new blend shapes and leaves the others bit for bit. `anny.faces.distribution.FaceShapeDistribution` samples face values for given phenotypes from `data/shape_calibration/face_prior.safetensors`, built by `python -m anny.faces.authoring.calibrate` (sources and licences in `data/faces/SOURCES.md`): the covariance of the ICT fits with one variance factor per group of at most 1 (no detail shapes below 18 years, and `head-age` held at 0), around anny's default face for each age. Only the skull-size shapes (`MEAN_SHAPES`) move the mean, by MAP fits to the head measurements of each anchor; the facial features stay at anny's default face, because means that also moved them met the nose, lip and face-depth targets through big noses, forward chins and thin lips, which looked old and harsh. Check changes to the calibration by rendering random faces in the viewer as well as by the measurement check: fits that meet the numbers can still give implausible faces. Judge each random face by whether it passes as a normal person of that age, and show the calibrated mean in the grid, since the default face hides a shift of the mean: `python -m anny.faces.authoring.review` renders that grid from the viewer page. `test.test_faces_calibration.TestPlausibleFaces` guards these choices and the facial spread of the approved prior (`APPROVED_FACIAL_SPREAD`); when a change breaks one of its tests, review the grid with the user before updating the test.

### Hairline

Every style shares anny's default hairline (`HAIRLINE_PHI` and `HAIRLINE_EL` in `hair/chart.py`: the minimum elevation of hair against the azimuth about the cranium centre), and the style fields count degrees above it. The hairline rests on anny's landmarks (`hair/authoring/anatomy.py`: the craniofacial landmarks and the outer ears, which MakeHuman's ear-translation targets move in full): the sideburn comes down in front of the ear to the tragion, and the hair meets the ear about 2 mm from its front and its top. `test.test_hair_styles.TestHairlineAnatomy` checks both, and the layout leaves out the ears (`on_ear`). The front and the temples rest on `python -m anny.hair.authoring.photos`, which measures the hair edge (forehead, pupils, temples, sideburns) on FairFace photos and on anny's portraits with MediaPipe; the visible edge lies below the natural hairline wherever a fringe falls, so the hairline sits near the photos' upper quartile. A change to the table needs `python -m anny.hair.authoring.layout`, then `presets` and the viewer build; check it with the anatomy tests, the photo benchmark and the review grid.

### Pose Parameterization

Five built-in variants: `local-ref` (the `Anny()` default), `local-bone`, `local-bone-world`, `world`, `world-orient`. Selected via the `pose_parameterization` argument to `Anny()`. The deprecated `create_fullbody_model(...)` preserves the old `local-bone` default.

### Authoring Rig and Viewer Frame

The pose, corrective and hair authoring code (`*/authoring/`) works on the authoring rig of `poses/authoring/rig.py`: anny's default body in the frame of the legacy pose library (metres, Y up, X toward the figure's left, Z forward, uniform scale 0.8916 so that the left eye stands at a fixed height). The viewer page draws in the same frame. `authoring_rig(phenotype)` builds it for other slider values, and the `ANNY_AUTHORING_PHENOTYPE` environment variable (a JSON object) sets the body of the authoring tools, e.g. for `python -m anny.correctives.authoring.evaluate`.

### Web Viewer

`viewer/` is a node project (TypeScript, three.js, esbuild). `src/anny_shape.ts` and `src/subdivision.ts` repeat anny's coefficient maths and the subdivision, and `test/test_viewer_parity.py` checks them against Python. `src/body.ts` rebuilds the fine body for any slider setting, `src/shading.ts` holds the shaders, and `src/main.ts` holds the renderer. `main.ts` hands the interface an `App` object (`src/ui/app.ts`) and calls back through `HOOKS`; the typed modules of `src/ui/` build the interface: `index.ts` (the top bar, the view bar, undo, the shortcuts, the name of the bone under the pointer), `segments.ts` (the Character segment on the left and the Stage segment on the right, each an icon rail and an inspector, bottom sheets on a phone), `character_ui.ts` (Characters, Body, Face, Hair, Skin & eyes), `stage_ui.ts` (Pose, Scene with the frame rate limit), `rig_ui.ts` (the Rig section: the one place for the bones and the skin weights, with the view of the rig, the chosen bone and the list of the bones), `controls.ts` (the slider with its typed value and reset), `camera_nav.ts` (pan within a box around the figure, the floor limit, the height rail, the double-click focus, the angles, the film offset that centres the figure between the inspectors), `stats.ts` (the performance card) and `icons.ts`. `test/vrm_page/` is the three-vrm test page of `test/test_vrm_three.py`, bundled with esbuild at test time: `@pixiv/three-vrm` and `gltf-validator` are devDependencies for the tests, and nothing in `src/` imports them. `npm run build` writes the single-file page `viewer/dist/anny_viewer.html`; `npx tsc --noEmit` type-checks. The page keeps one `<canvas>` (the screenshot tools select it), so the graphs of the interface are SVG, and `?shot` hides the whole interface. The skeleton (`SKEL` in `main.ts`) and the weight view (`WEIGHTS`: the body's geometry and skin with a material that colours the weights, and the chosen bone that both views share; `setRigView` picks the view) draw into a multisampled overlay (`OVL`) that blends over the canvas after the display pass, so the tone mapping and the accumulation leave them alone; the frame rate limit (`FPS` in `main.ts`) skips display frames in the loop, and the dynamic resolution judges slow frames against it. The Body section holds the sliders of `anny.viewer.build.SLIDERS`: anny's six default phenotypes and the three race phenotypes, whose values mix by their shares (the build uses `phenotypes="all"`, and 136 components reproduce the blend shapes exactly). The Characters section of the Character segment holds presets (`CHARACTERS` in `src/main.ts`) that set the phenotype sliders, the face and the colours, as a character creator's presets do; the colour looks set the colours only. Random face draws with `sampleFace` (`src/anny_shape.ts`) at the spread of the exported prior, and `test/test_viewer_parity.py` checks it against Python. The hair lives in `src/hair/`: `data.ts` decodes the scalp layout and the styles and builds the density volume, `glsl.ts` holds pass A (the guides follow the body, the pose and the physics) and pass B (the render strands, into a float texture that the ribbons read), `gpu.ts` holds the `Hair` class, `sim.ts` the solver and `colliders.ts` its capsules. Each piece of data lives at the level where it changes (layout, style, body, pose), and every style control is a uniform or an instance count. `Hair.setLod` draws a prefix of the render roots; `main.ts` sets it from the size of the head on screen (`updateHairLod`). Moving frames render at the scale of `DYN` in `main.ts` (the dynamic resolution), and the build puts the vertices of the corrective shapes first (`HOT` in `build.mjs`), so a frame of a clip uploads one small range. `python -m anny.viewer.benchmark` counts the uploads, draw calls and vertices of a moving frame, and the Performance card (the chart button of the top bar, the `` ` `` key or `?stats=1`) shows the frame rate, the CPU and GPU time of a frame, the draw calls, the memory and the load times, and runs a 10-second benchmark (`window.frameStats`, `window.__benchmark`). `test/test_hair_parity.py`, `test/test_hair_page.py` and `test/test_hair_dynamics.py` check the page's hair against `anny.hair.styles` and `anny.hair.dynamics`.

### Dependencies

`warp-lang`, `trimesh`, and `requests` are core dependencies. Optional extras:

- `smpl` (`smplx`, `chumpy`) — required for the `SMPL` and `SMPLX` model classes; install via `uv sync --extra smpl`
- `examples` (`gradio`, `jsonargparse`, `matplotlib`, `scipy`, `jupytext`, `notebook`, `py-soma-x`, ...) — needed for examples, tutorials, and SOMA comparison tests. `py-soma-x` is pinned to match `anny.models.soma.SOMA_ASSETS_REVISION`.
- `viewer` (`scipy`, `tetgen`, `embreex`) — needed for the viewer build and the authoring tools; install via `uv sync --extra viewer` (the page build also needs node 22 or later)
- `faces` (`scipy`, `mediapipe`, `pyarrow`, `matplotlib`, `playwright`) — needed for the face calibration and the photo benchmark (`python -m anny.faces.authoring.*`); install via `uv sync --extra faces` (MediaPipe needs the system libraries `libegl1` and `libgles2`)
- `pose` (`mediapipe`, `contourpy`) — needed for the pose from a picture (`opensculptboy pose`, `opensculptboy.pose_from_image`) and the flat drawings; MediaPipe needs the system libraries `libegl1` and `libgles2`
- `dev` (`ruff`, `build`, `twine`, `jsonschema`) — formatting checks (`test/test_ruff.py`), packaging and the VRM schema checks (`test/test_vrm_schema.py`)
