# Baseline body

This folder holds version 14 of the viewer. The baseline head sits on a full body with medium tousled hair, and a skeleton moves the figure through poses and animations. Corrective shapes from a soft-tissue simulation keep the volume of the joints as they bend, and the shoulder girdle follows the arms. The Pose panel holds 40 poses from the MakeHuman community pose packs, which are free to use under CC0. This version adds the collarbones and a fuller biceps to the body. The MakeHuman data and the head-only version stay in the `baseline_head` folder next to this one.

## Files

- `body_viewer.html` is the interactive viewer. Open it in a desktop browser with an internet connection, because the page loads three.js and two fonts from public CDNs. The Body and Face buttons switch between the full figure and a close view of the head. The Pose button opens the poses and animations, and the Character button opens the character panel.
- `baseline_body_glb.zip` holds `baseline_body.glb` with six meshes. The body and the two eyes form the figure, and the trunks, the shorts and the T-shirt are separate wearables over it. The body is one closed surface from the scalp to the soles, in the default body shape and the A-pose. The model stands 1.63 m tall in metres with Y up, and the colours are stored as vertex colours. The skeleton and the animation clips live in the viewer for now.
- `baseline_hair_strands.zip` holds one OBJ file with the hair, brow and lash strands as polylines at the same scale as the GLB. Blender imports each polyline as a chain of edges, and Object > Convert > Curve turns them into curves.
- `correctives.json` describes the drivers of the corrective shapes with the bone names of the rig. The section Soft tissue explains the format.
- `source.zip` holds the build scripts. `build/pipeline.sh` lists the steps in order, and the scripts still point at the folders of the cloud workspace where they were written.

## Poses and animations

The Pose panel holds seven animations and 50 poses in eight groups. The animations come first, and each group of poses folds open and shut; the group of the current pose opens by itself. A change of pose or animation blends over 0.4 seconds. The browser keeps the last choice for the next visit.

| Group | Poses |
| --- | --- |
| Reference | A-pose, T-pose |
| Standing | Relaxed, Weight on one leg, Hands on hips, Arms crossed, Mid-step, Hand on hip, At attention, At ease |
| Gestures | Thinking, Salute, Peace sign, Pointing, Waving goodbye, Cheering, Hero stance |
| Action | Ready stance, Reaching up, Crouch, Casting a spell, Arms spread, Low crouch |
| Martial arts | Fighting stance, Low guard, High kick, Block and strike, Boxing guard, Palm strike, Focus |
| Sports and fitness | Sprint, Tennis serve, Backhand, Star pose, Triangle pose, Toe-touch stretch, Cobra stretch, Push-up, Splits, Handstand |
| Sitting | Seated, Chin on hand, Talking with hands, Explaining, Ankle on knee |
| On the floor | Leaning back, Cross-legged, Hugging knees, Side sit, Head in hand |

In each group the first poses belong to this library, and the rest come from the MakeHuman community. The section Pose sources names their authors. A wooden stool appears under every pose that sits on a chair.

| Animation | Length | Motion |
| --- | --- | --- |
| Idle | 6.0 s | The figure breathes and shifts its weight slowly. The head turns to each side. |
| Walk | 1.1 s | A walk in place. Each foot moves back along the floor while it carries the weight. |
| Run | 0.72 s | A jog in place with a short flight between the steps. |
| Wave | 3.2 s | The right hand comes up and waves, then goes down again. |
| Nod | 2.2 s | Two nods of the head. |
| Shrug | 2.6 s | The shoulders rise and the palms turn up. |
| Jump | 2.4 s | The figure crouches and jumps about 22 cm. It lands with bent knees. |

Every animation loops. Pause stops an animation at the current moment, and Half speed slows it down. The view sharpens while the figure holds still, so a pose or a paused animation reaches full quality. The Body view fits each pose: a raised arm or a jump gets more room above, and a seated or crouching figure comes closer. The Face view follows the head while the figure moves.

The hair, the brows, the lashes and the eyes move with the head. Each hair strand follows the skin under its root. The wearables follow the skeleton with the body, and the skin under a worn garment sinks by 4 mm, so bends of the body keep the skin inside the fabric. The sinking fades out a few millimetres past each edge, where an elastic band would press into the skin.

### Pose sources

The MakeHuman community shares poses for the MakeHuman default skeleton, which is the skeleton of this rig. The packs used here are the MakeHuman system poses and the community packs poses01 to poses05, 117 poses in all, each under CC0. CC0 asks for no credit, and the table credits the authors all the same. The panel shows the author under the poses too.

| Author | Poses |
| --- | --- |
| Anrico | Chin on hand, Talking with hands, Explaining |
| callharvey3d | At attention, At ease, Thinking, Salute, Toe-touch stretch, Ankle on knee, Cross-legged, Hugging knees |
| culturalibre | Casting a spell, Arms spread, Block and strike |
| Elvaerwyn | Hero stance, Star pose, Triangle pose, Cobra stretch, Push-up, Splits, Head in hand |
| gpedroso | Peace sign, Low crouch, Boxing guard, Palm strike, Focus |
| joachip | Tennis serve |
| MakeHuman | Mid-step, Hand on hip, Pointing, Fighting stance, Low guard, High kick, Sprint, Handstand, Leaning back |
| punkduck | Waving goodbye, Backhand |
| spreadcore | Cheering |
| wolgade | Side sit |

Forty poses made the selection. The rest were left out for one of these reasons:

- the pose is built around a prop the viewer does not have, such as a bicycle, a guitar, a glass, a counter, a bow, a gun or a hammer;
- a hand rests on another person;
- the figure floats in the air or swims;
- the pose is a fashion-model pose;
- the pose repeats a pose of the library.

`build/import_poses.py` reads a pose file and puts it on the rig in five steps:

1. It finds the axes of the file from its rest skeleton. MakeHuman writes files with Y up, and MPFB2 in Blender writes them with Z up.
2. It gives each bone the rotation of the joint with the same name. The files rest in the same A-pose as the rig, so the rotations carry over as they are, and the rig keeps its own proportions.
3. It sets the shoulder girdle by the rule of the pose library. Where a file raises the girdle further, 30% of the difference stays, because the authors posed adult bodies and the same raise looks larger on this figure.
4. It turns a figure that faces far to one side toward the front, and it lays a figure that lies on the floor along the width of the view.
5. It puts the lowest point of the body on the floor, with the hips over the middle of the floor. A figure that sits in the air, on a chair in the original pose, gets the stool under its buttocks.

### Shoulder girdle

In every pose and animation the shoulder girdle follows the upper arm, as the collarbone and the shoulder blade do in the body. The girdle stays still for the first 20° of arm elevation, and after that it takes a growing share of the motion. At the full height of 1.63 m, the shoulder joint rises by 2.2 cm with the arm level to the side and by 7.1 cm with the arm straight up. The girdle also comes forward when the arm reaches forward or across the chest, and it goes back a little when the arm goes behind the body. With the arms down along the sides, the shoulders sit 6 mm lower than in the rest pose. A shrug adds its own raise on top of this motion.

The pose library now gives the direction of each arm relative to the chest, so the girdle can rise while the arm keeps its direction. In version 11 the raised arm in Reaching up leaned past the vertical toward the back of the head, and the girdle rose too little. These faults stretched the skin over the top of the shoulder. The arm now points almost straight up, with a slight lean forward and out to the side.

## Soft tissue

Plain skinning moves each vertex with a weighted blend of bones. Where a joint bends far, this blend loses volume. An elbow or a knee thins out at the bend, and a shoulder or a hip folds in. The viewer adds corrective shapes at the shoulders, the elbows, the hips and the knees, and these shapes restore the volume. The Soft tissue switch in the Pose panel turns the shapes off, so plain skinning can be compared with the corrected body.

A soft-tissue simulation made the shapes. The simulation fills the body with 276,000 tetrahedra. Rigid cores follow the bones: the long bones of the limbs, the ends of the bones at the elbows and the knees, the kneecaps, the collarbones, the ribcage, the spine and the pelvis. The hands, the feet and the head follow plain skinning. The solver moves the rest of the tissue so that each tetrahedron keeps its volume and, as far as it can, its shape. On the inner side of an elbow or a knee, the skin above the joint and the skin below it press against each other in the fold.

The simulation runs once for each key pose of a joint. The difference between the simulated body and plain skinning, moved back into the rest pose, becomes one corrective shape. At every key pose the viewer matches the simulated surface around that joint, and between the key poses it blends the shapes. The key poses of the shoulder move the girdle by the same rule as the poses and animations.

| Joint | Driver | Key poses |
| --- | --- | --- |
| Elbow | Bend angle | 0°, 90° and 130°. The rest pose bends the elbow by 45°. |
| Knee | Bend angle | 0°, 50°, 90° and 130°. The rest pose bends the knee by 11°. |
| Shoulder | Direction of the upper arm in the frame of the chest | 13 directions between the arm down along the side and the arm straight up, including forward, back and across the chest. |
| Hip | Direction of the thigh in the frame of the pelvis | 10 directions between 25° back and 125° forward, including out to the side and in across the body. |

Each side has 30 shapes, and the right side mirrors the left side. At the rest pose every weight is 0, so the rest shape stays exactly as it is. The wearables follow the corrected skin through the same binding that the body sliders use.

The table compares plain skinning and the corrected body with a full simulation of poses from the library. It counts the skin vertices where plain skinning lies more than 2 mm from the simulated surface, and it gives the average distance of these vertices from the simulated surface.

| Pose | Vertices | Plain skinning | Corrected |
| --- | --- | --- | --- |
| T-pose | 16,258 | 5.8 mm | 0.7 mm |
| Reaching up | 29,153 | 6.0 mm | 2.7 mm |
| Arms crossed | 20,695 | 7.5 mm | 5.9 mm |
| Hands on hips | 18,807 | 7.2 mm | 5.9 mm |
| Ready stance | 36,720 | 9.8 mm | 4.5 mm |
| Seated | 40,811 | 12.3 mm | 5.9 mm |
| Crouch | 51,587 | 12.6 mm | 7.0 mm |

The remaining distance comes from poses between the keys and from joints that bend together, such as a hip and a knee in the crouch. The shoulder driver reads only the direction of the upper arm, so a twisted arm, as in Hands on hips, keeps more of the distance. Reaching up gains the most from this version, because its arm now lies close to the new key straight up. With the arm raised across the head or behind the head, the simulation folds the armpit in on itself, so the correction fades out toward these two directions.

### Drivers

A driver sets the weights of the shapes of one joint from the skeleton.

- A hinge driver measures the angle between the upper bone and the lower bone from the positions of the three joints. The weights interpolate linearly between neighbouring keys.
- A cone driver measures the direction of the upper bone in the frame of the torso bone above it. The key directions form triangles on a sphere, and the weights are the barycentric coordinates of the current direction in its triangle.

`correctives.json` describes every joint with the bone names of this rig. The function `correctiveWeights` in the viewer is a reference implementation of both drivers.

```json
{
  "name": "knee.L", "type": "hinge", "bones": ["upperleg01.L", "lowerleg01.L", "foot.L"],
  "keys": [
    { "angle": 0.0, "shape": "knee_000.L" }, { "angle": 10.749, "shape": null },
    { "angle": 50.0, "shape": "knee_050.L" }, { "angle": 90.0, "shape": "knee_090.L" }, { "angle": 130.0, "shape": "knee_130.L" }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `type` | `hinge` or `cone`. |
| `bones` | For a hinge, the three bones whose heads give the upper bone and the lower bone. |
| `keys` | For a hinge, the key angles in degrees with their shapes. The key with the shape `null` is the rest pose. |
| `bone`, `end` | For a cone, the upper bone and the bone at its far end. Their heads give the direction. |
| `frame` | For a cone, the torso bone whose frame holds the direction. |
| `targets` | For a cone, the key directions in the rest frame with their shapes. A target with the shape `null` carries no shape: the rest pose, and the directions across the head or behind the head, where the correction fades out. |
| `triangles` | For a cone, the triangles of key directions as indices into `targets`. |

The file also states its conventions, such as the axes, the exact rules for the weights and the rule that moves the shoulder girdle in the key poses of the shoulder. Animations that move the girdle by the same rule match the shapes best.

### Game engines

The shapes work like morph targets, which every engine adds to the rest shape before skinning. In Unreal Engine, a Control Rig or an Animation Blueprint can compute the driver weights and set the morph target curves. In Godot, a script next to the Skeleton3D can set the blend shape values of the mesh on every frame. The drivers use only bone positions and rotations, so they keep working when an engine retargets animations onto this skeleton. Animations from other sources move the collarbones in their own way, and the shoulder shapes then match less closely. A Control Rig or a script can set the collarbones by the girdle rule in `correctives.json`. The GLB file in this folder has no skeleton yet, so the corrective shapes live in the viewer for now.

## Body shape

The default build is slim. The table gives tape measurements of the figure at 1.63 m, for the default shape and for the two ends of the Body fat slider.

| Measurement | Lean end | Default | Heavy end |
| --- | --- | --- | --- |
| Chest | 76.5 cm | 77.7 cm | 83.9 cm |
| Waist | 57.5 cm | 60.2 cm | 72.5 cm |
| Hips | 74.7 cm | 77.0 cm | 86.2 cm |
| Upper arm | 19.5 cm | 22.0 cm | 27.4 cm |
| Thigh | 37.1 cm | 39.8 cm | 46.4 cm |

The upper arm is measured halfway between the shoulder joint and the elbow.

The collarbones and the biceps are sculpted into the body as a layer of relief. Each collarbone runs as a soft ridge from the notch at the base of the neck to the tip of the shoulder, with a hollow above it, and the neck muscles rise from the notch. The biceps gives the front of the upper arm a gentle belly, and a little triceps fills the back. Before this relief, the middle of the upper arm was a plain tube, thinner than the forearm below the elbow. The relief adds about 2 cm to the girth of the upper arm.

The Body section of the Character panel has eight sliders. Each slider runs from -1 to +1, and 0 is the default shape. The sliders change the body below the neck, so the head and the fit of the hair stay the same. The worn wearables follow the body as it changes, and the joints of the skeleton move with the body. Double-click a slider to return it to 0.

| Slider | Ends | Change |
| --- | --- | --- |
| Body fat | Lean to Heavy | Fat over the whole body. |
| Muscle | Slight to Athletic | Muscle over the whole body. |
| Shoulders | Narrow to Broad | Width of the shoulders and the taper to the waist. |
| Chest | Flat to Deep | Depth of the chest. |
| Belly | Flat to Round | Shape of the belly. |
| Hips | Narrow to Wide | Width of the hips and size of the buttocks. |
| Arms | Thin to Thick | Fat and muscle on the arms. |
| Legs | Thin to Thick | Fat and muscle on the legs. |

Some sliders act on the same place. Body fat and Arms both change the arms, for example. The viewer adds their moves along the skin with a soft limit, and this limit keeps combined settings within a natural range. A slider on its own always reaches its full end. The Shoulders and Hips sliders mostly move whole parts of the body, so the viewer adds them without the limit.

## Looks and presets

The Character panel changes the look while the viewer runs. The panel covers the body shape, the skin, the hair and eye colours, and the outfit. The browser keeps the current look for the next visit. Copy preset gives the look as JSON text, and Paste preset loads one.

A preset looks like this:

```json
{
  "format": "portrait-of-a-boy/look@3",
  "name": "Baseline",
  "skin": { "tone": 0.35, "undertone": 0 },
  "hair": { "color": "#271f16" },
  "eyes": { "color": "#875f3d" },
  "outfit": {
    "base": { "item": "trunks", "color": "#303235" },
    "top": { "item": null, "color": "#d9d5cc" }
  },
  "body": { "build": 0, "muscle": 0, "shoulders": 0, "chest": 0, "belly": 0, "hips": 0, "arms": 0, "legs": 0 }
}
```

| Field | Values | Effect |
| --- | --- | --- |
| `skin.tone` | 0 (fair) to 1 (deep) | Picks the base skin colour on a six-step scale. The baked skin colour was painted at 0.35, and every other tone tints it. |
| `skin.undertone` | -1 (cool) to +1 (warm) | Moves the base colour toward pink at -1 and toward golden at +1. |
| `hair.color` | sRGB hex | Sets the hair colour. The brows and the lashes follow it in darker shades, and the scalp under the hair takes a colour between the skin and the hair. |
| `eyes.color` | sRGB hex | Sets the iris colour as it reads under light. |
| `outfit.base.item` | `"trunks"`, `"shorts"` or `null` | Picks the base layer. With `null` the figure shows as a plain grey mannequin. |
| `outfit.base.color` | sRGB hex | Sets the fabric colour of the base layer. |
| `outfit.top.item` | `"tshirt"` or `null` | Puts the T-shirt on or takes it off. |
| `outfit.top.color` | sRGB hex | Sets the fabric colour of the top. |
| `body.build` to `body.legs` | -1 to +1 | Sets the body slider with the same name. The field `build` is the Body fat slider. |

Missing fields fall back to the baseline, and the viewer ignores fields it does not know. Presets in the earlier `look@1` and `look@2` formats still load, and they show the default body shape. The example looks in the panel set the colours and the outfit, and they keep the current body shape. The pose stays outside the preset. Later versions will add fields for the face shape.

## Wearables

Each wearable is a separate mesh over the complete body, with rolled edges where it ends. `build/build_wearables.py` builds all of them from the body surface. A wearable is a region of the skin cut along its edges, pushed outward by an ease that can grow toward the hems, smoothed so skin detail stays hidden, and kept clear of the skin.

| Wearable | Slot | Fit |
| --- | --- | --- |
| Trunks | Base layer | Snug, 1.5 to 2 mm off the skin, legs ending 5.7 cm below the crotch. |
| Shorts | Base layer | Athletic cut, snug at the waistband and about 1 cm loose at the hems above the knee. |
| T-shirt | Top | Crew neck and short sleeves, 5 mm off the chest and about 1.3 cm loose at the hem. |

The body keeps no trace of any wearable. Its occlusion is baked without wearables, and each wearable adds its own contact shade on the skin in one of four slots, so the shade appears only while that wearable is on. The viewer treats the base layer as one slot with the choices Trunks, Shorts and None. With None, the whole figure turns into a plain grey mannequin, which is the view for checking fit and body shape.

Each wearable vertex is bound to the four nearest body vertices. When a body slider moves the skin, the wearable moves with the skin under it and keeps its fit. Each wearable vertex also takes its skin weights from the same four body vertices.

## Skin shading

The skin shader splits the light into two parts. The diffuse light goes into a second render target, and a screen-space pass spreads it with a separable kernel that follows the d'Eon skin profile. This pass softens shadow edges and fine detail in the way that light scattering under real skin does. The specular light uses two lobes. The body is drier and rougher than the face, and the T-zone of the face is the oiliest part.

Thin parts let light through. The shader reads the thickness toward the key light from the shadow maps, and it uses a baked thickness for the rim light and the room light. The ears and the fingers glow red when a light sits behind them.

The surface detail comes from procedural patterns. Fine skin lines cover the whole body, and larger pores sit on the nose and the cheeks. The finger joints carry creases, and faint veins show on the inner wrists and the feet. A soft sheen at grazing angles stands in for the fine vellus hair on the face and the limbs. These patterns are worked out in the rest pose and turn with the skin, so they stay in place when the body changes shape or moves.

The shadow maps follow the pose. The map for the head moves with the head, and the map for the body covers every joint. Shadows from parts further away, such as a hand in front of the body, get softer with the distance.

The previous method with a curvature lookup table is still available. Add `?sss=lut` to the address of the viewer to compare the two methods.

## Build notes

- `build/build_body.py` subdivides the head three times and the rest of the body twice, and it joins the two densities without cracks. It then adds the relief of `build/relief.py`, the collarbones and the biceps, as a displacement along the normals. `geo_cur.npz` keeps this displacement as `detail`, and the fit of the corrective shapes works on the body without it, because the simulated body has no relief. Skinning carries the relief along with the bones.
- `build/character.py` sets the default build below the neck: low weight and muscle, and small MakeHuman targets for the chest, the hips and the limbs. The change fades out over the neck, so the head and the fit of the hair stay the same. The file also defines the ends of each body slider as MakeHuman targets. The low ends share the fat targets with the default build, and their weights keep every target within its range when all the low ends add up.
- `build/build_morphs.py` bakes each slider end as a morph target on the final mesh. Catmull-Clark subdivision is linear in the vertex positions, so the script subdivides the change of the base mesh directly. It also binds the wearables to the body. The viewer stores the targets in steps of 0.01 mm, which keeps the shading smooth at the ends of the sliders.
- `build/rig.py` builds the skeleton from the MakeHuman default skeleton and its weights, both under CC0. The viewer keeps 84 of the 163 bones: the spine, the neck and the head, the limbs with their twist bones, the fingers, and one bone for each toe. The weights of the other bones go to the bones that take their place. The weights go through the same subdivision as the body, and each vertex keeps its eight largest weights. Below the collarbones up to eight bones overlap, and a cut to four weights left creases there. The script also bakes how far each joint moves at each slider end.
- Every bone rests with the world axes at its head. A pose gives each bone one rotation relative to its parent and gives the root an offset. `build/posing.py` holds the forward kinematics, the skinning, a two-bone IK solver and the grounding of the soles on the floor. `build/poselib.py` turns controls in body terms, such as a spine bend or an elbow angle, into bone rotations. Its function `girdle_for` holds the rule of the shoulder girdle, and the arm controls apply it by themselves.
- `build/poses.py` builds the ten poses of the library, and `build/anims.py` builds the seven clips. `build/bvh.py` reads BVH files, `build/import_poses.py` puts the community poses on the rig, and `build/poses_mh.py` lists the chosen poses with their labels and groups. The pose packs unpack into `ext/poses` next to the build folder. The walk and the run move the feet along paths on the floor, and IK places the legs. `build/export_motion.py` samples every clip at 30 frames per second.
- `build/sim.py` holds the soft-tissue simulation. It uses projective dynamics with Anderson acceleration. In each step every tetrahedron finds its nearest rotation and its nearest shape with the rest volume, and one sparse solve joins these targets. A key pose takes between 5 seconds and 2 minutes.
- `build/correctives.py` defines the key poses and the drivers. Its `train` step runs the simulation for every key pose. Its `build` step fits each shape on the final mesh with the final skin weights, so that skinning the rest shape plus the shape reaches the subdivided simulation. Where the blend of bones nearly cancels, a smoothness term keeps the shape smooth. The stored normal change makes the skinned normal match the corrected surface at the key pose, and an engine applies it like any morph target normal. The `build` step also writes the drivers to `corr_cur.json`, which this folder holds as `correctives.json`. The key straight up at the shoulder uses a lighter smoothness, because along the side of the chest the skin weights change fast from the torso to the arm.
- `web/encode.mjs` puts the vertices that the corrective shapes move at the front of the vertex buffer, so the viewer uploads one short range of vertices per frame.
- In `web/app.js` the skin shader softens the red light with a smoothed normal. The viewer turns this smoothed normal together with the corrected normal, and where a shape turns the normal far, the smoothed normal moves over to the corrected normal. This keeps a red and green fringe out of the armpit of a raised arm.
- `build/regions.py` sets the skin colour regions. It also builds the masks for pores, veins and vellus hair.
- `build/run_bake.py` bakes the body occlusion and the contact shade of each wearable. `build/bake_open.py` bakes the thickness that the light-through effect uses. It counts only the paths that come out into open space, so the mouth and the eye sockets stay solid.
- The build works at MakeHuman scale, where the figure is 1.45 m tall. `build/export_models.py` scales the exported files by 1.1241 to reach 1.63 m.
