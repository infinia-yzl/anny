# Baseline head

This folder holds the head from the Portrait of a Boy viewer, version 3. The boy has medium tousled hair with a straight fringe.

## Files

- `portrait_viewer.html` is the interactive viewer. Open it in a desktop browser with an internet connection, because the page loads three.js and two fonts from public CDNs.
- `baseline_head.glb` holds the head and both eyes as meshes in metres with Y up. The skin colour is stored as linear vertex colour, and the head mesh has 279,227 vertices.
- `baseline_hair_strands.zip` holds one OBJ file with the hair, brow and lash strands as polylines. The hair has 56,000 strands. Blender imports each polyline as a chain of edges, and Object > Convert > Curve turns them into curves.
- `source.zip` holds the build scripts for the head, the groom and the viewer page. The scripts still point at the folders of the cloud workspace where they were written.
- `makehuman_data.zip` holds the MakeHuman base mesh and morph targets that the build uses. The MakeHuman project publishes these assets under the CC0 licence.

## Build order

1. `build/build_geo2.py` morphs the MakeHuman base mesh, subdivides it and fits the eyes and lips.
2. `build/run_bake.py` bakes occlusion, thickness and curvature.
3. `build/hair2.py` grooms the hair, and `build/brows_lashes.py` grows the brows and lashes.
4. `build/export_dev.py` and `build/pack_data.py` prepare the vertex and strand data.
5. `web/encode.mjs` compresses the data, and `web/build_page.py` writes the viewer page.

The face shape settings live in `build/character.py`. The haircut settings (hairline, perimeter cut, lengths and texture) live in `build/hair2.py`.
