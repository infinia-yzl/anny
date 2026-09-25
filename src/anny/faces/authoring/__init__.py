# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
"""
Authoring tools for anny's faces. Each step runs as ``python -m anny.faces.authoring.<step>``:

- ``landmarks``: places the craniofacial landmarks on the MakeHuman base mesh
  (``data/keypoints/craniofacial.pth``);
- ``sources``: downloads the data sources into ``ANNY_CACHE_DIR/faces``;
- ``fit_3d``: fits anny's face shapes to the ICT-FaceKit identity space;
- ``calibrate``: builds the face-shape distribution (``data/shape_calibration/face_prior.safetensors``);
- ``photos`` and ``benchmark``: compare anny's faces with FairFace photographs.
"""
