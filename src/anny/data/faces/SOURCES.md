# Sources of anny's face data

anny's face shapes, their scaling and their distribution rest on the sources below. Every
source is free and downloads without an account (`python -m anny.faces.authoring.sources`).
The downloaded files stay in `ANNY_CACHE_DIR/faces`; this repository keeps only the parameters
derived from them.

| Source | Use in anny | Licence and attribution |
|---|---|---|
| MakeHuman targets (`data/mpfb2/targets`) | The face-shape parameters (`face_shapes.json`) and the lip mask that places the lip landmarks | CC0 1.0, as the rest of the MakeHuman assets in anny (see `data/mpfb2/LICENSE.md`) |
| ICT Face Model Light, [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit) | The 3D identity space that anny's face shapes are fitted to, and the 10 detail shapes (`detail_shapes.safetensors`), derived from the residuals of those fits | MIT licence. Li, R., Bladin, K., Zhao, Y., et al. *Learning Formation of Physically-Based Face Attributes*. CVPR 2020. |
| [ANSUR II](https://www.openlab.psu.edu/ansur2/), 2012 Anthropometric Survey of U.S. Army Personnel | Adult head and face measurements by sex (6,068 people) | Public release. Gordon, C. C., et al. *2012 Anthropometric Survey of U.S. Army Personnel: Methods and Summary Statistics*. NATICK/TR-15/007 (2014). Measurement definitions: Hotzman, J., et al. *Measurer's Handbook*. NATICK/TR-11/017 (2011). |
| [3D Facial Norms](https://www.facebase.org/resources/human/facial_norms/) summary statistics, FaceBase | Means and SDs of 34 facial measurements by age (3 to 40) and sex | Data obtained from the FaceBase database (www.facebase.org), 3D Facial Norms dataset. Weinberg, S. M., et al. *The 3D Facial Norms Database: Part 1*. Cleft Palate-Craniofacial Journal 53(6), 2016. Funded by NIDCR (U01-DE020078). |
| [CDC growth charts](https://www.cdc.gov/growthcharts/), head circumference for age | Head size and its spread from birth to 36 months | Public domain. Kuczmarski, R. J., et al. *2000 CDC Growth Charts for the United States*. Vital Health Stat 11(246), 2002. |
| [FairFace](https://github.com/joojs/fairface) | Photographs for the benchmark of face proportions by age, sex and race | CC BY 4.0. Kärkkäinen, K., and Joo, J. *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age*. WACV 2021. |
| [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) and canonical face model | Landmarks of the photographs and of anny's renders | Apache 2.0 licence, Google. |

Deferred because they need a registration: FLAME 2023 Open (CC BY 4.0) and the individual-level
3D Facial Norms data.
