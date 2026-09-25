# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0

# %% [markdown]
# # Parameterizing shapes with Anny
#

# %% [markdown]
# ### Instantiate the model
# Anny is shipped as a Python package that can be easily installed (see the README for details).

# %%
from IPython.display import Markdown, display
import torch
import roma  # A PyTorch library useful to deal with space transformations.
import anny  # The main library for the Anny model.
import trimesh  # For 3D mesh visualization.
import matplotlib.pyplot as plt
import anny.shape_distribution
import anny.anthropometry


# Instantiate the model, with all shape parameters available.
# Remark: the first instantiation may take a while. Latter calls will be faster thanks to caching.
anny_model = anny.Anny(phenotypes="all", local_changes="default")
# Use 32bit floating point precision on the CPU for this demo.
dtype = torch.float32
device = torch.device("cpu")
anny_model = anny_model.to(device=device, dtype=dtype)

# A simple transform to get a better view angle in 3D mesh visualizations.
trimesh_scene_transform = (
    roma.Rigid(
        linear=roma.euler_to_rotmat("x", [-90.0], degrees=True), translation=None
    )
    .to_homogeneous()
    .cpu()
    .numpy()
)

# %% [markdown]
# ### Template mesh
# This is the template mesh of Anny:

# %%
display(
    Markdown(
        f"{anny_model.template_vertices.shape[0]} vertices -- "
        f"{anny_model.faces.shape[0]} faces composed of "
        f"{anny_model.faces.shape[1]} vertices each."
    )
)
trimesh.Trimesh(
    vertices=anny_model.template_vertices.cpu().numpy(),
    faces=anny_model.faces.cpu().numpy(),
).apply_transform(trimesh_scene_transform).show()

# %% [markdown]
# ## Shape parameterization
#
# Anny can model a diversity of morphologies.
# We follow MakeHuman terminology, and parameterize diversity of body shapes using a set
# of *phenotype* parameters, typically between 0 and 1.
#
# *Note:* the values *african*, *caucasian* and *asian* parameters are normalized so that they sum to 1.
#
# **Word of caution regarding phenotypes:**
# *Phenotypes are based on preconceptions of artists regarding particular human traits.
# As a result, they encode by design stereotypes of MakeHuman artists, and one should not
# expect phenotype parameters to faithfully encode identity-related characteristics, such
# as gender, age or ethnicity.*

# %%
# List phenotype parameters
Markdown(
    "**List of phenotype parameters**: "
    + ", ".join([f"{label}" for label in anny_model.phenotype_labels])
)

# %% [markdown]
# ### Example
#
# Here we show an example of how the **age** parameter influences the resulting mesh.

# %%
batch_size = 5  # We can process multiple bodies at once in a batch.

phenotype_kwargs = {
    key: torch.full((batch_size,), fill_value=0.5, dtype=dtype, device=device)
    for key in anny_model.phenotype_labels
}
phenotype_kwargs["age"] = torch.linspace(
    0.0, 1.0, batch_size, dtype=dtype, device=device
)  # Example: vary the age parameter across the batch.
output = anny_model(phenotype_kwargs=phenotype_kwargs)

scene = trimesh.Scene()
for i in range(batch_size):
    # Create a mesh for each body in the batch.
    mesh = trimesh.Trimesh(
        vertices=output["vertices"][i].squeeze().cpu().numpy(),
        faces=anny_model.faces.cpu().numpy(),
    )
    transform = (
        roma.Rigid(
            linear=None,
            translation=torch.tensor([i * 1.0, 0.0, 0.0], dtype=dtype, device=device),
        )
        .to_homogeneous()
        .cpu()
        .numpy()
    )
    scene.add_geometry(mesh, transform=transform)

scene.apply_transform(
    trimesh_scene_transform
)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the bodies in it.

# %% [markdown]
# ## Local changes
# Additionnally one specify some more local morphological changes.
# Local change parameters values are typically expected to be chosen between -1 and 1,
# but one can use values outside this range to extrapolate changes even further.
#
# *Note: it is easy to produce unrealistic meshes when using significant local changes.*

# %%
display(
    Markdown(
        "**List of local changes parameters:** "
        + ", ".join(anny_model.local_change_labels)
    )
)

# %% [markdown]
# #### Local change example
#
# We show here the effect of the *stomach-pregnant-incr* local change parameter, as an example.

# %%
batch_size = 3  # We can process multiple faces at once in a batch.
pose_parameters = roma.Rigid.identity(
    dim=3, batch_shape=(batch_size, anny_model.bone_count)
).to_homogeneous()
phenotype_kwargs = {
    key: torch.full((batch_size,), fill_value=0.5)
    for key in anny_model.phenotype_labels
}
# In this example, we start from a stereotypical young adult woman mesh
phenotype_kwargs["age"].fill_(0.67)
phenotype_kwargs["gender"].fill_(1.0)

local_changes = {
    "stomach-pregnant-incr": torch.linspace(0, 1.0, batch_size)
}  # Example: vary the upperarm fat increment across the batch.
output = anny_model(
    phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=local_changes
)

scene = trimesh.Scene()
for i in range(batch_size):
    # Create a mesh for each face in the batch.
    mesh = trimesh.Trimesh(
        vertices=output["vertices"][i].squeeze().cpu().numpy(),
        faces=anny_model.faces.cpu().numpy(),
    )
    transform = (
        roma.Rigid(linear=None, translation=torch.tensor([i * 1.0, 0.0, 0.0]))
        .to_homogeneous()
        .cpu()
        .numpy()
    )
    scene.add_geometry(mesh, transform=transform)
scene.apply_transform(
    trimesh_scene_transform
)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the faces in

# %% [markdown]
# ## Face and head shapes
#
# `face_shapes` loads 103 named, symmetric shapes of the head and the face, built from the MakeHuman
# face targets: head archetypes (oval, round, square, ...), the size and the position of the eyes,
# the nose, the mouth, the chin, the cheeks and the ears. A value of +1 applies the positive target
# and -1 the negative one; the head archetypes and `chin-triangle` run from 0 to 1.
#
# Each group of shapes scales with the size of the matching part of the head, measured on
# craniofacial landmarks, so the offsets that MakeHuman authored on an adult stay in proportion on a
# child (`scale_face_shapes=False` applies them unchanged).

# %%
face_shape_model = anny.Anny(face_shapes="all").to(device=device, dtype=dtype)
display(
    Markdown(
        "**Face-shape parameters:** " + ", ".join(face_shape_model.face_shape_labels)
    )
)

batch_size = 3
phenotype_kwargs = {
    "age": torch.full((batch_size,), 0.8),
    "gender": torch.full((batch_size,), 1.0),
}
face_shape_kwargs = {
    "head-round": torch.tensor([0.0, 0.8, 0.0]),
    "nose-scale-vert": torch.tensor([0.0, 0.0, 1.0]),
    "chin-width": torch.tensor([0.0, 0.0, 1.0]),
}
output = face_shape_model(
    phenotype_kwargs=phenotype_kwargs, face_shape_kwargs=face_shape_kwargs
)
# the scale of each group of shapes, for a child and an adult
display(
    Markdown(
        "**Scale groups:** "
        + str(
            dict(
                zip(
                    anny.models.face_shapes.SCALE_GROUPS,
                    face_shape_model.face_shape_scales(
                        {"age": torch.tensor([0.2, 0.8])}
                    ).T.tolist(),
                )
            )
        )
    )
)

# %% [markdown]
# #### Calibrated random faces
#
# `anny.faces.distribution.FaceShapeDistribution` draws face values for given phenotypes. Its
# Gaussian comes from fits of anny to the identity space of ICT-FaceKit, calibrated so that anny's
# head and face measurements match the ANSUR II survey (adults), the 3D Facial Norms database
# (3 to 39 years) and the CDC growth charts (head circumference below 3 years). The sources and
# their licences are in `src/anny/data/faces/SOURCES.md`.

# %%
import anny.faces.distribution
from anny.faces.measurements import CraniofacialMeasurements

faces = anny.faces.distribution.FaceShapeDistribution(face_shape_model)
batch_size = 4
phenotype_kwargs = {
    "age": torch.full((batch_size,), 0.8),
    "gender": torch.tensor([0.0, 0.0, 1.0, 1.0]),
}
face_values = faces.sample(phenotype_kwargs, generator=torch.Generator().manual_seed(0))
output = face_shape_model(
    phenotype_kwargs=phenotype_kwargs, face_shape_kwargs=face_values
)

# head and face measurements (mm), as ANSUR II and 3D Facial Norms define them
measure = CraniofacialMeasurements(face_shape_model)
values = measure(output)
display(
    Markdown(
        "**Head length (mm):** "
        + ", ".join(f"{v:.0f}" for v in values["headlength"].tolist())
        + "; **nasal width (mm):** "
        + ", ".join(f"{v:.1f}" for v in values["nasalwidth"].tolist())
    )
)

scene = trimesh.Scene()
for i in range(batch_size):
    mesh = trimesh.Trimesh(
        vertices=output["vertices"][i].squeeze().cpu().numpy(),
        faces=face_shape_model.faces.cpu().numpy(),
    )
    transform = (
        roma.Rigid(linear=None, translation=torch.tensor([i * 0.3, 0.0, 0.0]))
        .to_homogeneous()
        .cpu()
        .numpy()
    )
    scene.add_geometry(mesh, transform=transform)
scene.apply_transform(trimesh_scene_transform)
scene.show()

# %% [markdown]
# ## Facial actions
#
# Full-body and head models can optionally expose facial actions.
# The dictionary form is convenient for sparse edits, while tensor input is convenient for batched optimization.

# %%
face_model = anny.Anny(rig="anny", topology="head", facial_actions="all").to(
    device=device, dtype=dtype
)

# Using with dict unit -> tensor input
facial_actions = {
    "jawOpen": torch.tensor([0.0, 0.6], dtype=dtype, device=device),
    "mouthSmileLeft": torch.tensor([0.0, 0.4], dtype=dtype, device=device),
    "mouthSmileRight": torch.tensor([0.0, 0.4], dtype=dtype, device=device),
}
dict_output = face_model(facial_actions=facial_actions)

# Using with batched Bx52 tensor input
batch_size = 5
values = torch.zeros(
    (batch_size, len(face_model.facial_action_labels)), dtype=dtype, device=device
)
values[:, face_model.facial_action_labels.index("jawOpen")] = torch.linspace(
    0.0, 1.0, batch_size, dtype=dtype, device=device
)
values[:, face_model.facial_action_labels.index("mouthSmileLeft")] = torch.linspace(
    0.0, 1.0, batch_size, dtype=dtype, device=device
)
values[:, face_model.facial_action_labels.index("mouthSmileRight")] = torch.linspace(
    0.0, 1.0, batch_size, dtype=dtype, device=device
)
output = face_model(facial_actions=values)

display(Markdown("**Face unit labels:** " + ", ".join(face_model.facial_action_labels)))

scene = trimesh.Scene()
for i in range(batch_size):
    # Create a mesh for each face in the batch.
    mesh = trimesh.Trimesh(
        vertices=output["vertices"][i].squeeze().cpu().numpy(),
        faces=face_model.faces.cpu().numpy(),
    )
    transform = (
        roma.Rigid(
            linear=roma.euler_to_rotmat("x", [90.0], degrees=True),
            translation=torch.tensor([i * 0.2, 0.0, 0.0]),
        )
        .to_homogeneous()
        .cpu()
        .numpy()
    )
    scene.add_geometry(mesh, transform=transform)
scene.apply_transform(
    trimesh_scene_transform
)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the faces in

# %% [markdown]
# ## Phenotype distribution

# %%
phenotype_distribution = anny.shape_distribution.SimpleShapeDistribution(
    anny_model,
    morphological_age_distribution=torch.distributions.Uniform(low=0.0, high=60.0),
)

real_age, phenotype_kwargs = phenotype_distribution.sample(batch_size=200)
output = anny_model(phenotype_kwargs=phenotype_kwargs)

scene = trimesh.Scene()
i = -1
for u in range(4):
    for v in range(5):
        i += 1
        assert i < output["vertices"].shape[0], "Batch size is too small for the grid."
        # Create a mesh for each face in the batch.
        mesh = trimesh.Trimesh(
            vertices=output["vertices"][i].squeeze().cpu().numpy(),
            faces=anny_model.faces.cpu().numpy(),
        )
        transform = (
            roma.Rigid(
                linear=None,
                translation=torch.tensor(
                    [v * 1.0, u * 1.0, 0.0], dtype=dtype, device=device
                ),
            )
            .to_homogeneous()
            .cpu()
            .numpy()
        )
        scene.add_geometry(mesh, transform=transform)
scene.apply_transform(
    trimesh_scene_transform
)  # Rotate the scene to have a better view.
scene.show()  # This will open a window to visualize the scene with all the faces in

# %% [markdown]
# ### Body measures
#
# We additionally provide a class to estimate some anthropometric measurements, assuming
# a body buoyancy of .98 in water.

# %%
real_age, phenotype_kwargs = phenotype_distribution.sample(batch_size=1000)
output = anny_model(phenotype_kwargs=phenotype_kwargs)

measurements = anny.anthropometry.Anthropometry(anny_model)
measures = measurements(output["rest_vertices"])

fig, axes = plt.subplots(1, 3, squeeze=True, figsize=(10, 5))
axes[0].scatter(real_age.cpu().numpy(), measures["height"].cpu().numpy())
axes[0].set_xlabel("Morphological age (year)")
axes[0].set_ylabel("Height (m)")
axes[1].scatter(real_age.cpu().numpy(), measures["waist_circumference"].cpu().numpy())
axes[1].set_xlabel("Morphological age (year)")
axes[1].set_ylabel("Waist circumference (m)")
axes[2].scatter(real_age.cpu().numpy(), measures["bmi"].cpu().numpy())
axes[2].set_xlabel("Morphological age (year)")
axes[2].set_ylabel("Body Mass Index estimate")
fig.tight_layout()
plt.show()
