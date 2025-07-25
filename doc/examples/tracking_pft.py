"""
===============================
Particle Filtering Tractography
===============================
Particle Filtering Tractography (PFT) :footcite:p:`Girard2014` uses tissue
partial volume estimation (PVE) to reconstruct trajectories connecting the gray
matter, and not incorrectly stopping in the white matter or in the corticospinal
fluid. It relies on a stopping criterion that identifies the tissue where the
streamline stopped. If the streamline correctly stopped in the gray matter, the
trajectory is kept. If the streamline incorrectly stopped in the white matter
or in the corticospinal fluid, PFT uses anatomical information to find an
alternative streamline segment to extend the trajectory. When this segment is
found, the tractography continues until the streamline correctly stops in the
gray matter.

PFT finds an alternative streamline segment whenever the stopping criterion
returns a position classified as 'INVALIDPOINT'.

This example is an extension of
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_probabilistic.py` and
:ref:`sphx_glr_examples_built_fiber_tracking_tracking_stopping_criterion.py`
examples. We begin by loading the data, fitting a Constrained Spherical
Deconvolution (CSD) reconstruction model, and defining the seeds.
"""

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking import utils
from dipy.tracking.stopping_criterion import CmcStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import pft_tracking, probabilistic_tracking
from dipy.viz import actor, colormap, has_fury, window

# Enables/disables interactive visualization
interactive = False

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(name="stanford_hardi")
label_fname = get_fnames(name="stanford_labels")
f_pve_csf, f_pve_gm, f_pve_wm = get_fnames(name="stanford_pve_maps")

data, affine, hardi_img = load_nifti(hardi_fname, return_img=True)
labels = load_nifti_data(label_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

pve_csf_data = load_nifti_data(f_pve_csf)
pve_gm_data = load_nifti_data(f_pve_gm)
pve_wm_data, _, voxel_size = load_nifti(f_pve_wm, return_voxsize=True)

shape = labels.shape

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)
csd_fit = csd_model.fit(data, mask=pve_wm_data)

seed_mask = labels == 2
seed_mask[pve_wm_data < 0.5] = 0
seeds = utils.seeds_from_mask(seed_mask, affine, density=2)

###############################################################################
# CMC/ACT Stopping Criterion
# ==========================
# Continuous map criterion (CMC) :footcite:p:`Girard2014` and
# Anatomically-constrained tractography (ACT) :footcite:p:`Smith2012` both use
# PVEs information from anatomical images to determine when the tractography
# stops. Both stopping criterion use a trilinear interpolation at the tracking
# position. CMC stopping criterion uses a probability derived from the PVE maps
# to determine if the streamline reaches a 'valid' or 'invalid' region. ACT uses
# a fixed threshold on the PVE maps. Both stopping criterion can be used in
# conjunction with PFT. In this example, we used CMC.

voxel_size = np.average(voxel_size[1:4])
step_size = 0.2

cmc_criterion = CmcStoppingCriterion.from_pve(
    pve_wm_data,
    pve_gm_data,
    pve_csf_data,
    step_size=step_size,
    average_voxel_size=voxel_size,
)

###############################################################################
# Particle Filtering Tractography
# ===============================
# `pft_back_tracking_dist` is the distance in mm to backtrack when the
# tractography incorrectly stops in the WM or CSF. `pft_front_tracking_dist`
# is the distance in mm to track after the stopping event using PFT.
#
# The `particle_count` parameter is the number of samples used in the particle
# filtering algorithm.
#
# `min_wm_pve_before_stopping` controls when the tracking can stop in the GM.
# The tractography must reaches a position where WM_pve >=
# `min_wm_pve_before_stopping` before stopping in the GM. As long as the
# condition is not reached and WM_pve > 0, the tractography will continue.
# It is particularlyusefull to set this parameter > 0.5 when seeding
# tractography at the WM-GM interface or in the sub-cortical GM. It allows
# streamlines to exit the seeding voxels until they reach the deep white
# matter where WM_pve > `min_wm_pve_before_stopping`.
pft_streamline_gen = pft_tracking(
    seeds,
    cmc_criterion,
    affine,
    max_cross=1,
    step_size=step_size,
    pft_back_tracking_dist=2,
    pft_front_tracking_dist=1,
    particle_count=15,
    return_all=False,
    min_wm_pve_before_stopping=1,
    max_angle=20.0,
    sphere=default_sphere,
    sh=csd_fit.shm_coeff,
)
streamlines = Streamlines(pft_streamline_gen)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_tractogram(sft, "tractogram_pft.trx")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(scene=scene, out_path="tractogram_pft.png", size=(800, 800))
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using particle filtering tractography

# Local Probabilistic Tractography
prob_streamline_generator = probabilistic_tracking(
    seeds,
    cmc_criterion,
    affine,
    step_size=step_size,
    return_all=False,
    sh=csd_fit.shm_coeff,
    max_angle=20.0,
    sphere=default_sphere,
)
streamlines = Streamlines(prob_streamline_generator)
sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_tractogram(sft, "tractogram_probabilistic_cmc.trx")

if has_fury:
    scene = window.Scene()
    scene.add(actor.line(streamlines, colors=colormap.line_colors(streamlines)))
    window.record(
        scene=scene, out_path="tractogram_probabilistic_cmc.png", size=(800, 800)
    )
    if interactive:
        window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Corpus Callosum using probabilistic tractography
#
#
#
# References
# ----------
#
# .. footbibliography::
#
