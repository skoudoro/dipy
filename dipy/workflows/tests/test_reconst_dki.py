from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from dipy.core.gradients import generate_bvecs
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.io.peaks import load_pam
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.workflows.reconst import ReconstDkiFlow


def test_reconst_dki():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_101D")
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0])
        mask_path = Path(out_dir) / "tmp_mask.nii.gz"
        save_nifti(mask_path, mask.astype(np.uint8), affine)

        dki_flow = ReconstDkiFlow()

        args = [data_path, bval_path, bvec_path, mask_path]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning,
            )
            dki_flow.run(*args, out_dir=out_dir, extract_pam_values=True)

        fa_path = dki_flow.last_generated_outputs["out_fa"]
        fa_data = load_nifti_data(fa_path)
        assert_equal(fa_data.shape, volume.shape[:-1])

        tensor_path = dki_flow.last_generated_outputs["out_dt_tensor"]
        tensor_data = load_nifti_data(tensor_path)
        assert_equal(tensor_data.shape[-1], 6)
        assert_equal(tensor_data.shape[:-1], volume.shape[:-1])

        ga_path = dki_flow.last_generated_outputs["out_ga"]
        ga_data = load_nifti_data(ga_path)
        assert_equal(ga_data.shape, volume.shape[:-1])

        rgb_path = dki_flow.last_generated_outputs["out_rgb"]
        rgb_data = load_nifti_data(rgb_path)
        assert_equal(rgb_data.shape[-1], 3)
        assert_equal(rgb_data.shape[:-1], volume.shape[:-1])

        md_path = dki_flow.last_generated_outputs["out_md"]
        md_data = load_nifti_data(md_path)
        assert_equal(md_data.shape, volume.shape[:-1])

        ad_path = dki_flow.last_generated_outputs["out_ad"]
        ad_data = load_nifti_data(ad_path)
        assert_equal(ad_data.shape, volume.shape[:-1])

        rd_path = dki_flow.last_generated_outputs["out_rd"]
        rd_data = load_nifti_data(rd_path)
        assert_equal(rd_data.shape, volume.shape[:-1])

        mk_path = dki_flow.last_generated_outputs["out_mk"]
        mk_data = load_nifti_data(mk_path)
        assert_equal(mk_data.shape, volume.shape[:-1])

        ak_path = dki_flow.last_generated_outputs["out_ak"]
        ak_data = load_nifti_data(ak_path)
        assert_equal(ak_data.shape, volume.shape[:-1])

        rk_path = dki_flow.last_generated_outputs["out_rk"]
        rk_data = load_nifti_data(rk_path)
        assert_equal(rk_data.shape, volume.shape[:-1])

        kt_path = dki_flow.last_generated_outputs["out_dk_tensor"]
        kt_data = load_nifti_data(kt_path)
        assert_equal(kt_data.shape[-1], 15)
        assert_equal(kt_data.shape[:-1], volume.shape[:-1])

        mode_path = dki_flow.last_generated_outputs["out_mode"]
        mode_data = load_nifti_data(mode_path)
        assert_equal(mode_data.shape, volume.shape[:-1])

        evecs_path = dki_flow.last_generated_outputs["out_evec"]
        evecs_data = load_nifti_data(evecs_path)
        assert_equal(evecs_data.shape[-2:], (3, 3))
        assert_equal(evecs_data.shape[:-2], volume.shape[:-1])

        evals_path = dki_flow.last_generated_outputs["out_eval"]
        evals_data = load_nifti_data(evals_path)
        assert_equal(evals_data.shape[-1], 3)
        assert_equal(evals_data.shape[:-1], volume.shape[:-1])

        peaks_dir_path = dki_flow.last_generated_outputs["out_peaks_dir"]
        peaks_dir_data = load_nifti_data(peaks_dir_path)
        assert_equal(peaks_dir_data.shape[-1], 9)
        assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

        peaks_idx_path = dki_flow.last_generated_outputs["out_peaks_indices"]
        peaks_idx_data = load_nifti_data(peaks_idx_path)
        assert_equal(peaks_idx_data.shape[-1], 3)
        assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

        peaks_vals_path = dki_flow.last_generated_outputs["out_peaks_values"]
        peaks_vals_data = load_nifti_data(peaks_vals_path)
        assert_equal(peaks_vals_data.shape[-1], 3)
        assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])

        pam = load_pam(dki_flow.last_generated_outputs["out_pam"])
        assert_allclose(pam.peak_dirs.reshape(peaks_dir_data.shape), peaks_dir_data)
        assert_allclose(pam.peak_values, peaks_vals_data)
        assert_allclose(pam.peak_indices, peaks_idx_data)

        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        bvals[0] = 5.0
        bvecs = generate_bvecs(len(bvals))

        tmp_bval_path = Path(out_dir) / "tmp.bval"
        tmp_bvec_path = Path(out_dir) / "tmp.bvec"
        np.savetxt(tmp_bval_path, bvals)
        np.savetxt(tmp_bvec_path, bvecs.T)
        dki_flow._force_overwrite = True
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning,
            )
            dki_flow.run(
                data_path,
                tmp_bval_path,
                tmp_bvec_path,
                mask_path,
                out_dir=out_dir,
                b0_threshold=0,
            )
