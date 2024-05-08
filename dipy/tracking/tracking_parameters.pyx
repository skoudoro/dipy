# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from dipy.tracking.tracker_deterministic cimport deterministic_tracker
from dipy.tracking.tracker_probabilistic cimport probabilistic_tracker
from dipy.tracking.tracker_ptt cimport parallel_transport_tracker
from dipy.tracking.utils import min_radius_curvature_from_angle

import numpy as np
cimport numpy as cnp


def generate_tracking_parameters(algo_name, *,
    int max_len=500, double step_size=0.5, double[:] voxel_size,
    double max_angle=30, double pmf_threshold=0.1, double probe_length=0.5,
    double probe_radius=0, int probe_quality=3, int probe_count=1,
    double data_support_exponent=1):

    cdef TrackingParameters params

    algo_name = algo_name.lower()

    if algo_name in ['deterministic', 'det']:
        params = TrackingParameters(max_len=max_len, step_size=step_size,
                                    voxel_size=voxel_size,
                                    pmf_threshold=pmf_threshold,
                                    max_angle=max_angle)
        params.set_tracker_c(deterministic_tracker)
        return params
    elif algo_name in ['probabilistic', 'prob']:
        params = TrackingParameters(max_len=max_len, step_size=step_size,
                                    voxel_size=voxel_size,
                                    pmf_threshold=pmf_threshold,
                                    max_angle=max_angle)
        params.set_tracker_c(probabilistic_tracker)
        return params
    elif algo_name == 'ptt':
        params = TrackingParameters(max_len=max_len, step_size=step_size,
                                    voxel_size=voxel_size,
                                    pmf_threshold=pmf_threshold,
                                    max_angle=max_angle,
                                    probe_length=probe_length,
                                    probe_radius=probe_radius,
                                    probe_quality=probe_quality,
                                    probe_count=probe_count,
                                    data_support_exponent=data_support_exponent)
        params.set_tracker_c(parallel_transport_tracker)
        return params
    #elif algo_name == 'eudx':
    #    return TrackingParameters(tracker=euDX_tracker,
    #                              max_len=max_len, step_size=step_size,
    #                              voxel_size=voxel_size)
    else:
        raise ValueError("Invalid algorithm name")



cdef class TrackingParameters:

    def __init__(self, max_len, step_size, voxel_size,
                 max_angle, pmf_threshold=None, probe_length=None,
                 probe_radius=None, probe_quality=None, probe_count=None,
                 data_support_exponent=None):
        cdef cnp.npy_intp i

        self.max_len = max_len
        self.step_size = step_size
        self.average_voxel_size = 0
        for i in range(3):
            self.voxel_size[i] = voxel_size[i]
            self.inv_voxel_size[i] = 1. / voxel_size[i]
            self.average_voxel_size += voxel_size[i] / 3

        self.max_angle = np.deg2rad(max_angle)
        self.cos_similarity = np.cos(self.max_angle)
        self.max_curvature = 1 / min_radius_curvature_from_angle(
            self.max_angle,
            self.step_size / self.average_voxel_size)

        self.sh = None
        self.ptt = None

        if pmf_threshold is not None:
            self.sh = ShTrackingParameters(pmf_threshold)

        if probe_length is not None and probe_radius is not None and probe_quality is not None and probe_count is not None and data_support_exponent is not None:
            self.ptt = ParallelTransportTrackingParameters(probe_length, probe_radius, probe_quality, probe_count, data_support_exponent)

    # def set_tracker(self, tracker):
    #     if callable(tracker):
    #         self.py_tracker = tracker
    #     else:
    #         raise TypeError("Object is not callable")

    cdef void set_tracker_c(self, func_ptr tracker) noexcept nogil:
        self.tracker = tracker


cdef class ShTrackingParameters:

    def __init__(self, pmf_threshold):
        self.pmf_threshold = pmf_threshold

cdef class ParallelTransportTrackingParameters:

    def __init__(self, double probe_length, double probe_radius,
                int probe_quality, int probe_count, double data_support_exponent):
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.data_support_exponent = data_support_exponent

        self.probe_step_size = self.probe_length / (self.probe_quality - 1)
        self.probe_normalizer = 1.0 / (self.probe_quality * self.probe_count)
        self.k_small = 0.0001

        # Adaptively set in Trekker
        self.rejection_sampling_nbr_sample = 10
        self.rejection_sampling_max_try = 100
