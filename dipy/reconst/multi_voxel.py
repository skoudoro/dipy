"""Tools to easily make multi voxel models"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from dipy.core.ndindex import ndindex
from dipy.reconst.quick_squash import quick_squash as _squash
from dipy.reconst.base import ReconstFit
from multiprocessing import Queue, cpu_count, Process, sharedctypes, Pool, Manager
import warnings


def parallel_fit_worker(model, input_queue, output_queue, *args, **kwargs):
    for data in iter(input_queue.get, 'STOP'):
        idx, value = data
        result = (idx, model.fit(value, kwargs))
        output_queue.put(result)
        del value


def parallel_voxel_fit(single_voxel_fit):
    """
    Wraps single_voxel_fit method to turn a model into a parallel multi voxel model.
    Use this decorator on the fit method of your model to take advantage of the
    MultiVoxelFit.

    Parameters
    -----------
    single_voxel_fit : callable
        Should have a signature like: model [self when a model method is being
        wrapped], data [single voxel data].

    Returns
    --------
    multi_voxel_fit_function : callable

    Examples:
    ---------
    >>> import numpy as np
    >>> from dipy.reconst.base import ReconstModel, ReconstFit
    >>> class Model(ReconstModel):
    ...     @parallel_voxel_fit
    ...     def fit(self, single_voxel_data):
    ...         return ReconstFit(self, single_voxel_data.sum())
    >>> model = Model(None)
    >>> data = np.random.random((2, 3, 4, 5))
    >>> fit = model.fit(data)
    >>> np.allclose(fit.data, data.sum(-1))
    True
    """

    def new_fit(model, data, mask=None, *args, **kwargs):
        """Fit method in parallel for every voxel in data """
        if data.ndim == 1:
            return single_voxel_fit(model, data)
        if mask is None:
            mask = np.ones(data.shape[:-1], bool)
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        print("parallel_voxel_fit")
        # Get number of processes
        nb_processes = int(kwargs['nb_processes']) if 'nb_processes' in kwargs else cpu_count()
        nb_processes = cpu_count() if nb_processes < 1 else nb_processes

        # Create queues
        task_queues = [Queue() for _ in range(nb_processes)]
        done_queue = Queue()

        # Get non null index from mask
        indexes = np.argwhere(mask)
        # convert indexes to tuple
        indexes = [(tuple(v), data[tuple(v)]) for v in indexes]
        # create chunks
        chunks_spacing = np.linspace(0, len(indexes), num=nb_processes + 1)
        chunks = [(indexes[int(chunks_spacing[i - 1]): int(chunks_spacing[i])]) for i in range(1, len(chunks_spacing))]
        # Create queue = Fill task queue with indexes
        for i, c in enumerate(chunks):
            [task_queues[i].put(val) for val in c]

        # Add to queue stop processes
        # for _ in range(nb_processes):
        #     task_queue.put('STOP')

        # Create queues
        # task_queue = Queue()
        # done_queue = Queue()

        print("adding stop")
        # Add to queue stop processes
        for q in task_queues:
            q.put('STOP')

        # Start worker processes
        print("Start worker processes")
        for q in task_queues:
            Process(target=parallel_fit_worker,
                    args=(model, q, done_queue, args),
                    kwargs=kwargs).start()



        print("create output array")
        # create output array
        fit_array = np.empty(data.shape[:-1], dtype=object)
        # fill output array with results
        for _ in range(len(indexes)):
            idx, val = done_queue.get()
            fit_array[idx] = val

        print("END")
        return MultiVoxelFit(model, fit_array, mask)
    return new_fit


def multi_voxel_fit(single_voxel_fit):
    """Method decorator to turn a single voxel model fit
    definition into a multi voxel model fit definition
    """
    def new_fit(self, data, mask=None):
        """Fit method for every voxel in data"""
        # If only one voxel just return a normal fit
        if data.ndim == 1:
            return single_voxel_fit(self, data)

        # Make a mask if mask is None
        if mask is None:
            shape = data.shape[:-1]
            strides = (0,) * len(shape)
            mask = as_strided(np.array(True), shape=shape, strides=strides)
        # Check the shape of the mask if mask is not None
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        # Fit data where mask is True
        fit_array = np.empty(data.shape[:-1], dtype=object)
        for ijk in ndindex(data.shape[:-1]):
            if mask[ijk]:
                fit_array[ijk] = single_voxel_fit(self, data[ijk])
        return MultiVoxelFit(self, fit_array, mask)
    return new_fit


class MultiVoxelFit(ReconstFit):
    """Holds an array of fits and allows access to their attributes and
    methods"""
    def __init__(self, model, fit_array, mask):
        self.model = model
        self.fit_array = fit_array
        self.mask = mask

    @property
    def shape(self):
        return self.fit_array.shape

    def __getattr__(self, attr):
        result = CallableArray(self.fit_array.shape, dtype=object)
        for ijk in ndindex(result.shape):
            if self.mask[ijk]:
                result[ijk] = getattr(self.fit_array[ijk], attr)
        return _squash(result, self.mask)

    def __getitem__(self, index):
        item = self.fit_array[index]
        if isinstance(item, np.ndarray):
            return MultiVoxelFit(self.model, item, self.mask[index])
        else:
            return item

    def predict(self, *args, **kwargs):
        """
        Predict for the multi-voxel object using each single-object's
        prediction API, with S0 provided from an array.
        """
        S0 = kwargs.get('S0', np.ones(self.fit_array.shape))
        idx = ndindex(self.fit_array.shape)
        ijk = next(idx)

        def gimme_S0(S0, ijk):
            if isinstance(S0, np.ndarray):
                return S0[ijk]
            else:
                return S0

        kwargs['S0'] = gimme_S0(S0, ijk)
        # If we have a mask, we might have some Nones up front, skip those:
        while self.fit_array[ijk] is None:
            ijk = next(idx)

        if not hasattr(self.fit_array[ijk], 'predict'):
            msg = "This model does not have prediction implemented yet"
            raise NotImplementedError(msg)

        first_pred = self.fit_array[ijk].predict(*args, **kwargs)
        result = np.zeros(self.fit_array.shape + (first_pred.shape[-1],))
        result[ijk] = first_pred
        for ijk in idx:
            kwargs['S0'] = gimme_S0(S0, ijk)
            # If it's masked, we predict a 0:
            if self.fit_array[ijk] is None:
                result[ijk] *= 0
            else:
                result[ijk] = self.fit_array[ijk].predict(*args, **kwargs)

        return result


class CallableArray(np.ndarray):
    """An array which can be called like a function"""
    def __call__(self, *args, **kwargs):
        result = np.empty(self.shape, dtype=object)
        for ijk in ndindex(self.shape):
            item = self[ijk]
            if item is not None:
                result[ijk] = item(*args, **kwargs)
        return _squash(result)
