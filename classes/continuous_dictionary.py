import numpy as np

from classes.gaussian_kernel import GaussianKernel
from utils.models import code
from utils.constants import RELEVANCE_THRESHOLD, RANDOM_STATE


class ContinuousGaussianDictionary(dict):

    def __init__(self, kernel_radius: float, temporal_kernel_radius: float):
        dict.__init__(self)
        self.kernel_radius = kernel_radius
        self.temporal_kernel_radius = temporal_kernel_radius

    def __getitem__(self, key):
        position, timestamp = code(key[:-1]), key[-1]
        if len(self) == 0:
            raise KeyError("Dict is empty")
        if (*position, timestamp) in self.keys():
            return dict.__getitem__(self, (*position, timestamp))
        else:
            kernel = GaussianKernel(position, self.kernel_radius)
            temporal_kernel = GaussianKernel(
                np.array([timestamp]), self.temporal_kernel_radius
            )
            values, weights = list(), list()
            for other_key in self.keys():
                other_position, other_timestamp = code(other_key[:-1]), other_key[-1]
                values.append(list(dict.__getitem__(self, other_key)))
                weights.append(
                    (kernel.pdf(other_position) + temporal_kernel.pdf(other_timestamp))
                    / 2
                )

            values = np.array(values)
            weights = np.array(weights)
            # print(size)
            if np.sum(weights) > RELEVANCE_THRESHOLD:
                weights /= np.sum(weights)
                try:
                    shape = sorted(values.shape)
                    if len(shape) > 1:
                        new_shape = shape[-2], shape[-1]
                    else:
                        new_shape = (1, shape)
                    values = np.reshape(values, new_shape)
                except ValueError:
                    print(values[0])
                    raise ValueError("Whatever")
                weights = np.reshape(weights, (1, -1))

                return np.dot(values, weights.T).reshape((-1,))
            else:
                return RANDOM_STATE.uniform(-1, 1, size=values.shape[1])

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def predict(self, key, *args, **kwargs):
        return (
            self[key],
            None,
        )  # None is added to cope with the predict function of the pc-gym package


class ContinuousByRegionDictionary(dict):
    def __init__(self):
        dict.__init__(self)

    def __getitem__(self, key):
        region = None
        # key = code(key)
        if key in self.keys():
            return dict.__getitem__(self, key)
        for existing_key in self.keys():
            if np.all(
                (np.array(list(key)) >= np.array(list(existing_key[0])))
                & (np.array(list(key)) <= np.array(list(existing_key[1])))
            ):
                region = existing_key
                break
        if region is None:
            raise ValueError(
                "Element not in the covered region OR dict keys do not cover the whole action space: "
                + str(key)
            )
        return dict.__getitem__(self, region)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def predict(self, key, *args, **kwargs):
        return (
            self[key],
            None,
        )  # None is added to cope with the predict function of the pc-gym package
