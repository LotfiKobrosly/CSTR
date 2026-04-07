import numpy as np

from gaussian_kernel import GaussianKernel


class ContinuousGaussianDictionary(dict):

    def __init__(self, kernel_radius: float):
        dict.__init__(self)
        self.kernel_radius = kernel_radius

    def __getitem__(self, key):
        if len(self) == 0:
            raise KeyError("Dict is empty")
        if key in self.keys():
            return dict.__getitem__(self, key)
        else:
            kernel = GaussianKernel(key, self.kernel_radius)
            values, weights = list(), list()
            for other_key in self.keys():
                values.append(list(self[other_key]))
                weights.append(kernel.pdf(other_key))

            values = np.array(values)
            weights = np.array(weights)
            weights /= np.sum(weights)
            weights = np.reshape(weights, values.shape)

            return np.dot(values.T, weights)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class ContinuousByRegionDictionary(dict):
    def __getitem__(self, key):
        region = None
        for existing_key in self.keys():
            if np.all(
                (np.array(list(key)) >= np.array(list(existing_key[0])))
                & (np.array(list(key)) <= np.array(list(existing_key[1])))
            ):
                region = existing_key
                break
        if region is None:
            raise ValueError(
                "Element not in the covered region OR dict keys do not cover the whole action space"
            )
        return dict.__getitem__(self, region)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
