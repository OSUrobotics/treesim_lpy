from abc import ABC, abstractproperty
import numpy as np

class ParamSet:

    PARAMS = []

    def __init__(self, explicit_params, *args, **kwargs):
        self._explicit_params = explicit_params
        self._stored_values = {}

        arg_keys = explicit_params[:len(args)]
        self._stored_values.update(dict(zip(arg_keys, args)))
        self._stored_values.update(kwargs)

    def __getattr__(self, item):
        return self._stored_values[item]

    def __setattr__(self, key, value):
        if key not in self._explicit_params:
            return AttributeError('{} is not an explicitly stored value for class {}'.format(key, type(self).__name__))

    @property
    def args(self):
        return [self._stored_values[k] for k in self._explicit_params]

class Branch(ParamSet):

    PARAMS = ['length', 'radius', 'age', 'max_radius', 'k_res']

    @property
    def volume(self):
        return self.length * np.pi * self.radius ** 2

    @property
    def resistance(self):
        return self.k_res * self.length / (np.pi * self.radius ** 2)

    @property
    def capacity(self):
        raise NotImplementedError

class Bud(ParamSet):
    PARAMS = ['type', 'age', 'light', 'suppression']

    @property
    def max_leaves(self):
        return 5

class Leaf(ParamSet):
    PARAMS = ['light', 'sugar']

class Meristem(ParamSet):

    PARAMS = ['age', 'prog', 'base_rad']


