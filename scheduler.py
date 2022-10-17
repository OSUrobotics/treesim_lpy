from abc import ABC, abstractmethod
from numbers import Number
import numpy as np

class Scheduler(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Number:
        ...

class ConstantScheduler(Scheduler):
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class LinearScheduler(Scheduler):
    """
    Returns a linearly interpolated value between t=0 and t=period between start and end.
    Will return the start value if t<0 and the end value if t>period if the appropriate bounds are set.
    """

    def __init__(self, start, end, period, bound_start=False, bound_end=False):
        self.start = start
        self.end = end
        self.period = period
        self.bound_start = bound_start
        self.bound_end = bound_end

    def __call__(self, age):
        if age <= 0 and self.bound_start:
            return self.start
        elif age >= self.period and self.bound_end:
            return self.end
        else:
            return self.start + (self.end - self.start) * age / self.period


class ExponentialScheduler(Scheduler):
    """
    Returns an interpolated value according to the exponential decay function end + (start - end) * e^(-kt).
    """

    def __init__(self, start, end, k, as_half_life=False):
        if as_half_life:
            k = np.log(2) / k

        self.start = start
        self.end = end
        self.k = k

    def __call__(self, age):
        return self.end + (self.start - self.end) * np.exp(-self.k * age)

