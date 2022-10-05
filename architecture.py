from collections import defaultdict
import numpy as np
from openalea.plantgl.all import *
from abc import ABC, abstractmethod


class Architecture:
    """
    Defines a collection of StructuralElement objects sorted into different categories.
    """

    def __init__(self):
        self._collection = defaultdict(list)

    def add_item(self, category, item, to_list=True):
        if to_list:
            self._collection[category].append(item)
        else:
            self._collection[category] = item

    def __getitem__(self, item):
        return self._collection[item]

    def __setitem__(self, key, value):
        self._collection[key] = value

    def get(self, *args, **kwargs):
        return self._collection.get(*args, **kwargs)

    @property
    def attractor_grid(self):
        pts = []
        for _, v in self._collection.items():
            if isinstance(v, list):
                pts.extend(map(lambda x: np.array(x.point_rep).astype(np.double), v))
            else:
                pts.append(np.array(v.point_rep).astype(np.double))

        return Point3Grid((1, 1, 1), pts)


class StructuralElement(ABC):

    @abstractmethod
    def bend_segment_curve(self, start, end):
        ...

    @property
    @abstractmethod
    def point_rep(self):
        ...


class WireVector(StructuralElement):

    def __init__(self, start, end, one_sided=True):
        """
        Represents a wire as a vector with an origin at start that passes through end.
        If one_sided is True, any point that is "behind" the start will be measured with respect to the start.
        """

        self.start = np.array(start)
        self.end = np.array(end)
        norm = np.linalg.norm(self.start - self.end)
        if np.linalg.norm(self.start - self.end) < 1e-6:
            raise ValueError('Start and end for the wire are too close!')
        self.ray = (self.end - self.start) / norm
        self.one_sided = one_sided

        # TEMP - REDO LATER
        self.num_branch = 0

    def bend_segment_curve(self, branch_start, branch_end):
        """
        Given a branch segment with the specified start and end points along with a target wire,
        compute a curve approximation associated with bending the segment towards the wire.
        """

        pts = []

        branch_start = np.array(branch_start)
        branch_end = np.array(branch_end)

        seg_len = np.linalg.norm(branch_start - branch_end)

        # Check if the segment is close enough to be able to reach the wire
        target = self.compute_wire_point_at_distance(branch_start, seg_len)
        if target is None:
            return None

        current_to_target = np.array(target) - np.array(branch_end)
        start_to_current = np.array(branch_end) - np.array(branch_start)

        x_vals = np.linspace(0, 1, 10 * int(seg_len) + 1, endpoint=True)[1:]
        deflections = self.deflection_at_x(current_to_target, x_vals * seg_len, seg_len)
        pts = branch_start + deflections + x_vals.reshape(-1, 1) * start_to_current.reshape(1, -1)

        return pts

    @staticmethod
    def deflection_at_x(d, x, L):
        """d is the max deflection, x is the current location we need deflection on and L is the total length"""
        x_term = (x ** 2) / (L ** 3 + 0.001) * (3 * L - x)
        if isinstance(x_term, np.ndarray):
            return x_term.reshape(-1, 1) * (d / 2).reshape(1, -1)
        else:
            return (d / 2) * x_term

    def compute_distance_and_closest_pt(self, pt):
        pt = np.array(pt)

        # For a one-sided wire, check to make sure the point is a positive distance in the ray direction
        if self.one_sided and (pt - self.start).dot(self.ray) < 0:
            return np.linalg.norm(pt - self.start), self.start

        closest_pt = self.start + (pt - self.start).dot(self.ray) * self.ray
        return np.linalg.norm(pt - closest_pt), closest_pt

    def compute_wire_point_at_distance(self, pt, dist):
        closest_dist, closest_pt = self.compute_distance_and_closest_pt(pt)
        if closest_dist > dist:
            return None

        along_wire_dist = np.sqrt(dist ** 2 - closest_dist ** 2)
        return closest_pt + along_wire_dist * self.ray

    @property
    def point_rep(self):
        return self.end

    # TEMP - REDO LATER
    def add_branch(self):
        self.num_branch += 1
