"""
Defines the abstract class BasicWood, class Wire and class Support.
"""

from openalea.plantgl.all import *
import copy
import numpy as np
from openalea.plantgl.scenegraph.cspline import CSpline
import random as rd
from collections import defaultdict

import collections

eps = 1e-6

from abc import ABC, abstractmethod


# class Tree():
#   #branch_dict = {}
#   trunk_dict = {}
#   """ This class will have all the parameters required to grow the tree, i.e. the transition
#   prob, max trunk length, max branch length etc. Each tree will have its own children branch and trunk classes """
#   def __init__(self):
#     self.trunk_num_buds_segment = 5
#     self.branch_num_buds_segment = 5
#     self.trunk_bud_break_prob = 0.5
#     self.branch_bud_break_prob = 0.5
#     self.num_branches = 0
#     self.num_trunks = 0


# # BRANCH AND TRUNK SUBCLASS OF WOOD		

class BasicWood(ABC):

    @staticmethod
    def clone(obj):
        try:
            return copy.deepcopy(obj)
        except copy.Error:
            raise copy.Error(f'Not able to copy {obj}') from None

    def __init__(self, copy_from=None, max_buds_segment: int = 5, thickness: float = 0.1,
                 thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,
                 tie_axis: tuple = (0, 1, 1), order: int = 0, color: int = 0, name: str = None):  # ,\
        # bud_break_prob_func: "Function" = lambda x,y: rd.random()):

        # Location variables
        if copy_from:
            self.__copy_constructor__(copy_from)
            return
        self.start = Vector3(0, 0, 0)
        self.end = Vector3(0, 0, 0)
        # Tying variables
        self.last_tie_location = Vector3(0, 0, 0)
        self.has_tied = False
        self.guide_points = []
        self.current_tied = False
        self.guide_target = -1  # Vector3(0,0,0)
        self.tie_axis = tie_axis
        self.tie_updated = True
        # Information Variables
        self.__length = 0
        self.age = 0
        self.cut = False
        self.order = order
        self.num_branches = 0
        self.branch_dict = collections.deque()
        self.color = color
        # Growth Variables
        self.max_buds_segment = max_buds_segment
        self.thickness = thickness
        self.thickness_increment = thickness_increment
        self.growth_length = growth_length
        self.max_length = max_length

    def __copy_constructor__(self, copy_from):
        update_dict = copy.deepcopy(copy_from.__dict__)
        for k, v in update_dict.items():
            setattr(self, k, v)
        # self.__dict__.update(update_dict)

    @abstractmethod
    def is_bud_break(self) -> bool:
        """This method defines if a bud will break or not -> returns true for yes, false for not. Input can be any variables"""
        pass
        # Example
        # prob_break = self.bud_break_prob_func(num_buds, self.num_buds_segment)
        # #Write dummy probability function
        # if prob_break > self.bud_break_prob:
        #   return True
        # return False

    @abstractmethod
    def grow(self) -> None:
        """This method can define any internal changes happening to the properties of the class, such as reduction in thickness increment etc."""
        pass

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, length):
        self.__length = min(length, self.max_length)

    def grow_one(self):
        self.age += 1
        self.length += 1
        self.grow()

    @abstractmethod
    def create_branch(self) -> "BasicWood Object":
        """Returns how a new order branch when bud break happens will look like if a bud break happens"""
        pass
        # new_object = BasicWood.clone(self.branch_object)
        # return new_object
        # return BasicWood(self.num_buds_segment/2, self.bud_break_prob, self.thickness/2, self.thickness_increment/2, self.growth_length/2,\
        # self.max_length/2, self.tie_axis, self.bud_break_max_length/2, self.order+1, self.bud_break_prob_func)

    def update_guide(self, guide_target):
        self.guide_target = guide_target
        if self.guide_target == -1:
            return
        start = self.last_tie_location if self.has_tied else self.start
        assert isinstance(guide_target, WireVector)
        curve, i_target = self.get_control_points(guide_target.end, start, self.end, guide_target)
        if i_target:
            self.guide_points.extend(curve)
            # print(i_target, self.guide_points[-1])
            # self.last_tie_location = copy.deepcopy(Vector3(i_target)) #Replaced by updating location at StartEach

    def tie_lstring(self, lstring, index):
        spline = CSpline(self.guide_points)
        # print(lstring[index+1].name in ['&','/','SetGuide'], lstring[index+1])
        remove_count = 0
        if not self.has_tied:
            if lstring[index + 1].name in ['&', '/', 'SetGuide']:
                # print("DELETING", lstring[index+1])
                del (lstring[index + 1])
                remove_count += 1
            self.has_tied = True
        if lstring[index + 1].name in ['&', '/', 'SetGuide']:
            # print("DELETING", lstring[index+1])
            del (lstring[index + 1])
            remove_count += 1

        lstring.insertAt(index + 1, 'SetGuide({}, {})'.format(spline.curve(), self.length))
        return lstring, remove_count

    def tie_update(self):
        self.last_tie_location = copy.deepcopy(self.end)
        self.tie_updated = True

    def deflection_at_x(self, d, x, L):
        """d is the max deflection, x is the current location we need deflection on and L is the total length"""
        x_term = (x ** 2) / (L ** 3 + 0.001) * (3 * L - x)
        if isinstance(x_term, np.ndarray):
            return x_term.reshape(-1, 1) * (d/2).reshape(1, -1)
        else:
            return (d / 2) * x_term

    # return d*(1 - np.cos(*np.pi*x/(2*L))) #Axial loading

    def get_control_points(self, final_target, start, current, wire_obj):
        pts = []

        final_target = np.array(final_target)
        start = np.array(start)
        current = np.array(current)

        curve_len = np.linalg.norm(start - current)

        # Compute the closest point on the wire to the final target
        # Check if the segment is close enough to be able to reach
        dist_to_wire, target = wire_obj.compute_distance_and_closest_pt(final_target)
        if dist_to_wire > curve_len:
            print("SHORT")
            return pts, None

        current_to_target = np.array(target) - np.array(current)
        start_to_current = np.array(current) - np.array(start)

        x_vals = np.linspace(0, 1, 10 * int(curve_len) + 1, endpoint=True)[1:]
        deflections = self.deflection_at_x(current_to_target, x_vals * curve_len, curve_len)
        pts = start + deflections + x_vals.reshape(-1, 1) * start_to_current.reshape(1, -1)

        return map(tuple, pts), tuple(target)


# class Branch(BasicWood):
#   def __init__(self, num_buds_segment: int = 5, bud_break_prob: float = 0.8, thickness: float = 0.1,\
#                thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,\
#                tie_axis: tuple = (0,1,1), bud_break_max_length: int = 5, order: int = 0, bud_break_prob_func: "Function" = lambda x,y: rd.random()):
#     super().__init__(num_buds_segment, bud_break_prob, thickness, thickness_increment, growth_length,\
#                max_length, tie_axis, bud_break_max_length, order, bud_break_prob_func)


# class Trunk(BasicWood):
#   """ Details of the trunk while growing a tree, length, thickness, where to attach them etc """
#   def __init__(self, num_buds_segment: int = 5, bud_break_prob: float = 0.8, thickness: float = 0.1,\
#                thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,\
#                tie_axis: tuple = (0,1,1), bud_break_max_length: int = 5, order: int = 0, bud_break_prob_func: "Function" = lambda x,y: rd.random()):
#      super().__init__(num_buds_segment, bud_break_prob, thickness, thickness_increment, growth_length,\
#                max_length, tie_axis, bud_break_max_length, order, bud_break_prob_func)

class Architecture:
    """
    Defines a collection of WireVectors sorted into different categories.
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
                pts.extend(map(lambda x: np.array(x.end).astype(np.double), v))
            else:
                pts.append(np.array(v.end).astype(np.double))

        return Point3Grid((1,1,1), pts)


class WireVector:

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

    def compute_distance_and_closest_pt(self, pt):
        pt = np.array(pt)

        # For a one-sided wire, check to make sure the point is a positive distance in the ray direction
        if self.one_sided and (pt - self.start).dot(self.ray) < 0:
            return np.linalg.norm(pt - self.start), self.start

        closest_pt = self.start + (pt - self.start).dot(self.ray) * self.ray
        return np.linalg.norm(pt - closest_pt), closest_pt

    # TEMP - REDO LATER
    def add_branch(self):
        self.num_branch += 1

class Wire():
    """ Defines a trellis wire in the 3D space """

    def __init__(self, id: int, point: tuple, axis: tuple):
        self.__id = id
        self.__axis = axis
        x, y, z = point
        self.point = Vector3(x, y, z)
        self.num_branch = 0

    def add_branch(self):
        self.num_branch += 1




class Support():
    """ All the details needed to figure out how the support is structured in the environment, it is a collection of wires"""

    def __init__(self, points: list, num_wires: int, spacing_wires: int, trunk_wire_pt: tuple, \
                 branch_axis: tuple, trunk_axis: tuple):

        self.num_wires = num_wires
        self.spacing_wires = spacing_wires
        self.branch_axis = branch_axis
        self.branch_supports = self.make_support(points)  # Dictionary id:points
        self.trunk_axis = None
        self.trunk_wire = None
        if trunk_axis:
            self.trunk_axis = trunk_axis
            self.trunk_wire = Wire(-1, trunk_wire_pt, self.trunk_axis)  # Make it a vector?
            points.append(trunk_wire_pt)

        self.attractor_grid = Point3Grid((1, 1, 1), list(points))

    def make_support(self, points):
        supports = {}
        for id, pt in enumerate(points):
            supports[id] = Wire(id, pt, self.branch_axis)
        return supports
