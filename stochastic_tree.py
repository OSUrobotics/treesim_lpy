"""
Defines the abstract class BasicWood, class Wire and class Support.
"""

from openalea.plantgl.all import *
import copy
from openalea.plantgl.scenegraph.cspline import CSpline
from architecture import StructuralElement
from typing import Optional
import collections
from abc import ABC, abstractmethod


class BasicWood(ABC):

    @staticmethod
    def clone(obj):
        try:
            return copy.deepcopy(obj)
        except copy.Error:
            raise copy.Error(f'Not able to copy {obj}') from None

    def __init__(self, copy_from=None, max_buds_segment: int = 5, thickness: float = 0.1,
                 thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,
                 order: int = 0, color: int = 0):

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
        self.guide_target = None
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

    def __repr__(self):
        return '{}<({:.2f},{:.2f},{:.2f}), ({:.2f},{:.2f},{:.2f})>'.format(
            self.__class__.__name__, *self.start, *self.end
        )

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


    def update_guide(self, guide_target: Optional[StructuralElement]):
        """
        Update the curve associated with bending the wood object towards a passed in guide.
        """

        self.guide_target = guide_target
        if self.guide_target is None:
            return

        start = self.last_tie_location if self.has_tied else self.start
        curve = guide_target.bend_segment_curve(start, self.end)
        if curve is not None:
            self.guide_points.extend(map(tuple, curve))

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
