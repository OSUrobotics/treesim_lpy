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

    COUNT = 0

    @classmethod
    def copy_from(cls, obj):
        kwargs = copy.deepcopy(obj.__dict__)
        return cls(**kwargs)

    def __init__(self, max_buds_segment: int = 5, thickness: float = 0.1,
                 thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,
                 order: int = 0, color: int = 0, can_tie=False, name=None):

        # Location variables
        self.start = Vector3(0, 0, 0)
        self.end = Vector3(0, 0, 0)
        # Tying variables
        self.can_tie = can_tie
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
        if not name:
            self.name = f'{self.__class__.__name__}{self.__class__.COUNT}'

        self.__class__.COUNT += 1


    def __repr__(self):
        return '{}<({:.2f},{:.2f},{:.2f}), ({:.2f},{:.2f},{:.2f})>'.format(
            self.name, *self.start, *self.end
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

    def produce_tie_guide(self):
        if not self.can_tie:
            return None

        spline = CSpline(self.guide_points)
        elem = f'SetGuide({spline.curve()}, {self.length})'
        return elem

    def tie_update(self):
        if not self.can_tie:
            raise ValueError('Cannot update the tie for a branch which cannot be tied!')
        self.last_tie_location = copy.deepcopy(self.end)
        self.tie_updated = True
