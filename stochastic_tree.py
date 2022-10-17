"""
Defines the abstract class BasicWood, class Wire and class Support.
"""
from __future__ import annotations

from openalea.plantgl.all import *
import copy
from openalea.plantgl.scenegraph.cspline import CSpline
from architecture import StructuralElement
from typing import Optional
import collections
from abc import ABC, abstractmethod
from scheduler import Scheduler, ConstantScheduler
from typing import Optional, Union, Tuple, List
from numbers import Number


class BasicWood(ABC):

    COUNT = 0

    @classmethod
    def copy_from(cls, obj):
        kwargs = copy.deepcopy(obj.__dict__)
        return cls(**kwargs)

    def __init__(self, max_buds_segment: int = 5, thickness_schedule: Optional[Scheduler] = None,
                 length_schedule: Optional[Scheduler] = None, color: Union[int, Tuple[int, int, int]] = 0,
                 can_tie=False, name=None):

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
        self.age = 0
        self.cut = False
        self.num_branches = 0
        self.branch_dict = collections.deque()
        self.color = color
        # Growth Variables
        self.max_buds_segment = max_buds_segment
        self.thickness_schedule = thickness_schedule or ConstantScheduler(0.1)
        self.length_schedule = length_schedule or ConstantScheduler(1.0)
        if not name:
            self.name = f'{self.__class__.__name__}{self.__class__.COUNT}'

        self.__class__.COUNT += 1


    def __repr__(self):
        return '{}<({:.2f},{:.2f},{:.2f}), ({:.2f},{:.2f},{:.2f})>'.format(
            self.name, *self.start, *self.end
        )

    # @abstractmethod
    # def is_bud_break(self) -> bool:
    #     """This method defines if a bud will break or not -> returns true for yes, false for not. Input can be any variables"""
    #     pass
    #     # Example
    #     # prob_break = self.bud_break_prob_func(num_buds, self.num_buds_segment)
    #     # #Write dummy probability function
    #     # if prob_break > self.bud_break_prob:
    #     #   return True
    #     # return False

    @abstractmethod
    def get_branch_growth_on_segment(self, start, end) -> List[Tuple[Number, BasicWood], ...]:
        """
        Returns a list of number-BasicWood pairs specifying the objects to be grown
        """
        ...

    def grow(self) -> None:
        """This method can define any internal changes happening to the properties of the class, such as reduction in thickness increment etc."""
        pass

    def grow_one(self):

        prev_len = self.length

        self.age += 1
        self.grow()

        new_len = self.length
        len_diff = new_len - prev_len

        branches = self.get_branch_growth_on_segment(prev_len, new_len)
        return len_diff, branches

    @property
    def length(self):
        return self.length_schedule(self.age)

    @property
    def thickness(self):
        return self.thickness_schedule(self.age)

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
