from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from graph_utils import LStringGraphDual
import curves

class PruningStrategy:

    @abstractmethod
    def apply_strategy(self, **kwargs):
        ...

    def viz(self):
        return []


class UFOPruningStrategy(PruningStrategy):
    def __init__(self, trunk_targets, wall_targets, **params):
        self.trunk_targets = trunk_targets
        self.wall_targets = wall_targets
        self.params = params
        self.tree = None

        # State variables
        self.next_trunk_target = 0
        self.leaders_assigned = False

    def set_tree(self, tree: LStringGraphDual):
        self.tree = tree

    def __getitem__(self, item):
        return self.params[item]

    def apply_strategy(self, **kwargs):
        self.tie_down_trunk()
        self.examine_leaders()

    def tie_down_trunk(self):
        if not self.next_trunk_target < len(self.trunk_targets):
            return

        target = self.trunk_targets[self.next_trunk_target]
        root_branch = self.tree.search_branches('generation', '=', 0, assert_unique=True)
        nodes = self.tree.branches[root_branch]['nodes']

        # Find the last internode that was tied down and return nodes for all subsequent internodes
        last_tie_idx = 0
        for idx, (n1, n2) in enumerate(zip(nodes[:-1], nodes[1:])):
            data = dict(self.tree.graph.edges[n1, n2].get('post_modules', []))
            if 1 in data.get('Flags', []):
                last_tie_idx = idx + 1
        print('Index of last tie: {}'.format(last_tie_idx))
        nodes = nodes[last_tie_idx:]
        pts = self.tree.branches[root_branch]['points'][last_tie_idx:]

        offsets = np.linalg.norm(pts[:-1] - pts[1:], axis=1)
        cumul_lens = np.cumsum(offsets)
        length = cumul_lens[-1]
        if length > 0:
            _, dist_to_target, _, target_pt = target.get_point_sequence_dist(pts)
            if length > dist_to_target + np.linalg.norm(target_pt - pts[0]):
                params, _, rez = curves.run_cubic_bezier_strain_opt([pts[0], target_pt],
                                                                    pts[2 if nodes[0] == 0 else 1] - pts[0], 1)
                if rez.success:
                    print('Tying to Target {}'.format(self.next_trunk_target))
                    curve_pts = curves.CubicBezier(*params[0]).eval(np.linspace(0, 1, 10))
                    print(curve_pts)
                    curve_len = np.sum(np.linalg.norm(curve_pts[:-1] - curve_pts[1:], axis=1))

                    tie_down_node_idx = (np.argmax(cumul_lens > curve_len) or len(nodes) - 2) + 1
                    self.tree.set_guide_on_nodes(nodes[:tie_down_node_idx + 1], curve_pts, 'GlobalGuide')

                    self.next_trunk_target += 1

    def examine_leaders(self):
        root_branch = self.tree.search_branches('generation', '=', 0, assert_unique=True)
        if root_branch['length'] < 5:
            self.rub_off_trunk_buds()

        elif not self.leaders_assigned:
            self.assign_leaders()

        else:
            self.prune_and_tie_down_leaders()

    def rub_off_trunk_buds(self):
        pass

    def assign_leaders(self):
        pass

    def prune_and_tie_down_leaders(self):
        pass