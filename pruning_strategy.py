from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from graph_utils import LStringGraphDual
import curves
from collections import defaultdict
from branch_operations import Target, LinearTarget, PointTarget

class PruningStrategy:

    @abstractmethod
    def apply_strategy(self, **kwargs):
        ...

    def viz(self):
        return []


class UFOPruningStrategy(PruningStrategy):
    def __init__(self, trunk_targets, wire_walls, **params):
        self.trunk_targets = trunk_targets
        self.wire_walls = wire_walls
        self.params = params
        self.tree = None
        self._trunk_branch = None

        # State variables
        self.next_trunk_target = 0
        self.leaders_assigned = False

    # Helper methods

    @property
    def trunk_branch(self):
        if self._trunk_branch is None:
            root_branch = self.tree.search_branches('generation', '=', 0, assert_unique=True)
            self._trunk_branch = self.tree.branches[root_branch]
        return self._trunk_branch

    def set_tree(self, tree: LStringGraphDual):
        self.tree = tree
        self._trunk_branch = None

    def __getitem__(self, item):
        return self.params[item]

    def find_module(self, module, edge, pre=True):

        pre_mods = self.tree.graph.edges[edge].get('pre_modules' if pre else 'post_modules', [])
        flags = None
        for mod in pre_mods:
            if mod[0] == module:
                flags = mod[1]
                break
        return flags


    def tie_branch(self, nodes, target_obj: Target, tie_module=None):
        pts = np.array([self.tree.graph.nodes[n]['position'] for n in nodes])
        offsets = np.linalg.norm(pts[:-1] - pts[1:], axis=1)
        cumul_lens = np.cumsum(offsets)
        length = cumul_lens[-1]
        if length > 0:

            _, dist_to_target, _, target_pt = target_obj.get_point_sequence_dist(pts)
            if length > dist_to_target + np.linalg.norm(target_pt - pts[0]):
                params, _, rez = curves.run_cubic_bezier_strain_opt([pts[0], target_pt],
                                                                    pts[2 if nodes[0] == 0 else 1] - pts[0], 1)
                if rez.success:




                    curve_pts = curves.CubicBezier(*params[0]).eval(np.linspace(0, 1, 10))
                    curve_len = np.sum(np.linalg.norm(curve_pts[:-1] - curve_pts[1:], axis=1))
                    tie_down_node_idx = (np.argmax(cumul_lens > curve_len) or len(nodes) - 2)
                    self.tree.set_guide_on_nodes(nodes[:tie_down_node_idx + 1], curve_pts, 'GlobalGuide')

                    print('-------------\nSuccessfully tied branch to Target {}'.format(target_obj))
                    print('Start pt: {:.3f}, {:.3f}, {:.3f}'.format(*pts[0]))
                    print('End pt: {:.3f}, {:.3f}, {:.3f}'.format(*curve_pts[-1]))

                    tied_edge = (nodes[tie_down_node_idx], nodes[tie_down_node_idx+1])
                    if tie_module:
                        self.tree.graph.edges[tied_edge]['pre_modules'].append(tie_module)

                    return tied_edge

        return None

    def find_buds(self, nodes):
        buds = []
        for node in nodes:
            next_nodes = self.tree.graph.successors(node)
            for next_node in next_nodes:
                if self.tree.graph.nodes[next_node]['name'] == 'B':
                    buds.append((node, next_node))
        return buds

    # Related to steps

    def apply_strategy(self, **kwargs):
        self.tie_down_trunk()
        self.examine_leaders()

    def tie_down_trunk(self):

        nodes = self.trunk_branch['nodes']
        # Find the last internode that was tied down and return nodes for all subsequent internodes
        last_tie_idx = 0
        for idx, (n1, n2) in enumerate(zip(nodes[:-1], nodes[1:])):
            if self.find_module('Tie', (n1, n2)):
                last_tie_idx = idx + 1

        if not self.next_trunk_target < len(self.trunk_targets):
            self.tree.stub_branch(nodes[last_tie_idx + 1:], self.params.get('trunk_excess_stub', 2.5))
            return

        target = self.trunk_targets[self.next_trunk_target]
        print('Index of last tie: {}'.format(last_tie_idx))
        nodes = nodes[last_tie_idx:]

        tied_edge = self.tie_branch(nodes, target, ('Tie', ['Trunk', self.next_trunk_target]))
        if tied_edge:
            self.next_trunk_target += 1

    def examine_leaders(self):
        if self.trunk_branch['length'] < 5:
            self.rub_off_trunk_buds()
        self.manage_leaders()


    def rub_off_trunk_buds(self, avoid_marked=True):
        trunk_nodes = self.trunk_branch['nodes']
        to_remove = []
        for edge in self.find_buds(trunk_nodes):
            if not avoid_marked or (avoid_marked and self.find_module('Mark', edge)):
                to_remove.append(edge)

        self.tree.graph.remove_edges_from(to_remove)


    def manage_leaders(self):
        first_gen_branches = self.tree.search_branches('generation', '=', 1)
        # Determine the status of each branch - Unassigned and untied (0), assigned but untied (1), tied (2)
        existing_ties = defaultdict(list)

        for branch_id in first_gen_branches:
            branch = self.tree.branches[branch_id]
            nodes = branch['nodes']

            # Iterate through edges in reverse order and see if there are any ties
            for edge in zip(nodes[-2:1:-1], nodes[-1:2:-1]):
                tie_info = self.find_module('Tie', edge)
                if tie_info is not None:
                    wall_id, wall_tie_idx = tie_info
                    existing_ties[wall_id].append(nodes[0])
                    self.manage_tied_branch(nodes[nodes.index(edge[1]):], wall_id, wall_tie_idx)
                    break
            else:
                # No existing tie was found - Check to see if it's marked for tying
                mark = self.find_module('Mark', (nodes[0], nodes[1]))
                if mark is not None:
                    wall_id = mark[0]
                    existing_ties[wall_id].append(nodes[0])
                    self.manage_untied_branch(nodes, wall_id)

        # Check to see if all leaders have been assigned to each wall
        leaders_per_wall = self.params.get('leaders_per_wall', 5)
        is_full = True
        for wall_id in range(len(self.wire_walls)):
            trunk_nodes = existing_ties[wall_id]
            if len(trunk_nodes) < leaders_per_wall:
                is_full = False

        # If there is room, search for new buds that can be added to each wall
        if not is_full:
            desired_spacing = self.params.get('leader_spacing', 1.5)
            root_branch_id = self.tree.search_branches('generation', '=', 0, assert_unique=True)
            root_branch = self.tree.branches[root_branch_id]
            root_nodes = root_branch['nodes']

            # Find ties on the main root
            last_tie = None
            for i, edge in enumerate(zip(root_nodes[:-1], root_nodes[1:])):
                if self.find_module('Tie', edge) is not None:
                    last_tie = i
            if not last_tie:
                return
            root_nodes = root_nodes[:last_tie+1]

            pts = root_branch['points']
            dists = np.linalg.norm(pts[:-1] - pts[1:], axis=1)
            cum_dists = np.cumsum(dists)

            # For each wall, convert the root nodes to spacings along the trunk
            node_dist_map = dict(zip(root_nodes[1:], cum_dists))
            node_dist_map[root_nodes[0]] = 0

            wall_spacings = {}
            for wall_id, trunk_nodes in existing_ties.items():
                spacings = [node_dist_map[n] for n in trunk_nodes]
                wall_spacings[wall_id] = sorted(spacings)

            trunk_buds = self.find_buds(root_nodes)

            print('Looking for leaders (eligible trunk buds: {})'.format(len(trunk_buds)))

            for bud_edge in trunk_buds:
                node_on_trunk = bud_edge[0]
                cum_dist = node_dist_map[node_on_trunk]
                print('{}: {:.3f}'.format(bud_edge, cum_dist))
                if cum_dist < self.params.get('trunk_bare_dist', 5):
                    continue

                best_spacing = None
                best_wall_id = None
                for wall_id, spacings in wall_spacings.items():
                    if len(spacings) >= leaders_per_wall:
                        continue
                    spacing = query_dist_from_line_points(cum_dist, spacings)
                    if spacing < desired_spacing:
                        continue

                    if best_spacing is None or spacing < best_spacing:
                        best_spacing, best_wall_id = spacing, wall_id

                if best_wall_id is not None:

                    print('Marked bud {} to wall {}'.format(bud_edge, best_wall_id))
                    self.tree.graph.edges[bud_edge]['pre_modules'] = [('Mark', [best_wall_id])]

                    # Simulate bud activation
                    self.tree.graph.nodes[bud_edge[1]]['args'][1] = 0.9

                    wall_spacings[best_wall_id].append(cum_dist)
                    wall_spacings[best_wall_id].sort()

    def manage_tied_branch(self, nodes, wall_id, wall_tie_idx):

        if wall_tie_idx == len(self.wire_walls[wall_id]) - 1:
            self.tree.stub_branch(nodes, self.params.get('leader_excess_stub', 2.5))
        else:
            next_target = self.wire_walls[wall_id][wall_tie_idx + 1]
            base_pt = self.tree.graph.nodes[nodes[0]]['position']
            _, wire_target = next_target.get_point_dist(base_pt)
            self.tie_branch(nodes, PointTarget(wire_target), ('Tie', [wall_id, wall_tie_idx + 1]))


    def manage_untied_branch(self, nodes, wall_id):

        next_target = self.wire_walls[wall_id][0]
        base_pt = self.tree.graph.nodes[nodes[0]]['position']
        _, wire_target = next_target.get_point_dist(base_pt)
        self.tie_branch(nodes, PointTarget(wire_target), ('Tie', [wall_id, 0]))


def query_dist_from_line_points(pt, sorted_array):
    if not len(sorted_array):
        return np.inf

    if pt < sorted_array[0]:
        return sorted_array[0] - pt

    elif pt >= sorted_array[-1]:
        return pt - sorted_array[-1]

    else:
        for l, u in zip(sorted_array[:-1], sorted_array[1:]):
            if l <= pt < u:
                return min(pt-l, u-pt)