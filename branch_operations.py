import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from networkx.readwrite import json_graph
from collections import defaultdict
import os
from abc import ABC, abstractmethod
from openalea.plantgl.scenegraph.cspline import CSpline

class Target(ABC):

    @abstractmethod
    def get_segment_dist(self, seg_0, seg_1):
        ...

    def get_point_sequence_dist(self, pts):
        best_dist = np.inf
        best_to_return = None
        for idx, (pt_1, pt_2) in enumerate(zip(pts[:-1], pts[1:])):
            if np.linalg.norm(pt_1 - pt_2) < 1e-5:
                continue


            target_dist, alt_dist, pt = self.get_segment_dist(pt_1, pt_2)
            if target_dist < best_dist:
                best_dist = target_dist
                best_to_return = (idx, target_dist, alt_dist, pt)

        return best_to_return


class PointTarget(Target):
    def __init__(self, pt):
        self.pt = np.array(pt)

    def get_segment_dist(self, seg_0, seg_1):
        seg_0 = np.array(seg_0)
        seg_1 = np.array(seg_1)
        seg_len = np.linalg.norm(seg_1 - seg_0)
        vec = (seg_1 - seg_0) / seg_len

        diff = self.pt - seg_0
        proj_len = diff.dot(vec)
        target_pt = seg_0 + proj_len * vec
        target_dist = np.linalg.norm(self.pt - target_pt)
        if not (0 <= proj_len <= seg_len):
            if proj_len < seg_len / 2:
                alt_pt = seg_0
                alt_dist = np.linalg.norm(self.pt - seg_0)
            else:
                alt_pt = seg_1
                alt_dist = np.linalg.norm(self.pt - seg_1)
        else:
            alt_pt = target_pt
            alt_dist = target_dist

        return target_dist, alt_dist, self.pt


class LinearTarget(Target):

    # https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d

    def __init__(self, origin, ray):
        self.origin = np.array(origin)
        self.ray = normalize(np.array(ray))

    def get_segment_dist(self, seg_0, seg_1):
        seg_0 = np.array(seg_0)
        seg_1 = np.array(seg_1)
        seg_len = np.linalg.norm(seg_1 - seg_0)
        vec = normalize(seg_1 - seg_0)

        r1, e1 = self.origin, self.ray
        r2, e2 = seg_0, vec
        n = np.cross(e1, e2)
        d = np.abs(n.dot(r1 - r2))
        t1 = np.cross(e2, n).dot(r2-r1)
        t2 = np.cross(e1, n).dot(r2-r1)

        closest_wire_pt = r1 + t1 * e1
        closest_wire_dist = d

        if not (0 <= t2 <= seg_len):
            if t2 < seg_len / 2:
                alt_pt = seg_0
                alt_dist = np.linalg.norm(closest_wire_pt - seg_0)
            else:
                alt_pt = seg_1
                alt_dist = np.linalg.norm(closest_wire_pt - seg_1)
        else:
            alt_pt = r2 + t2 * e2
            alt_dist = d

        return closest_wire_dist, alt_dist, closest_wire_pt


def normalize(vec):
    return vec / np.linalg.norm(vec)

def reorganize_branch_edges(edges):

    starts, ends = zip(*edges)
    info = dict(edges)
    starts = set(starts)
    ends = set(ends)
    start_nodes = list(starts - ends)
    assert len(start_nodes) == 1

    cur_node = start_nodes[0]
    branch = [cur_node]
    while cur_node in info:
        next_node = info[cur_node]
        branch.append(next_node)
        cur_node = next_node

    return branch


def extract_bendable_branches(graph):
    root_node = [n for n in graph if list(graph.successors(n)) and not list(graph.predecessors(n))][0]
    tip_nodes = {n for n in graph if not list(graph.successors(n)) and list(graph.predecessors(n))}

    # Annotate the graph for upstream dependencies
    branch_edges = defaultdict(set)
    for tip_node in tip_nodes:
        upstream_branches = set()
        last_branch_id = None
        for edge in nx.dfs_edges(graph.reverse(), source=tip_node):
            edge = edge[::-1]
            branch_id = graph.edges[edge]['branch_id']
            branch_edges[branch_id].add(edge)
            if last_branch_id is not None and branch_id != last_branch_id:
                upstream_branches.add(branch_id)
            graph.edges[edge]['upstream_branches'] = graph.edges[edge].get('upstream_branches', set()).union(upstream_branches)
            last_branch_id = branch_id
            if edge[1] == root_node:
                break

    # For each branch, locate any existing tie nodes
    bendable_branch_info = {}
    for branch_id, branch_edges in branch_edges.items():
        branch_nodes = reorganize_branch_edges(branch_edges)
        upstream = graph.edges[branch_nodes[0], branch_nodes[1]]['upstream_branches']
        start_idx = 0
        for i, node in enumerate(branch_nodes[1:], start=1):
            if graph[node].get('type') == 'T':
                start_idx = i
        nodes_from_last_tie = branch_nodes[start_idx:]
        if len(nodes_from_last_tie) > 1:

            pts = np.array([graph.nodes[n]['position'] for n in nodes_from_last_tie])
            total_len = np.linalg.norm(pts[:-1] - pts[1:], axis=1).sum()
            bendable_branch_info[branch_id] = {
                'nodes': nodes_from_last_tie,
                'upstream': upstream,
                'length': total_len,
                'points': pts,
                'type': graph.edges[nodes_from_last_tie[0], nodes_from_last_tie[1]]['name']
            }
    
    return bendable_branch_info


def compute_distance_to_target(graph, branch, target: Target):
    segs = [(graph.nodes[x]['position'], graph.nodes[y]['position']) for x, y in zip(branch[:-1], branch[1:])]

    min_true_dist = np.inf
    best_target = None

    ray_dist = None # Captures ray dist for the final branch segment only
    ray_target = None

    for seg in segs:
        ray_dist, true_dist, pt_target = target.get_segment_dist(*seg)
        ray_target = pt_target
        if true_dist < min_true_dist:
            min_true_dist, best_target = true_dist, pt_target

    if min_true_dist < ray_dist:
        return min_true_dist, best_target
    else:
        return ray_dist, ray_target


def visualize_skeleton(graph, diag=None):

    pts = np.array([graph.nodes[n]['position'] for n in graph])

    starts = np.array([graph.nodes[e[0]]['position'] for e in graph.edges])
    ends = np.array([graph.nodes[e[1]]['position'] for e in graph.edges])
    bids = [graph.edges[e].get('branch_id', -1) for e in graph.edges]
    from matplotlib import colormaps
    hsv_map = colormaps.get('hsv')
    colors = [hsv_map(((11 * bid) % 16) / 16) for bid in bids]

    segs = np.concatenate([starts.reshape((-1, 1, 3)), ends.reshape((-1, 1, 3))], axis=1)
    lc = Line3DCollection(segs, colors=colors)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    ax.add_collection3d(lc)


    if diag is not None:
        plt.plot(*diag.T, linestyle='dashed')

    y_min, y_max = ax.get_ylim()
    if y_max - y_min < 0.5:
        ax.set_ylim(-2.5, 2.5)

    plt.show()


if __name__ == '__main__':

    import json
    # path = r'C:\Users\davijose\Documents\data\test.json'
    path = os.path.join('test', 'test.json')

    with open(path, 'r') as fh:
        data = json.load(fh)

    graph = json_graph.node_link_graph(data)
    branches = extract_bendable_branches(graph)
    target_obj = PointTarget([0.5, 0.0, 6.6])
    # target_obj = PointTarget([4.3, 0.0, 6.6])

    best_cand = None
    best_target = None
    best_cost = np.inf

    for branch_id, branch_info in branches.items():
        nodes = branch_info['nodes']
        dist, target = compute_distance_to_target(graph, nodes, target_obj)
        total_len = branch_info['length']
        if total_len > dist + np.linalg.norm(branch_info['points'][0] - target):
            cost = dist
            if cost < best_cost:
                best_cand, best_cost, best_target = branch_id, cost, target

    if best_cand is not None:

        print('Tying down branch {}'.format(best_cand))

        from curves import run_cubic_bezier_strain_opt, CubicBezier
        nodes = branches[best_cand]['nodes']
        branch_info = branches[best_cand]
        first_pt = branch_info['points'][0]
        next_pt = branch_info['points'][1]

        params = run_cubic_bezier_strain_opt([first_pt, best_target], next_pt - first_pt)[0][0]
        curve = CubicBezier(*params)
        curve_pts = curve(np.linspace(0, 1, 50))
        spline = CSpline(curve_pts)
        length = branch_info['length']
        guide = f'SetGuide({spline.curve()}, {length})'

        print(guide)
        diagnostic = curve_pts
    else:
        diagnostic = None

    visualize_skeleton(graph, diag=diagnostic)

