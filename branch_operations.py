import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from networkx.readwrite import json_graph

def extract_bendable_branches(graph):
    root_node = [n for n in graph if list(graph.successors(n)) and not list(graph.predecessors(n))][0]
    tip_nodes = [n for n in graph if not list(graph.successors(n)) and list(graph.predecessors(n))]

    # Annotate the graph for upstream dependencies
    for tip_node in tip_nodes:
        upstream_branches = set()
        last_branch_id = None
        for edge in nx.dfs_edges(graph.reverse(), source=tip_node):
            branch_id = graph[edge]['branch_id']
            if last_branch_id is not None and branch_id != last_branch_id:
                upstream_branches.add(branch_id)
            graph[edge]['upstream_branches'] = graph[edge].get('upstream_branches', set()).union(upstream_branches)
            last_branch_id = branch_id
            if edge[1] == root_node:
                break

    import pdb
    pdb.set_trace()









def visualize_skeleton(graph):

    pts = np.array([graph.nodes[n]['position'] for n in graph])

    starts = np.array([graph.nodes[e[0]]['position'] for e in graph.edges])
    ends = np.array([graph.nodes[e[1]]['position'] for e in graph.edges])
    bids = [graph.edges[e]['branch_id'] for e in graph.edges]
    from matplotlib import colormaps
    hsv_map = colormaps.get('hsv')
    colors = [hsv_map((bid % 16) / 16) for bid in bids]

    segs = np.concatenate([starts.reshape((-1, 1, 3)), ends.reshape((-1, 1, 3))], axis=1)
    lc = Line3DCollection(segs, colors=colors)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    ax.add_collection3d(lc)

    plt.show()


if __name__ == '__main__':

    import json
    path = r'C:\Users\davijose\Documents\data\test.json'
    with open(path, 'r') as fh:
        data = json.load(fh)

    graph = json_graph.node_link_graph(data)
    extract_bendable_branches(graph)

    visualize_skeleton(graph)

