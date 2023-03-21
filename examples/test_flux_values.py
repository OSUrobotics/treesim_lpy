import sys
import networkx as nx
import numpy as np
sys.path.append('../')
from graph_utils import compute_source_sink_values

class NodeProperties:
    def __init__(self, capacity, zero_sink, init_mass=0.0):
        self.capacity = capacity
        self.zero_sink = zero_sink
        self.mass = init_mass

    @property
    def source(self):
        return self.zero_sink - self.mass * self.zero_sink / self.capacity


class BranchStorageNodeProperties(NodeProperties):
    def __init__(self, length, width, k_vol, zero_sink, init_mass=0.0):

        volume = length * width
        cap = volume * k_vol
        super().__init__(cap, zero_sink, init_mass=init_mass)

class RootSource:
    def __init__(self, init_source=0.0, init_mass=0):
        self.source = init_source
        self.mass = init_mass


def visualize_graph(graph, width_multiplier=20):
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    import matplotlib.patches as mp
    import matplotlib.cm as cm

    fig, ax = plt.subplots()

    annotations = []

    pts = np.array([graph.nodes[n]['point'] for n in graph.nodes])
    concs = np.array([graph.nodes[n]['props'].mass for n in graph.nodes])
    sources = np.array([graph.nodes[n]['props'].source for n in graph.nodes])
    conc_cmap = cm.rainbow((concs - concs.min()) / (concs.max() - concs.min()))
    annotations.extend([('Mass: {:.3f}\nSource: {:.3f}'.format(conc, source), pt) for pt, conc, source in zip(pts, concs, sources)])

    line_segs = []
    widths = []
    flux_vals = []


    for edge in graph.edges:
        flux = graph.edges[edge]['flux']

        if flux > 0:
            s, e = edge
        else:
            e, s = edge

        pt_s = np.array(graph.nodes[s]['point'])
        pt_e = np.array(graph.nodes[e]['point'])

        line_segs.append([pt_s, pt_e])
        widths = graph.edges[edge]['width'] * width_multiplier
        flux_vals.append(abs(flux))

        annotations.append(('Flux: {:.3f}'.format(abs(flux)), (pt_s + pt_e) / 2))

    lc = mc.LineCollection(line_segs, linewidths=widths)

    flux_vals = np.array(flux_vals)
    flux_cmap = cm.rainbow((flux_vals - flux_vals.min()) / (flux_vals.max() - flux_vals.min()))

    arrows = [mp.Arrow(*ls[0], *ls[1]-ls[0], color=c) for ls, c in zip(line_segs, flux_cmap)]
    pc = mc.PatchCollection(arrows)

    ax.add_collection(lc)
    # ax.add_collection(pc)
    ax.scatter(pts[:,0], pts[:,1], color=conc_cmap)
    for annotation in annotations:
        ax.annotate(*annotation)
    plt.show()





def compute_resistances(graph, edge_coeff=0.1, sink_coeff=1.0):
    for edge in graph.edges:
        if isinstance(edge[1], int):
            # Regular edge
            info = graph.edges[edge]
            r = edge_coeff * info['length'] / (np.pi * info['width'] ** 2)
        else:
            pred = list(graph.predecessors(edge[0]))[0]
            real_edge = (pred, edge[0])
            info = graph.edges[real_edge]
            # Implicit sink - The resistance is proportional to the cube of the radius and inverse to the length
            # TODO: This needs to incorporate the capacity somehow!
            r = sink_coeff * 2 * np.pi / 3 * info['width'] ** 3 / info['length']
        graph.edges[edge]['resistance'] = r


def commit_fluxes(graph, flux_attrib='flux', time_step=1.0):
    for edge in graph.edges:
        conc_start = graph.nodes[edge[0]]['props'].mass
        conc_end = graph.nodes[edge[1]]['props'].mass
        flux = graph.edges[edge][flux_attrib]


        graph.nodes[edge[0]]['props'].mass = conc_start - flux * time_step
        graph.nodes[edge[1]]['props'].mass = conc_end + flux * time_step



if __name__ == '__main__':

    test_graph_base = [[0,0], [0,1], [-1,2], [1,2]]
    edges = [[0,1], [1,2], [1,3]]
    widths = [0.1, 0.1, 0.1]




    tip_properties = {
        0: RootSource(init_source=5.0),
        2: NodeProperties(capacity=4, zero_sink=-4),
        3: NodeProperties(capacity=4, zero_sink=-4),
    }

    # test_graph_base = [
    #     [0,0],  # Root
    #     [1,1],
    #     [1,2],
    #     [1,3],  # 3: Tip
    #     [1.75,1],
    #     [1.85,1.85],
    #     [1.95,2.70], # 6: tip
    #     [3.87,1],
    #     [4.37,1.5],
    #     [3.87,2.0],
    #     [4.37,2.75], # 10: tip
    #     [1,4]        # 11: tip
    # ]
    #
    # tip_properties = {
    #     0: {
    #         'source': 10,
    #     },
    #     3: {
    #         'source': -2,
    #         'capacity': 10,
    #         'consumption': lambda x: max(0, min(x-2, 1))
    #     },
    #     6: {
    #         'source': -3,
    #         'capacity': 5,
    #         'consumption': lambda x: max(0, min(x-1, 2)),
    #     },
    #     10: {
    #         'source': -2,
    #         'capacity': 8,
    #         'consumption': lambda x: max(0, min((x-2) / 2, 2))
    #     },
    #     11: {
    #         'source': -5,
    #         'capacity': 20,
    #         'consumption': lambda x: max(0, min(x * 2, 4))
    #     }
    # }
    #
    # edges = [
    #     [0,1],
    #     [1,4],
    #     [4,7],
    #     [7,11],
    #     [1,2],
    #     [2,3],
    #     [4,5],
    #     [5,6],
    #     [7,8],
    #     [8,9],
    #     [9,10]
    # ]
    #
    # widths = [0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    graph = nx.DiGraph()
    for node, pt in enumerate(test_graph_base):
        graph.add_node(node, point=pt)
    for edge, width in zip(edges, widths):
        len = np.linalg.norm(np.array(test_graph_base[edge[1]]) - np.array(test_graph_base[edge[0]]))
        graph.add_edge(*edge, width=width, length=len)
    nx.set_node_attributes(graph, tip_properties, 'props')

    for i in range(5):

        # For all intermediate nodes, find their previous edge and add an edge corresponding to the capacity
        intermediate_nodes = [n for n in graph.nodes if graph.out_degree(n) and graph.in_degree(n)]
        for n in intermediate_nodes:
            dummy_node = '{}_s'.format(n)
            graph.add_node(dummy_node)
            graph.add_edge(n, dummy_node)

            last_edge = (list(graph.predecessors(n))[0], n)
            info = graph.edges[last_edge]

            props = graph.nodes[n].get('props', None)
            if props is None:
                props = BranchStorageNodeProperties(info['length'], info['width'], k_vol=0.2, zero_sink=-1)
            graph.nodes[dummy_node]['props'] = props

        compute_resistances(graph)

        import pdb
        pdb.set_trace()

        compute_source_sink_values(graph, 0)
        to_remove = []
        nodes_to_remove = []
        for edge in graph.edges:
            if not isinstance(edge[1], int):
                props = graph.nodes[edge[1]]['props']
                graph.nodes[edge[0]]['props'] = props
                to_remove.append(edge)
                nodes_to_remove.append(edge[1])
        graph.remove_edges_from(to_remove)
        graph.remove_nodes_from(nodes_to_remove)

        for n in graph.nodes:
            graph.nodes[n]['props'].mass = 0
        commit_fluxes(graph)


        visualize_graph(graph)




