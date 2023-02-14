from stochastic_tree import BasicWood
import numpy as np
import networkx as nx
import openalea.lpy as lpy
import openalea.plantgl.all as plantgl
import functools
from openalea.plantgl.scenegraph.cspline import CSpline
from abc import ABC, abstractmethod

class LStringGraphDual:
    def __init__(self, graph: nx.DiGraph):

        self.validate_graph(graph)
        self.graph = graph
        self.info = self.extract_graph_metadata(self.graph)
        self._branches = None

    @property
    def branches(self):
        if self._branches is None:
            self._branches = self.extract_branches()
        return self._branches

    def validate_graph(self, graph):
        assert isinstance(graph, nx.DiGraph)

    def extract_graph_metadata(self, graph=None):
        if graph is None:
            graph = self.graph

        info = {}

        # Find root - Only node with no predecessors and 1 successor
        roots = [n for n in graph.nodes if len(list(graph.predecessors(n))) == 0 and len(list(graph.successors(n))) > 0]
        assert len(roots) == 1
        info['root'] = roots[0]

        return info

    def extract_branches(self):
        bids = {self.graph.edges[e]['branch_id'] for e in self.graph.edges}
        all_branches = {}
        for bid in bids:
            all_edges = [e for e in self.graph.edges if self.graph.edges[e]['branch_id'] == bid]
            starts, ends = zip(*all_edges)
            starts = set(starts)
            ends = set(ends)
            start_nodes = starts - ends
            end_nodes = ends - starts
            assert len(start_nodes) == 1 and len(end_nodes) == 1
            mapping = dict(all_edges)
            node = list(start_nodes)[0]
            branch_nodes = []
            while node is not None:
                branch_nodes.append(node)
                node = mapping.get(node, None)


            # Extract metadata for each branch
            data = {}
            data['nodes'] = branch_nodes

            # Length
            pts = np.array([self.graph.nodes[n]['position'] for n in branch_nodes if isinstance(n, int)])
            data['points'] = pts
            data['length'] = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
            all_branches[bid] = data

        # Get branch generation
        @functools.lru_cache(maxsize=None)
        def gen_search(bid):
            start_node = all_branches[bid]['nodes'][0]
            preds = list(self.graph.predecessors(start_node))
            if not preds:
                return 0
            else:
                return gen_search(self.graph.edges[preds[0], start_node]['branch_id']) + 1

        for bid in all_branches:
            all_branches[bid]['generation'] = gen_search(bid)

        return all_branches


    def search_branches(self, attrib, cond, value, assert_unique=False):
        if cond == '<':
            func = lambda x: x[attrib] < value
        elif cond == '=':
            func = lambda x: x[attrib] == value
        elif cond == '>':
            func = lambda x: x[attrib] > value
        else:
            raise ValueError()
        bids = [bid for bid in self.branches if func(self.branches[bid])]
        if assert_unique:
            assert len(bids) == 1
            return bids[0]
        return bids


    def set_guide_on_nodes(self, nodes, points, guide_module='SetGuide'):
        length = np.sum(np.linalg.norm(points[:-1] - points[1:], axis=1))
        diff = np.linalg.norm(self.graph.nodes[nodes[0]]['position'] - points[0])
        if diff > 1e-3:
            print('[WARNING] Attempting to set a curve guide that is not close to the recorded start.')
        for i, edge in enumerate(zip(nodes[:-1], nodes[1:])):
            if i == 0:
                self.graph.edges[edge]['pre_turtle_modules'] = [(guide_module, [CSpline(points).curve(), length])]
                # print('[{}] {}'.format(node, self.graph.nodes[node]['pre_turtle_modules']))
            else:
                self.graph.edges[edge]['pre_turtle_modules'] = []

            if i == len(nodes) - 2:
                self.graph.edges[edge]['post_turtle_modules'] = [('EndGuide', [])]
                self.graph.edges[edge]['post_modules'].append(('Flags', [1]))
            else:
                self.graph.edges[edge]['post_turtle_modules'] = []

    def set_node_attributes(self, mapping, name=None):
        nx.set_node_attributes(self.graph, mapping, name=name)

    def stub_branch(self, nodes, stub_len, replacement_module='S', save_pre_modules=True, save_post_modules=False):

        cumul = 0
        for i, edge in enumerate(zip(nodes[:-1], nodes[1:])):

            pt_1 = np.array(self.graph.nodes[edge[0]]['position'])
            pt_2 = np.array(self.graph.nodes[edge[1]]['position'])
            diff = pt_2 - pt_1
            dist = np.linalg.norm(diff)

            if cumul < stub_len <= cumul + dist:
                to_cut = stub_len - cumul
                if not save_pre_modules:
                    self.graph.edges[edge]['pre_modules'] = []
                if not save_post_modules:
                    self.graph.edges[edge]['post_modules'] = []

                self.graph.edges[edge]['post_modules'] = []
                self.graph.edges[edge]['args'][0] = to_cut
                self.graph.nodes[edge[1]]['name'] = replacement_module

                for succ in self.graph.successors(edge[1]):
                    self.graph.remove_edge(edge[1], succ)

                return True
            cumul += dist
        return False


    def to_lstring(self):

        def explore(edge):

            queued_modules = []
            start_bid = self.graph.edges[edge]['branch_id']

            while edge is not None:

                data = self.graph.edges[edge]
                queued_modules.extend(data.get('pre_turtle_modules', []))
                queued_modules.extend(data.get('pre_modules', []))
                queued_modules.append((data['name'], data['args']))
                queued_modules.extend(data.get('post_modules', []))
                queued_modules.extend(data.get('post_turtle_modules', []))

                node_info = self.graph.nodes[edge[1]]
                if node_info['name']:
                    queued_modules.append((node_info['name'], node_info.get('args', [])))

                next_edge = None
                for node in self.graph.successors(edge[1]):
                    new_edge = (edge[1], node)
                    bid = self.graph.edges[new_edge]['branch_id']
                    if bid == start_bid:
                        next_edge = new_edge
                    else:
                        queued_modules.append(('[', ()))
                        queued_modules.extend(explore(new_edge))
                        queued_modules.append((']', ()))

                edge = next_edge
            return queued_modules

        modules = []
        root = self.info['root']
        root_info = self.graph.nodes[root]
        modules.append((root_info['name'], root_info.get('args', [])))

        first_edge = (root, list(self.graph.successors(root))[0])
        modules.extend(explore(first_edge))
        lstring = lpy.Lstring()
        for module_name, module_args in modules:
            module = lpy.newmodule(module_name, *module_args)
            lstring.append(module)

        return lstring

    @classmethod
    def from_lstring(cls, lstring, node_modules, internode_modules, terminal_modules=None, dir_modules=None):

        """
        This function takes in an LString and turns it into a graph format.
        In order to obtain the same LString after calling to_lstring(), the LString must obey these conventions:
        - Without brackets, the format is (Node)(Pre-internode mods)(Internode)(Post-internode mods)(Node)
        - With brackets, the format is (Node)[(Pre-internode mods)(Internode)...(Post-internode mods)(Node)]
        If these conventions are not followed exactly, certain elements may be swap order and lead to unusual results.
        Note that if there are multiple splits at a single node, the order is not guaranteed to be deterministic.
        However the L-string should represent the exact same physical system.

        Terminal modules: In some cases, you may wish to have a module that represents a combination of an internode
        and a node, e.g. B = a bud, L = a leaf, etc. In this case, these modules are considered to be TERMINAL;
        nothing should follow it. Therefore a terminal module should always be immediately followed by a ].
        """

        if terminal_modules is None:
            terminal_modules = []

        turtle_modules = ['^', '&', 'Left', 'Right', 'Up', 'Down', 'RollL', 'RollR'] + (dir_modules or [])

        final_graph = nx.DiGraph()

        last_node = None

        queued_modules = []
        queued_turtle_modules = []
        queued_internode = {}

        stack = []
        bid_counter = Counter()
        bid = bid_counter()

        # TODO: Write out the directional modifiers separately

        for i, module in enumerate(lstring):
            name = module.name
            if name in node_modules:
                module_id = module.args[0]
                assert module_id not in final_graph
                final_graph.add_node(module_id, name=name, args=module.args)
                if last_node is not None:
                    queued_internode['post_modules'] = queued_modules
                    queued_internode['post_turtle_modules'] = queued_turtle_modules
                    final_graph.add_edge(last_node, module_id, **queued_internode)
                    queued_internode = {}
                    queued_modules = []
                    queued_turtle_modules = []

                last_node = module_id
            elif name in internode_modules:
                queued_internode = {'name': name, 'args': module.args, 'branch_id': bid, 'pre_modules': queued_modules,
                                    'pre_turtle_modules': queued_turtle_modules}
                queued_modules = []
                queued_turtle_modules = []

            elif name in turtle_modules:
                queued_turtle_modules.append((name, module.args))

            elif name in terminal_modules:
                assert not queued_internode

                dummy_node = 'null_{}'.format(id(module))
                final_graph.add_node(dummy_node, name='', args=[])
                final_graph.add_edge(last_node, dummy_node, name=name, args=module.args, pre_modules=queued_modules,
                                     pre_turtle_modules=queued_turtle_modules, branch_id=bid)
                queued_modules = []
                queued_turtle_modules = []
                bid = bid_counter()

            elif name == '[':
                assert not queued_internode
                stack.append((last_node, bid))
                bid = bid_counter()
            elif name == ']':
                last_node, bid = stack.pop()
            else:
                queued_modules.append((name, module.args))

        return cls(final_graph)


class Converter:

    def serialize(self, module):
        return module.name, module.args

    def deserialize(self, name, args):
        module = lpy.newmodule(name, *args)
        return module


class SetGuideSplineConverter(Converter):
    def serialize(self, module):
        assert module.name == 'SetGuide'
        info = {'points': module.args[0].ctrlPointList}
        try:
            info['length'] = module.args[1]
        except IndexError:
            pass
        return module.name, info

    def deserialize(self, name, kwargs):
        assert name == 'SetGuide'


class Counter:
    def __init__(self):
        self.val = 0

    def __call__(self, *args, **kwargs):
        cur_val = self.val
        self.val += 1
        return cur_val


if __name__ == '__main__':
    import pickle
    with open(r'D:\Documents\Temp\test.pickle', 'rb') as fh:
        tree = pickle.load(fh)

    tree.branches
    import pdb
    pdb.set_trace()

    tree.to_lstring()


    # import matplotlib.pyplot as plt
    # curve = OGH([0, 0], [1, 1], [5, 0], [1, 0])
    # pts = curve.eval()
    #
    # plt.plot(pts[:,0], pts[:,1])
    # plt.axis('equal')
    # plt.show()
    #
    # curve = OGH([0, 0, 0], [1, 0, 0], [3, 3, 3], [0, 1, 0])
    # pts = curve.eval()
    #
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(pts[:,0], pts[:,1], pts[:,2])
    # plt.show()
    #
    # curve_1 = OGH([0, 0], [1, 1], [5, 0], [1, 0])
    # curve_2 = OGH([0, 0], [1, 1], [10, 0], [1, 0])
    # curve_3 = OGH([0, 0], [1, 1], [5, 0], [-1, -1])
    #
    # print(f'Curve 1:\n\tStrain: {curve_1.strain}\n\tLength: {curve_1.approx_len}')
    # print(f'Curve 2:\n\tStrain: {curve_2.strain}\n\tLength: {curve_2.approx_len}')
    # print(f'Curve 3:\n\tStrain: {curve_3.strain}\n\tLength: {curve_3.approx_len}')
    #
    # print()
    #
