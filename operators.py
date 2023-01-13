from stochastic_tree import BasicWood
import numpy as np
import networkx as nx
import openalea.lpy as lpy

def extract_skeleton_from_lstring(lstring, node_modules, internode_modules):

    final_graph = nx.DiGraph()

    last_node = None
    queued_edge = {}

    stack = []
    for module in lstring:
        name = module.name
        if name in node_modules:
            module_id = module.args[0]
            assert module_id not in final_graph
            final_graph.add_node(module_id, name=name, attribs=module.args[1:])
            if last_node is not None:
                final_graph.add_edge(last_node, module_id, **queued_edge)
            last_node = module_id
        elif name in internode_modules:
            queued_edge = {'name': name, 'attribs': module.args}
        elif name == '[':
            stack.append(last_node)
        elif name == ']':
            last_node = stack.pop()

    return final_graph

def insert_modules(lstring, module_to_insert, node_pairs, node_modules, at_end=False):

    """
    modules_to_insert: A module type to insert (currently as a string)
    node_pairs: A list of directed pairs of nodes for the insertion (e.g. [(3,155), (20,5)])
    at_end: If True, will place module right before the child node. Otherwise, will place right before parent node.
    """

    idx = 0
    last_node_and_idx = (None, None)
    split_idx = {}
    stack = []      # (Previous parent node, previous parent idx)

    while idx < len(lstring):
        module = lstring[idx]
        name = module.name
        if name in node_modules:
            node_id = module.args[0]
            last_node_id, last_node_idx = last_node_and_idx
            edge = (last_node_id, node_id)
            if (last_node_id, node_id) in node_pairs:
                new_module = lpy.newmodule(module_to_insert)
                if at_end:
                    lstring.insertAt(idx, new_module)
                else:
                    last_split_idx = split_idx.get(last_node_id)
                    lstring.insertAt((last_split_idx or last_node_idx) + 1, new_module)
                idx += 1
            last_node_and_idx = (node_id, idx)

        elif name == '[':
            stack.append(last_node_and_idx)
            split_idx[last_node_and_idx[0]] = idx
        elif name == ']':
            last_node_and_idx = stack.pop()

        idx += 1

















def tie_at(lstring, idx, guide, depth=1):
    """
    Given an L-String, the index of the branch to be guided, and a curve to be followed,
    update the existing guide.
    """

    # Check if there is a rotation/existing guide following the current object
    # If so, remove it
    for _ in range(depth):
        if lstring[idx + 1].name in ['&', '/', 'SetGuide']:
            del lstring[idx + 1]

    lstring.insertAt(idx + 1, guide)



def tie_all(lstring):
    for idx, lsys_elem in enumerate(lstring):
        try:
            wood = lsys_elem[0].type
        except (TypeError, AttributeError, IndexError):
            continue

        if not isinstance(wood, BasicWood) or not wood.can_tie:
            continue

        # If an element has not had its tie updated, we continue to let it grow out
        if not wood.tie_updated or not wood.guide_points:
            continue

        guide = wood.produce_tie_guide()
        wood.tie_updated = False

        depth = 1
        if not wood.has_tied:
            depth = 2
            wood.has_tied = True

        tie_at(lstring, idx, guide, depth=depth)

        return True
    return False


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # curve = OGH([0, 0], [1, 1], [5, 0], [1, 0])
    # pts = curve.eval()
    #
    # plt.plot(pts[:,0], pts[:,1])
    # plt.axis('equal')
    # plt.show()
    #
    curve = OGH([0, 0, 0], [1, 0, 0], [3, 3, 3], [0, 1, 0])
    pts = curve.eval()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(pts[:,0], pts[:,1], pts[:,2])
    plt.show()

    curve_1 = OGH([0, 0], [1, 1], [5, 0], [1, 0])
    curve_2 = OGH([0, 0], [1, 1], [10, 0], [1, 0])
    curve_3 = OGH([0, 0], [1, 1], [5, 0], [-1, -1])

    print(f'Curve 1:\n\tStrain: {curve_1.strain}\n\tLength: {curve_1.approx_len}')
    print(f'Curve 2:\n\tStrain: {curve_2.strain}\n\tLength: {curve_2.approx_len}')
    print(f'Curve 3:\n\tStrain: {curve_3.strain}\n\tLength: {curve_3.approx_len}')

    print()

