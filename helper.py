from openalea.plantgl.all import NurbsCurve
from openalea.lpy import Lsystem, newmodule
from random import uniform
import numpy as np


def dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def get_energy_mat(branches, arch):
    num_branches = len(branches)
    num_wires = len(list(arch.branch_supports.values()))
    energy_matrix = np.ones((num_branches, num_wires)) * np.inf
    for branch_id, branch in enumerate(branches):
        if branch.has_tied:
            continue
        for wire_id, wire in arch.branch_supports.items():
            if wire.num_branch >= 1:
                continue
            energy_matrix[branch_id][wire_id] = (dist_sq(wire.point, branch.end) + dist_sq(wire.point, branch.start)) / 2

    return energy_matrix


def decide_guide(energy_matrix, branches, arch, max_energy=None):

    if not energy_matrix.size:
        return

    i_min, j_min = np.unravel_index(energy_matrix.argmin(), energy_matrix.shape)
    if np.isinf(energy_matrix[i_min, j_min]) or (max_energy and energy_matrix[i_min, j_min] > max_energy):
        return

    if not branches[i_min].has_tied:
        branches[i_min].guide_target = arch.branch_supports[j_min]
        energy_matrix[i_min,:] = energy_matrix[:, j_min] = np.inf


def cut_from(pruning_id, s, path=None):
    """Check cut_string_from_manipulation for manual implementation"""
    s.insertAt(pruning_id, newmodule('F'))
    s.insertAt(pruning_id + 1, newmodule('%'))
    return s


# Test utilities

def amplitude(x): return 2


def cut_using_string_manipulation(pruning_id, s, path=None):
    """Cuts starting from index pruning_id until branch
          end signified by ']' or the entire subtrees if pruning_id starts from leader"""
    bracket_balance = 0
    cut_num = pruning_id
    s[cut_num].append("no cut")
    cut_num += 1
    pruning_id += 1
    total_length = len(s)
    while (pruning_id < total_length):
        if s[cut_num].name == '[':
            bracket_balance += 1
        if s[cut_num].name == ']':
            if bracket_balance == 0:
                break
            else:
                bracket_balance -= 1
        del s[cut_num]
        pruning_id += 1  # Insert new node cut at the end of cut
    if path != None:
        new_lsystem = Lsystem(path)  # Figure out to include time in this
        new_lsystem.axiom = s
        return new_lsystem
    s.insertAt(cut_num, newmodule("I(1, 0.05)"))
    return s


def pruning_strategy(it, lstring):
    if ((it + 1) % 8 != 0):
        return lstring
    cut = False
    curr = 0
    while curr < len(lstring):
        if lstring[curr] == '/':
            if not (angle_between(lstring[curr].args[0], 0, 50) or angle_between(lstring[curr].args[0], 130, 180)):
                if (len(lstring[curr].args) > 1):
                    if lstring[curr].args[1] == "no cut":
                        curr += 1
                        continue

                print("Cutting", curr, lstring[curr], (lstring[curr].args[0] + 180))
                # lstring[curr].append("no cut")
                lstring = cut_from(curr + 1, lstring)
        elif lstring[curr] == '&':
            if not (angle_between(lstring[curr].args[0], 0, 50) or angle_between(lstring[curr].args[0], 130, 180)):
                if (len(lstring[curr].args) > 1):
                    if lstring[curr].args[1] == "no cut":
                        curr += 1
                        continue
                print("Cutting", curr, lstring[curr], (lstring[curr].args[0] + 180))
                # lstring[curr].append("no cut")
                lstring = cut_from(curr + 1, lstring)
        curr += 1

    return lstring


def angle_between(angle, min, max):
    angle = (angle + 90)
    if angle > min and angle < max:
        return True
    return False


def myrandom(radius):
    return uniform(-radius, radius)


def gen_noise_branch(radius, nbp=20):
    return NurbsCurve(
        [(0, 0, 0, 1), (0, 0, 1 / float(nbp - 1), 1)] + [(myrandom(radius * amplitude(pt / float(nbp - 1))),
                                                          myrandom(radius * amplitude(pt / float(nbp - 1))),
                                                          pt / float(nbp - 1), 1) for pt in range(2, nbp)],
        degree=min(nbp - 1, 3), stride=nbp * 100)


def ed(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def get_energy_mat(branches, arch):
    num_branches = len(branches)
    num_wires = len(list(arch.branch_supports.values()))
    energy_matrix = np.ones((num_branches, num_wires)) * np.inf
    # print(energy_matrix.shape)
    for branch_id, branch in enumerate(branches):
        if branch.has_tied:
            continue
        for wire_id, wire in arch.branch_supports.items():
            if wire.num_branch >= 1:
                continue
            energy_matrix[branch_id][wire_id] = ed(wire.point, branch.end) / 2 + ed(wire.point,
                                                                                    branch.start) / 2  # +v.num_branches*10+branch.bend_energy(deflection, curr_branch.age)
    return energy_matrix