import numpy as np

from twoqaoan.util import invert_permutation, greedy_graph_color
from twoqaoan.qubit_map import QubitMap

def _pairs_to_linegraph_adjacency(pairs):
    n_pairs = len(pairs)
    mat = np.zeros((n_pairs, n_pairs), dtype=int)
    for j in range(n_pairs-1):
        for k in range(j+1, n_pairs):
            pair1, pair2 = pairs[j], pairs[k]
            if (pair1[0] in pair2) or (pair1[1] in pair2):
                mat[j, k], mat[k, j] = 1, 1
    return mat

def graph_color_schedule(sequence):
    logical_pairs = [el[1] for el in sequence]
    adjmat = _pairs_to_linegraph_adjacency(logical_pairs)
    color = greedy_graph_color(adjmat)
    grouping = {}
    for el_idx, color_idx in color.items():
        try:
            tmp = grouping[color_idx]
        except KeyError:
            tmp = []
        tmp.append(sequence[el_idx])
        grouping[color_idx] = tmp
    schedule = []
    for group in grouping.values():
        schedule += group
    return schedule

def swap_interaction_combine(sequence):
    sequence = list(sequence)
    swaps_idx = [j for j, el in enumerate(sequence) if el[0] == 'swap']
    interactions_idx = [j for j, el in enumerate(sequence) if el[0] == 'interaction']
    unified_interactions_idx = []
    for swap_idx in swaps_idx:
        logical_s = sequence[swap_idx][1]
        for interaction_idx in interactions_idx:
            logical_i = sequence[interaction_idx][1]
            if ((logical_s == logical_i) or (logical_s == (logical_i[1], logical_i[0]))) and (not (interaction_idx in unified_interactions_idx)):
                sequence[swap_idx] = ('swapint',) + sequence[swap_idx][1:]
                unified_interactions_idx.append(interaction_idx)
                break
    sequence = [el for j, el in enumerate(sequence) if not j in unified_interactions_idx]
    return sequence

def schedule_v1(sequence, physical_coupled):
    sequence = list(sequence)
    physical_to_logical_el, logical_to_physical_el, sequence = sequence[-1], sequence[0], sequence[1:-1]
    physical_to_logical, logical_to_physical = physical_to_logical_el[1], logical_to_physical_el[1]
    new_sequence_reversed = [tuple(physical_to_logical)]
    num_qubits = len(physical_to_logical)
    logical_to_physical_final = invert_permutation(physical_to_logical)
    current_map = QubitMap(num_qubits, physical_coupled, logical_to_physical=logical_to_physical_final)
    while len(sequence) > 0:
        eltype, logical, physical = sequence[-1]
        if eltype in ('swap', 'swapint'):
            new_sequence_reversed.append((eltype, logical, physical))
            current_map = current_map.copy().swap(logical[0], logical[1], physical_indices=False)
            sequence = sequence[:-1]
            continue
        logical_distances = current_map.logical_distances
        logical_to_physical_current = current_map.logical_to_physical
        schedule_add, schedule_add_idx = [], []
        for j, el in enumerate(sequence):
            eltype, logical, physical = el
            if (eltype in ('interaction',)) and (logical_distances[logical[0], logical[1]] == 1):
                physical = (logical_to_physical_current[logical[0]], logical_to_physical_current[logical[1]])
                schedule_add.append((eltype, logical, physical))
                schedule_add_idx.append(j)
        new_sequence_reversed += schedule_add
        sequence = [el for j, el in enumerate(sequence) if not j in schedule_add_idx]
    new_sequence_reversed.append(('logical to physical', logical_to_physical))
    new_sequence = new_sequence_reversed[::-1]
    return new_sequence

def schedule_v2(sequence, physical_coupled):
    # same as v1 but with graph coloring
    sequence = list(sequence)
    physical_to_logical_el, logical_to_physical_el, sequence = sequence[-1], sequence[0], sequence[1:-1]
    physical_to_logical, logical_to_physical = physical_to_logical_el[1], logical_to_physical_el[1]
    new_sequence_reversed = [tuple(physical_to_logical)]
    num_qubits = len(physical_to_logical)
    logical_to_physical_final = invert_permutation(physical_to_logical)
    current_map = QubitMap(num_qubits, physical_coupled, logical_to_physical=logical_to_physical_final)
    while len(sequence) > 0:
        eltype, logical, physical = sequence[-1]
        if eltype in ('swap', 'swapint'):
            new_sequence_reversed.append((eltype, logical, physical))
            current_map = current_map.copy().swap(logical[0], logical[1], physical_indices=False)
            sequence = sequence[:-1]
            continue
        logical_distances = current_map.logical_distances
        logical_to_physical_current = current_map.logical_to_physical
        schedule_add, schedule_add_idx = [], []
        for j, el in enumerate(sequence):
            eltype, logical, physical = el
            if (eltype in ('interaction',)) and (logical_distances[logical[0], logical[1]] == 1):
                physical = (logical_to_physical_current[logical[0]], logical_to_physical_current[logical[1]])
                schedule_add.append((eltype, logical, physical))
                schedule_add_idx.append(j)
        schedule_add = graph_color_schedule(schedule_add)
        new_sequence_reversed += schedule_add
        sequence = [el for j, el in enumerate(sequence) if not j in schedule_add_idx]
    new_sequence_reversed.append(('logical to physical', logical_to_physical))
    new_sequence = new_sequence_reversed[::-1]
    return new_sequence

def schedule_v3(sequence):
    # preserve the groups within swaps, but graph color schedule
    sequence = list(sequence)
    physical_to_logical_el, logical_to_physical_el, sequence = sequence[-1], sequence[0], sequence[1:-1]
    new_sequence = []
    current_group = []
    for el in sequence + ['END']:
        if el == 'END' or el[0] in ('swap', 'swapint'):
            new_sequence += graph_color_schedule(current_group)
            current_group = []
            if el[0] in ('swap', 'swapint'):
                new_sequence.append(el)
        elif el[0] in ('interaction',):
            current_group.append(el)
        else:
            raise ValueError
    new_sequence = [logical_to_physical_el] + new_sequence + [physical_to_logical_el]
    return new_sequence