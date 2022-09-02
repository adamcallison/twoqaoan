import numpy as np

def symmetrize_pairs(pairs, standardize=False):
    if standardize:
        pairs = standardize_pairs(pairs)
    sym_pairs = []
    for pair in pairs:
        sym_pairs.append(pair)
        sym_pairs.append((pair[1], pair[0]))
    return sym_pairs

def standardize_pairs(pairs, symmetrize=False):
    s_pairs = []
    for pair in pairs:
        if pair[0] == pair[1]:
            continue
        pair = (min(pair), max(pair))
        if not pair in s_pairs:
            s_pairs.append(pair)
    s_pairs = sorted(s_pairs)
    if symmetrize:
        s_pairs = symmetrize_pairs(s_pairs)
    return s_pairs

def floyd_warshall(n_vertices, edges, standardize=True, symmetrize=True):

    if standardize:
        edges = standardize_pairs(edges)
    if symmetrize:
        edges = symmetrize_pairs(edges)

    edges = symmetrize_pairs(edges, standardize=True)
    distances = np.ndarray((n_vertices, n_vertices), dtype=float)
    distances[:,:] = float('inf')
    for edge in edges:
        distances[edge[0], edge[1]] = 1.0
    np.fill_diagonal(distances, 0.0)
    for k in range(n_vertices):
        for i in range(n_vertices):
            for j in range(n_vertices):
                if distances[i, j] > distances[i, k] + distances[k, j]:
                    distances[i, j] = distances[i, k] + distances[k, j]
    return distances

def qap_cost(hamiltonian_interactions, qubit_map, standardize=False):
    if standardize:
        hamiltonian_interactions = standardize_pairs(hamiltonian_interactions)
    logical_distances = qubit_map.logical_distances
    cost = 0.0
    for q1, q2 in hamiltonian_interactions:
        cost += logical_distances[q1, q2]
    return cost

def _first_available_color(color_list):
    """Return smallest non-negative integer not in the given list of colors."""
    color_set = set(color_list)
    count = 0
    while True:
        if count not in color_set:
            return count
        count += 1

def greedy_graph_color(adjacency):
    color = dict()
    n = adjacency.shape[0]
    for j in range(n):
        neighbours = np.arange(n)[adjacency[j]==1]

        used_neighbour_colors = [\
            color[nbr] for nbr in neighbours if nbr in color\
            ]

        color[j] = _first_available_color(used_neighbour_colors)
    return color

def permute_array(a, perm):
    new_a = np.zeros_like(a)
    n = len(perm)
    for j in range(n):
        for k in range(n):
            new_a[j, k] = a[perm[j], perm[k]]
    return new_a

def invert_permutation(perm):
    perm = np.array(perm)
    tmp = np.empty_like(perm)
    tmp[perm] = np.arange(perm.shape[0])
    return tmp
