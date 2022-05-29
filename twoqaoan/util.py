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

def adjacency_matrix(n_vertices, edges, standardize=True, symmetrize=True):

    if standardize:
        edges = standardize_pairs(edges)
    if symmetrize:
        edges = symmetrize_pairs(edges)

    edges = symmetrize_pairs(edges, standardize=True)
    adj = np.zeros((n_vertices, n_vertices), dtype=float)
    for edge in edges:
        adj[edge[0], edge[1]] = 1.0
    return adj

def qap_cost(hamiltonian_adjacency, qubit_distances):
    return np.sum(hamiltonian_adjacency*qubit_distances)

def nearest_neighbours(hamiltonian_couplings, qubit_distances):
    nn_pairs, non_nn_pairs = [], []
    for pair in hamiltonian_couplings:
        if qubit_distances[pair[0], pair[1]] == 1.0:
            nn_pairs.append(pair)
        else:
            non_nn_pairs.append(pair)
    return nn_pairs, non_nn_pairs
