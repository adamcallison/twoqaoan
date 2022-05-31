import numpy as np
import twoqaoan.util as util
import twoqaoan.perm_util as perm_util

def _route(hamiltonian_couplings, hardware_couplings, initial_permutation):
    n = len(initial_permutation)
    hamiltonian_couplings = util.standardize_pairs(hamiltonian_couplings)
    hardware_couplings = util.standardize_pairs(hardware_couplings)
    qubit_distances = util.floyd_warshall(n, hardware_couplings, \
        standardize=False, symmetrize=True)

    permutation = initial_permutation.copy()
    inv_permutation = perm_util.invert_permutation(permutation)
    perms = [permutation]

    unrouted_gates = list(hamiltonian_couplings)
    unrouted_gates_phys = perm_util.permute_pairs(unrouted_gates, permutation)
    nn_gates_phys, unrouted_gates_phys = util.nearest_neighbours(\
        unrouted_gates_phys, qubit_distances)
    nn_gates, unrouted_gates = \
        perm_util.permute_pairs(nn_gates_phys, inv_permutation), \
        perm_util.permute_pairs(unrouted_gates_phys, inv_permutation)

    nn_gates_collection = [nn_gates]
    swaps = []
    routed_all = True
    while len(unrouted_gates) > 0:
        unrouted_dists = \
            [qubit_distances[ug[0], ug[1]] for ug in unrouted_gates_phys]
        shortest_dist = min(unrouted_dists)

        closest_gates_phys = [\
            ug for i, ug in enumerate(unrouted_gates_phys) \
            if unrouted_dists[i] == shortest_dist\
            ]

        if len(closest_gates_phys) == 1:
            closest_gate_phys = closest_gates_phys[0]
        else:
            closest_gate_phys = np.random.default_rng().choice(\
                closest_gates_phys)
        q1, q2 = closest_gate_phys

        swaps_phys = [\
            hc for hc in hardware_couplings if ((q1 in hc) or (q2 in hc))\
            ]
        swaps_phys = util.standardize_pairs(swaps_phys, symmetrize=False)

        candidate_permutations = []
        for swap_phys in swaps_phys:
            candidate_permutation = permutation.copy()

            candidate_permutation[swap_phys[0]], \
                candidate_permutation[swap_phys[1]] = \
                candidate_permutation[swap_phys[1]], \
                candidate_permutation[swap_phys[0]]

            candidate_permutations.append(candidate_permutation)

        closest_gate_cand = [\
            perm_util.permute_pair(closest_gate_phys, cand_perm) for \
            cand_perm in candidate_permutations\
            ]
        closer = [\
            qubit_distances[x[0], x[1]] < shortest_dist for \
            x in closest_gate_cand\
            ]

        if np.any(closer):
            swaps_phys = [swap_phys for j, swap_phys in enumerate(swaps_phys) if closer[j]]
            candidate_permutations = [cand_perm for j, cand_perm in enumerate(candidate_permutations) if closer[j]]

        # can probably check here if there is only 1 candidate, and avoid
        # further work if so

        adj = util.adjacency_matrix(n, unrouted_gates)

        candidate_adjs = [\
            perm_util.permute_array(adj, cand_perm) \
            for cand_perm in candidate_permutations\
            ]

        candidate_costs = [\
            util.qap_cost(cand_adj, qubit_distances) \
            for cand_adj in candidate_adjs\
            ]

        min_cost = min(candidate_costs)

        best_candidate_permutations_idx = [\
            i for i, cand_perm in enumerate(candidate_permutations) \
            if candidate_costs[i] == min_cost\
            ]

        best_candidate_permutations = [\
        candidate_permutations[idx] for idx in best_candidate_permutations_idx\
        ]
        # haven't implemented other 2 criteria

        if len(best_candidate_permutations_idx) == 1:
            best_candidate_permutation_idx = best_candidate_permutations_idx[0]
        else:
            best_candidate_permutation_idx = np.random.default_rng().choice(\
                best_candidate_permutations_idx)

        permutation = candidate_permutations[best_candidate_permutation_idx]
        swap_phys = swaps_phys[best_candidate_permutation_idx]
        swaps.append(swap_phys)

        inv_permutation = perm_util.invert_permutation(permutation)
        perms.append(permutation)

        unrouted_gates_phys = perm_util.permute_pairs(unrouted_gates, \
            permutation)
        nn_gates_phys, unrouted_gates_phys = util.nearest_neighbours(\
            unrouted_gates_phys, qubit_distances)

        nn_gates, unrouted_gates = \
            perm_util.permute_pairs(nn_gates_phys, inv_permutation), \
            perm_util.permute_pairs(unrouted_gates_phys, inv_permutation)

        nn_gates_collection.append(nn_gates)

        if len(swaps) == len(hamiltonian_couplings):
            nn_gates_collection[-1] = nn_gates_collection[-1] + unrouted_gates
            unrouted_gates = []
            routed_all = False

        nn_gates_collection = [\
            util.standardize_pairs(nn_gates, symmetrize=False) \
            for nn_gates in nn_gates_collection\
            ]

    return swaps, perms, nn_gates_collection, routed_all

def route(hamiltonian_couplings, hardware_couplings, initial_permutation, \
    runs, verbose=False):
    for run in range(runs):
        if verbose:
            pc = 100*(run+1)/runs
            print(f"{pc:.2f}% done", end="\r")
        swaps_cand, perms_cand, nn_gates_collection_cand, routed_all_cand = \
            _route(hamiltonian_couplings, hardware_couplings, \
            initial_permutation)
        n_swaps_cand = len(swaps_cand)
        if (run == 0) or (n_swaps_cand < n_swaps):
            n_swaps = n_swaps_cand
            swaps = swaps_cand
            perms = perms_cand
            nn_gates_collection = nn_gates_collection_cand
            routed_all = routed_all_cand
    return swaps, perms, nn_gates_collection, routed_all

def routed_implementation(swaps, perms, nn_gates_collection):
    n = perms[0].shape[0]
    instruction = ('map', tuple(perms[0]))
    sequence = [instruction]

    nn_gates = nn_gates_collection[0]
    nn_gates_permuted = perm_util.permute_pairs(nn_gates, perms[0])
    for j, nn_gate in enumerate(nn_gates):
        nn_gate_permuted = nn_gates_permuted[j]
        nn_gate_permuted = (min(nn_gate_permuted), max(nn_gate_permuted))
        instruction = ('hamiltonian_gate', nn_gate, nn_gate_permuted)
        sequence.append(instruction)

    for j, swap in enumerate(swaps):
        instruction = ('swap', swap)
        sequence.append(instruction)
        nn_gates = nn_gates_collection[j+1]
        nn_gates_permuted = perm_util.permute_pairs(nn_gates, perms[j+1])
        for k, nn_gate in enumerate(nn_gates):
            nn_gate_permuted = nn_gates_permuted[k]
            nn_gate_permuted = (min(nn_gate_permuted), max(nn_gate_permuted))
            instruction = ('hamiltonian_gate', nn_gate, nn_gate_permuted)
            sequence.append(instruction)

    instruction = ('unmap', tuple(perms[-1]))
    sequence.append(instruction)
    return sequence
