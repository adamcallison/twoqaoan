import numpy as np

from twoqaoan.util import standardize_pairs, qap_cost

def _route(hamiltonian_interactions, initial_map):
    #physical_coupled = initial_map.physical_coupled
    nn_gates_collection = []
    swaps = []
    unrouted_gates = list(hamiltonian_interactions)
    
    unrouted_gates_new, nn_gates = [], []
    logical_coupled = initial_map.logical_coupled
    for ur_gate in unrouted_gates:
        if ur_gate in logical_coupled:
            nn_gates.append(ur_gate)
        else:
            unrouted_gates_new.append(ur_gate)
    unrouted_gates = unrouted_gates_new
    nn_gates_collection.append(nn_gates)
    qubit_maps = [initial_map.copy()]

    routed_all = None
    while len(unrouted_gates) > 0:
        
        logical_distances = qubit_maps[-1].logical_distances
        ur_gate_distances = [logical_distances[ur_gate[0], ur_gate[1]] for ur_gate in unrouted_gates]
        shortest_distance = np.min(ur_gate_distances)
        closest_ur_gates = [ur_gate for j, ur_gate in enumerate(unrouted_gates) if ur_gate_distances[j] == shortest_distance]
        
        #closest_ur_gates = [np.random.default_rng().choice(closest_ur_gates)] #hack for checking something
        
        candidate_swaps = []
        for cug in closest_ur_gates:
            q1 = cug[0]
            new_candidate_swaps = [(q1, j) for j, x in enumerate(logical_distances[q1]) if x == 1]
            candidate_swaps += new_candidate_swaps
            q2 = cug[0]
            new_candidate_swaps = [(q2, j) for j, x in enumerate(logical_distances[q2]) if x == 1]
            candidate_swaps += new_candidate_swaps
        candidate_swaps = standardize_pairs(candidate_swaps)
        candidate_maps = [qubit_maps[-1].copy().swap(*candidate_swap, physical_indices=False) for candidate_swap in candidate_swaps]
        candidate_swap_costs = [qap_cost(unrouted_gates, qm) for qm in candidate_maps]
        lowest_swap_cost = np.min(candidate_swap_costs)
        best_swaps_idx = [j for j, swap in enumerate(candidate_swaps) if candidate_swap_costs[j] == lowest_swap_cost]
        
        closer_all = []
        for best_swap_idx in best_swaps_idx:
            closer = [candidate_maps[best_swap_idx].logical_distances[cug[0], cug[1]] < shortest_distance for cug in closest_ur_gates]
            closer_all.append(np.any(closer))
        if np.any(closer_all):
            best_swaps_idx = [best_swap_idx for j, best_swap_idx in enumerate(best_swaps_idx) if closer_all[j]]
            
        # other criteria not implemented
        
        # for criteria 2, can i just insert the swap that has been NN for the longest?? NOOO
        
        # do i need to build up dependency graph??
        
        # does criteria 3 do anything that a simple pass later on can't?
        
        # for 2, try just selecting swap that is furthest from the previous one:
        #if (not (len(best_swaps_idx) == 1)) and (not (len(swaps) == 0)):
        if False: # did not help!
            logical_distances = qubit_maps[-1].logical_distances
            previous_swap = swaps[-1]
            swap_dists = [
                np.min((
                logical_distances[candidate_swaps[best_swap_idx][0], previous_swap[0]],
                logical_distances[candidate_swaps[best_swap_idx][0], previous_swap[1]],
                logical_distances[candidate_swaps[best_swap_idx][1], previous_swap[0]],
                logical_distances[candidate_swaps[best_swap_idx][1], previous_swap[1]]
                )) for best_swap_idx in best_swaps_idx
            ]
            #tmp = len(best_swaps_idx)
            best_swaps_idx = [best_swap_idx for j, best_swap_idx in enumerate(best_swaps_idx) if swap_dists[j] == np.max(swap_dists)]
            #print(tmp, len(best_swaps_idx))
                          
        if len(best_swaps_idx) == 1:
            best_swap_idx = best_swaps_idx[0]
        else:
            best_swap_idx = np.random.default_rng().choice(best_swaps_idx)
        best_swap, best_map = candidate_swaps[best_swap_idx], candidate_maps[best_swap_idx]
        
        unrouted_gates_new, nn_gates = [], []
        logical_coupled = best_map.logical_coupled
        for ur_gate in unrouted_gates:
            if ur_gate in logical_coupled:
                nn_gates.append(ur_gate)
            else:
                unrouted_gates_new.append(ur_gate)
        unrouted_gates = unrouted_gates_new
        nn_gates_collection.append(nn_gates)
        swaps.append(best_swap)
        qubit_maps.append(best_map)
        
        if len(swaps) >= len(hamiltonian_interactions):
            routed_all = False
            nn_gates_collection[-1] = nn_gates_collection[-1] + unrouted_gates
            break
            
    if routed_all is None:
        routed_all = True
    return swaps, nn_gates_collection, qubit_maps, routed_all

def route(hamiltonian_interactions, initial_map, attempts=1, verbose=False):
    nswaps_best, routed_all_best = float('inf'), False
    for j in range(attempts):
        if verbose:
            print(f"Attempt {j+1} of {attempts}.    ", end="\r")
        swaps, nn_gates_collection, qubit_maps, routed_all = _route(hamiltonian_interactions, initial_map)
        nswaps = len(swaps)
        if nswaps < nswaps_best or (routed_all and (not routed_all_best)):
            nswaps_best = nswaps
            swaps_best, nn_gates_collection_best, qubit_maps_best, routed_all_best = swaps, nn_gates_collection, qubit_maps, routed_all
            
    if verbose:
        print(f"{nswaps_best} swaps used and {'all gates' if routed_all_best else 'some gates not'} routed.")
    return swaps_best, nn_gates_collection_best, qubit_maps_best, routed_all_best

def route_postprocess(initial_map, swaps, nn_gates_collection, verbose=False):
    assert len(swaps) == len(nn_gates_collection) - 1
    sequence = []
    current_map = initial_map.copy()
    for j, nn_gates in enumerate(nn_gates_collection):
        logical_to_physical = current_map.logical_to_physical
        sequence_new = [("interaction", x, (logical_to_physical[x[0]], logical_to_physical[x[1]])) for x in nn_gates]
        if not j == len(nn_gates_collection) - 1:
            swap = swaps[j]
            sequence_new.append(("swap", swap, (logical_to_physical[swap[0]], logical_to_physical[swap[1]])))
            current_map.swap(*swap, physical_indices=False)
        sequence += sequence_new
    final_map = current_map.copy()
    
    connected = 0
    physical_coupled = initial_map.physical_coupled
    for el in sequence:
        if ((el[2][0], el[2][1]) in physical_coupled) or ((el[2][1], el[2][0]) in physical_coupled):
            connected += 1
    all_connected = (connected == len(sequence))
    if verbose:
        pc = 100*connected/len(sequence)
        print(f"{pc}% of gates are physically coupled.")
        
    logical_to_physical_initial = initial_map.logical_to_physical
    physical_to_logical_final = final_map.physical_to_logical
    
    sequence = [('logical to physical', tuple(logical_to_physical_initial))] + sequence + [('physical to logical', tuple(physical_to_logical_final))]
        
    return sequence, final_map, all_connected
