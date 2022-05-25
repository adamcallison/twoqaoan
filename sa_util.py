import numpy as np
import util
import perm_util

def random_permutation_for_annealing(annealing_inputs):
    n_vertices = annealing_inputs[0].shape[0]
    permutation = np.random.default_rng().permutation(n_vertices)
    return permutation

def qap_cost_for_annealing(annealing_inputs, permutation):
    hamiltonian_adjacency, qubit_distances = annealing_inputs
    hamiltonian_adjacency = perm_util.permute_array(hamiltonian_adjacency, permutation)
    return util.qap_cost(hamiltonian_adjacency, qubit_distances)

def permutation_neighbour_for_annealing(annealing_inputs, current_permutation):
    n_vertices = annealing_inputs[0].shape[0]
    swap_pair = np.random.default_rng().choice(n_vertices, size=2, replace=False)
    new_permutation = current_permutation.copy()
    new_permutation[swap_pair[0]], new_permutation[swap_pair[1]] = new_permutation[swap_pair[1]], new_permutation[swap_pair[0]]
    return new_permutation

def boltzmann_acceptance_rule(current_cost, candidate_cost, temperature):
    if candidate_cost <= current_cost:
        accept = True
    else:
        acceptance_probability = np.exp(-(candidate_cost-current_cost)/temperature)
        test_probability = np.random.default_rng().uniform()
        accept = test_probability <= acceptance_probability
    return accept

def temperature_schedule(annealing_inputs, iterations, iteration):
    T_max = 10.0
    scale = np.log((iterations+1)/(iteration+1))/np.log(iterations+1)
    T = T_max*scale
    return T
