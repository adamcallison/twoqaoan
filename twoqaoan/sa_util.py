import numpy as np
from twoqaoan.qubit_map import QubitMap
from twoqaoan.util import qap_cost

def random_map_for_annealing(annealing_inputs):
    num_qubits, hamiltonian_interactions, physical_coupled = annealing_inputs
    logical_to_physical = np.random.default_rng().permutation(num_qubits)
    qubit_map = QubitMap(num_qubits, physical_coupled, logical_to_physical=logical_to_physical)
    return qubit_map

def qap_cost_for_annealing(annealing_inputs, qubit_map):
    num_qubits, hamiltonian_interactions, physical_coupled = annealing_inputs
    return qap_cost(hamiltonian_interactions, qubit_map)

def map_neighbour_for_annealing(annealing_inputs, current_map):
    num_qubits, hamiltonian_interactions, physical_coupled = annealing_inputs
    swap_pair = np.random.default_rng().choice(num_qubits, size=2, replace=False)
    new_map = current_map.copy()
    new_map.swap(swap_pair[0], swap_pair[1], physical_indices=False)

    return new_map

def boltzmann_acceptance_rule(current_cost, candidate_cost, temperature):
    if candidate_cost <= current_cost:
        accept = True
    else:

        acceptance_probability = \
            np.exp(-(candidate_cost-current_cost)/temperature)

        test_probability = np.random.default_rng().uniform()
        accept = test_probability <= acceptance_probability
    return accept

def temperature_schedule(annealing_inputs, iterations, iteration):
    T_max = 10.0
    scale = np.log((iterations+1)/(iteration+1))/np.log(iterations+1)
    T = T_max*scale
    return T

