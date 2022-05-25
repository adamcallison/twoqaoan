import util
import simulated_annealing
import sa_util
import perm_util
import routing

def mapping_sa(nqubits, hamiltonian_couplings, hardware_couplings, \
    sa_iterations, sa_runs, verbose=False):
    qubit_distances = util.floyd_warshall(nqubits, hardware_couplings, \
        standardize=True, symmetrize=True)
    ham_adj = util.adjacency_matrix(nqubits, hamiltonian_couplings, \
        standardize=True, symmetrize=True)
    extra_inputs = (ham_adj, qubit_distances)
    best_perm, best_cost, costs = simulated_annealing.simulated_annealing(\
        extra_inputs, sa_iterations, sa_runs, \
        sa_util.random_permutation_for_annealing, \
        sa_util.qap_cost_for_annealing, \
        sa_util.permutation_neighbour_for_annealing, \
        sa_util.boltzmann_acceptance_rule, \
        sa_util.temperature_schedule, \
        verbose=verbose)

    if verbose:
        print(f"Pre-map cost: {util.qap_cost(ham_adj, qubit_distances)}")
        ham_adj_perm = perm_util.permute_array(ham_adj, best_perm)
        print(f"Post-map cost: {util.qap_cost(ham_adj_perm, qubit_distances)}")

    return best_perm, best_cost, costs

def route_and_get_sequence(hamiltonian_couplings, hardware_couplings, \
    initial_permutation, runs, verbose=False):
    swaps, perms, inv_perms, nn_gates_collection, routed_all = routing.route(\
        hamiltonian_couplings, hardware_couplings, initial_permutation, runs, \
        verbose=verbose)
    if verbose:
        if routed_all:
            print("All routed!   ")
        else:
            print("Some not routed    ")

    sequence = routing.routed_implementation(swaps, perms, nn_gates_collection)

    return swaps, perms, inv_perms, nn_gates_collection, sequence
