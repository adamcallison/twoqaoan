from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

import twoqaoan.util as util
import twoqaoan.simulated_annealing as simulated_annealing
import twoqaoan.sa_util as sa_util
import twoqaoan.perm_util as perm_util
import twoqaoan.routing as routing

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

def sequence_from_Jmat(Jmat, hardware_couplings, sa_iterations, sa_runs, \
    routing_runs, verbose=False):
    nqubits = Jmat.shape[0]
    hamiltonian_couplings = []
    for j in range(0, nqubits-1):
        for k in range(j+1, nqubits):
            Jcoeff = Jmat[j, k] + Jmat[k, j]
            if np.abs(Jcoeff) > 0:
                hamiltonian_couplings.append((j, k))
    hamiltonian_couplings = util.standardize_pairs(hamiltonian_couplings)
    best_perm, best_cost, costs = mapping_sa(nqubits, hamiltonian_couplings, \
        hardware_couplings, sa_iterations, sa_runs, verbose=verbose)
    swaps, perms, inv_perms, nn_gates_collection, sequence = \
        route_and_get_sequence(hamiltonian_couplings, hardware_couplings, \
        best_perm, routing_runs, verbose=verbose)
    return swaps, perms, inv_perms, nn_gates_collection, sequence

def _circuit_from_hamiltonian(J, h, c, qaoa_param):
    n = h.shape[0]
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)
    for q1 in range(n):
        hval = h[q1]
        if not (hval == 0.0):
            qc.rz(2*qaoa_param*hval, q1)
    for q1 in range(n-1):
        for q2 in range(q1+1, n):
            Jcoeff = J[q1, q2] + J[q2, q1]
            if Jcoeff == 0.0:
                continue
            qc.rzz(2*qaoa_param*Jcoeff, q1, q2)
    if not (c == 0.0):
        qc.global_phase = qc.global_phase - (qaoa_param*c)
    return qc

def _circuit_from_hamiltonian_optimized(J_sequence, J, h, c, qaoa_param):
    n = h.shape[0]
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)

    initial_permutation, inv_initial_permutation = None, None
    if J_sequence[0][0] == 'map':
        initial_permutation = np.array(J_sequence[0][1], dtype=int)
        inv_initial_permutation = perm_util.invert_permutation(\
            initial_permutation)
        h = h.copy()
        J_sequence = J_sequence[1:]

    for q1 in range(n):
        logical_qubit = q1
        if initial_permutation is None:
            physical_qubit = q1
        else:
            physical_qubit = inv_initial_permutation[q1]
        hval = h[physical_qubit]
        if not (hval == 0.0):
            qc.rz(2*qaoa_param*hval, q1)

    final_permutation = None
    for i, instruction in enumerate(J_sequence):
        if instruction[0] == 'swap':
            qc.swap(instruction[1][0], instruction[1][1])
        elif instruction[0] == 'hamiltonian_gate':
            physical_pair = instruction[2]
            logical_pair = instruction[1]
            Jcoeff = J[logical_pair[0], logical_pair[1]] + J[logical_pair[1], \
                logical_pair[0]]
            if Jcoeff == 0:
                continue
            qc.rzz(2*qaoa_param*Jcoeff, physical_pair[0], physical_pair[1])
        elif instruction[0] == 'map':
            raise ValueError("map can only be done as first step")
        elif instruction[0] == 'unmap':
            if not (i == (len(J_sequence)) - 1):
                raise ValueError("unmap can only be done as last step")
            else:
                final_permutation = np.array(instruction[1])

    if not (c == 0.0):
        qc.global_phase = qc.global_phase - (qaoa_param*c)
    return qc, initial_permutation, final_permutation

def circuit_from_hamiltonian(J, h, c, qaoa_param=None, \
    hardware_couplings=None, sa_iterations=None, sa_runs=None, \
    routing_runs=None, optimize=True, verbose=False):
    if qaoa_param is None:
        qaoa_param = 1.0
    if not optimize:
        qc = _circuit_from_hamiltonian(J, h, c, qaoa_param)
    else:
        swaps, perms, inv_perms, nn_gates_collection, sequence = \
            sequence_from_Jmat(J, hardware_couplings, sa_iterations, sa_runs, \
            routing_runs, verbose=verbose)
        qc = _circuit_from_hamiltonian_optimized(sequence, J, h, c, qaoa_param)
    return qc
