import numpy as np
import twoqaoan.twoqaoan as twoqaoan
import twoqaoan.util as util
import twoqaoan.mapping as mapping
import twoqaoan.routing as routing
from qiskit.test.mock import FakeLondon, FakeTokyo
from qiskit.quantum_info import Statevector

def test_circuit_from_hamiltonian():
    diff = _test_circuit_from_hamiltonian()
    #print(diff)
    assert diff < 1e-10

def _test_circuit_from_hamiltonian(fakehardware='london', routing_runs=100, \
    verbose=False):
    #     logic separated from the main test function so it can be easily run
    # without pytest
    fakehardware = fakehardware.lower()
    if fakehardware == 'london':
        ft = FakeLondon()
        n = 5

    if fakehardware == 'tokyo':
        ft = FakeTokyo()
        n = 20

    couplings = []
    for gate in ft.properties().gates:
        qubits = gate.qubits
        if not len(qubits) == 2:
            continue
        qubits = tuple(np.sort(qubits))
        if not qubits in couplings:
            couplings.append(qubits)

    J = np.random.default_rng().normal(loc=0.0, scale=1.0, size=(n, n))
    h = J.diagonal().copy()
    np.fill_diagonal(J, 0.0)
    c = np.random.default_rng().normal(loc=0.0, scale=1.0)
    for j in range(0, n-1):
        for k in range(j+1, n):
            rn = np.random.default_rng().uniform()
            if rn > 0.5:
                J[j, k] = J[k, j] = 0.0

    n = h.shape[0]
    ham_pairs = []
    for j in range(0, n-1):
        for k in range(j+1, n):
            if np.abs(J[j, k]+J[k, j]) > 0.0:
                ham_pairs.append((j, k))

    qc = twoqaoan.circuit_from_hamiltonian(J, h, c, qaoa_param=1.0)

    iterations, runs, attempts = 1000, 5, 10
    best_map, best_cost, best_costs = mapping.map_by_annealing(n, ham_pairs, couplings, iterations, runs, verbose=verbose)
    swaps, nn_gates_collection, qubit_maps, routed_all = routing.route(ham_pairs, best_map, attempts=attempts, verbose=False)
    J_sequence, final_map, all_connected = routing.route_postprocess(best_map, swaps, nn_gates_collection, verbose=False)

    qc_opt, logical_to_physical, physical_to_logical = twoqaoan.circuit_from_hamiltonian_optimized(J_sequence, J, h, c, qaoa_param=1.0)

    for j in range(n):
        qc.rx(1.0, j)
        qc_opt.rx(1.0, j)

    physical_to_logical_initial = util.invert_permutation(logical_to_physical)
    logical_to_physical_final = util.invert_permutation(physical_to_logical)

    N = 2**n
    test_amps = np.random.default_rng().uniform(size=N)
    test_phases = np.random.default_rng().uniform(size=N)*2*np.pi
    test_state = test_amps*test_phases
    test_state = test_state/np.sqrt(np.dot(test_state, test_state))
    test_state_perm = util.permute_state(test_state, physical_to_logical_initial)
    #test_state_perm = util.permute_state(test_state, logical_to_physical)
    test_statev = Statevector(test_state)
    test_statev_perm = Statevector(test_state_perm)


    test_fstatev = test_statev.copy().evolve(qc)
    test_fstatev_opt_perm = test_statev_perm.copy().evolve(qc_opt)
    test_fstate = test_fstatev.data
    test_fstate_opt_perm = test_fstatev_opt_perm.data

    test_fstate_opt = util.permute_state(test_fstate_opt_perm, logical_to_physical_final)
    #test_fstate_opt = util.permute_state(test_fstate_opt_perm, physical_to_logical)

    test_fprobs = np.abs(test_fstate)**2
    test_fprobs_opt = np.abs(test_fstate_opt)**2

    test_fprobs_s = np.sort(test_fprobs)
    test_fprobs_opt_s = np.sort(test_fprobs_opt)

    #print(logical_to_physical_final, physical_to_logical)
    #print(logical_to_physical, physical_to_logical_initial)
    #print(swaps)

    diff = np.abs((test_fstate - test_fstate_opt)).sum()
    #print(diff)
    #print(qc.count_ops(), qc_opt.count_ops())
    return diff
