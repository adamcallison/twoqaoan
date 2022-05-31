import numpy as np
import twoqaoan.twoqaoan as twoqaoan
import twoqaoan.perm_util as perm_util
from qiskit.test.mock import FakeLondon, FakeTokyo
from qiskit.quantum_info import Statevector

def test_circuit_from_hamiltonian():
    diff = _test_circuit_from_hamiltonian()
    assert diff < 1e-10

def _test_circuit_from_hamiltonian(fakehardware='london', verbose=False):
    fakehardware = fakehardware.lower()
    if fakehardware == 'london':
        ft = FakeLondon()
        n = 5

    if fakehardware == 'tokyo':
        ft = FakeTokyo()
        n = 20

    J = np.random.default_rng().normal(loc=0.0, scale=1.0, size=(n, n))
    h = J.diagonal().copy()
    np.fill_diagonal(J, 0.0)
    c = np.random.default_rng().normal(loc=0.0, scale=1.0)
    for j in range(0, n-1):
        for k in range(j+1, n):
            rn = np.random.default_rng().uniform()
            if rn > 0.5:
                J[j, k] = J[k, j] = 0.0

    qc = twoqaoan.circuit_from_hamiltonian(J, h, c, qaoa_param=1.0, \
        optimize=False)

    hardware_couplings = []
    for gate in ft.properties().gates:
        qubits = gate.qubits
        if not len(qubits) == 2:
            continue
        qubits = tuple(np.sort(qubits))
        if not qubits in hardware_couplings:
            hardware_couplings.append(qubits)
    qc_opt, initial_permutation, final_permutation = \
        twoqaoan.circuit_from_hamiltonian(J, h, c, qaoa_param=1.0, \
        hardware_couplings=hardware_couplings, sa_iterations=100, \
        sa_runs=100, routing_runs=100, optimize=True, verbose=verbose)

    for j in range(n):
        qc.rx(1.0, j)
        qc_opt.rx(1.0, j)

    N = 2**n
    test_amps = np.random.default_rng().uniform(size=N)
    test_phases = np.random.default_rng().uniform(size=N)*2*np.pi
    test_state = test_amps*test_phases
    test_state = test_state/np.sqrt(np.dot(test_state, test_state))
    test_state_perm = perm_util.permute_state(test_state, initial_permutation)
    test_statev = Statevector(test_state)
    test_statev_perm = Statevector(test_state_perm)


    test_fstatev = test_statev.copy().evolve(qc)
    test_fstatev_opt_perm = test_statev_perm.copy().evolve(qc_opt)
    test_fstate = test_fstatev.data
    test_fstate_opt_perm = test_fstatev_opt_perm.data

    inv_final_permutation = perm_util.invert_permutation(final_permutation)
    test_fstate_opt = perm_util.permute_state(test_fstate_opt_perm, \
        inv_final_permutation)

    test_fprobs = np.abs(test_fstate)**2
    test_fprobs_opt = np.abs(test_fstate_opt)**2

    test_fprobs_s = np.sort(test_fprobs)
    test_fprobs_opt_s = np.sort(test_fprobs_opt)

    diff = np.abs((test_fstate - test_fstate_opt)).sum()
    return diff
