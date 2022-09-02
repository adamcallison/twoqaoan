from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

def circuit_from_hamiltonian(J, h, c, qaoa_param):
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
    
def circuit_from_hamiltonian_optimized(J_sequence, J, h, c, qaoa_param):
    n = h.shape[0]
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)

    logical_to_physical = None
    if J_sequence[0][0] == 'logical to physical':
        logical_to_physical = np.array(J_sequence[0][1], dtype=int)
        J_sequence = J_sequence[1:]

    for q1 in range(n):
        logical_qubit = q1
        if logical_to_physical is None:
            physical_qubit = q1
        else:
            physical_qubit = logical_to_physical[q1]
        #hval = h[physical_qubit]
        hval = h[q1]
        if not (hval == 0.0):
            #qc.rz(2*qaoa_param*hval, q1) # SHOULD THIS BE PHYSICAL QUBIT???
            qc.rz(2*qaoa_param*hval, physical_qubit) # SHOULD THIS BE PHYSICAL QUBIT???

    physical_to_logical = None
    for i, instruction in enumerate(J_sequence):
        # putting swapints as just two gates next to each other to allow parameterized gates
        if instruction[0] in ('swap', 'swapint'):
            physical_pair = instruction[2]
            logical_pair = instruction[1]
            qc.swap(physical_pair[0], physical_pair[1])
        if instruction[0] in ('interaction', 'swapint'):
            physical_pair = instruction[2]
            logical_pair = instruction[1]
            Jcoeff = J[logical_pair[0], logical_pair[1]] + J[logical_pair[1], \
                logical_pair[0]]
            if Jcoeff == 0:
                continue
            qc.rzz(2*qaoa_param*Jcoeff, physical_pair[0], physical_pair[1])
        if instruction[0] == 'logical to physical':
            raise ValueError("'logical to physical' can only be done as first step")
        if instruction[0] == 'physical to logical':
            if not (i == (len(J_sequence)) - 1):
                raise ValueError("'physical to logical' can only be done as last step")
            else:
                physical_to_logical = np.array(instruction[1])

    if not (c == 0.0):
        qc.global_phase = qc.global_phase - (qaoa_param*c)
    return qc, logical_to_physical, physical_to_logical