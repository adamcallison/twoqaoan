import numpy as np

import twoqaoan.perm_util as perm_util
import twoqaoan.util as util

def schedule(waps, perms, nn_gates_collection, hardware_couplings):
    raise NotImplementedError # not finished!
    n = len(perms[0])
    qubit_distances = util.floyd_warshall(n, hardware_couplings, \
        standardize=True, symmetrize=True)
    # schedule will assume perfect routing
    nn_gates_collection = [list(nn_gates) for nn_gates in nn_gates_collection]
    perms = list(perms)
    swaps = list(swaps)

    colors = util.greedy_graph_color(n, nn_gates_collection[0])
    colors2 = {}
    for key, val in colors.items():
        try:
            colors2[val].append(key)
        except KeyError:
            colors2[val] = [key]
    gate_collections = list(tel.values())
    n_gates = [len(x) for x in gate_collections]
    idx = np.argsort(n_gates)
    gate_collections = [gate_collections[j] for j in idx]
    gate_collections_phys = [\
        perm_util.permute_pairs(gc, perms[0]) for gc in gate_collections\
        ]
    initial_schedule_cycles = [\
        ("hamiltonian_gate")

    initial_schedule_cycles = list(gate_collections)
    initial_schedule_cycles_phys = [\
        perm_util.permute_pairs(cycle, perms[0]) for \
        cycle in initial_schedule_cycles\
        ]

    schedule_cycles_rev = []
    schedule_cycles_phys_rev = []
    while sum([len(nn_gates) for nn_gates in nn_gates_collection]) > 0:
        current_perm = perms.pop()
        current_nn_gates = nn_gates_collection.pop()
        current_swap = swaps.pop()
        to_schedule = current_nn_gates
        for j, earlier_nn_gates in enumerate(nn_gates_collection):
            earlier_nn_gates_phys = perm_util.permute_pairs(\
                earlier_nn_gates_phys)
            not_to_schedule = []
            for gate_phys in earlier_nn_gates_phys:
                if qubit_distances[gate_phys[0], gate_phys[1]] == 1:
                    to_schedule.append(gate_phys)
                else:
                    not_to_schedule.append(gate_phys)
            earlier_nn_gates[j] = not_to_schedule

            colors = util.greedy_graph_color(n, to_schedule)
            colors2 = {}
            for key, val in colors.items():
                try:
                    colors2[val].append(key)
                except KeyError:
                    colors2[val] = [key]
            gate_collections = list(tel.values())
            n_gates = [len(x) for x in gate_collections]
            idx = np.argsort(n_gates)
            gate_collections = [gate_collections[j] for j in idx]

            schedule_cycles_rev_new = list(gate_collections)
            schedule_cycles_rev_new_phys = [\
                perm_util.permute_pairs(cycle, current_perm) for \
                cycle in schedule_cycles_new\
                ]
            schedle_cycles_rev += schedule_cycles_rev_new
            schedle_cycles_phys_rev += schedule_cycles_phys_rev_new

            # NEED TO SCHEDULE THE SWAP
