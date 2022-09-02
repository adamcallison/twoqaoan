from twoqaoan.sa_util import random_map_for_annealing, qap_cost_for_annealing, map_neighbour_for_annealing, boltzmann_acceptance_rule, temperature_schedule
from twoqaoan.simulated_annealing import simulated_annealing

def map_by_annealing(num_qubits, hamiltonian_interactions, physical_coupled, iterations, runs, verbose=False):
    annealing_inputs = (num_qubits, hamiltonian_interactions, physical_coupled)
    best_map, best_cost, best_costs = simulated_annealing(annealing_inputs, iterations, runs,
                                                            random_map_for_annealing,
                                                            qap_cost_for_annealing,
                                                            map_neighbour_for_annealing,
                                                            boltzmann_acceptance_rule,
                                                            temperature_schedule,
                                                            verbose=verbose)
    return best_map, best_cost, best_costs