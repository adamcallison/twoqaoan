import numpy as np

def simulated_annealing_run(extra_inputs, iterations,
                           initial_state_generator,
                           cost_function,
                           candidate_generator,
                           acceptance_rule,
                           acceptance_parameter_generator,
                           verbose=False):

    best_cost = float('inf')
    state = initial_state_generator(extra_inputs)
    cost = cost_function(extra_inputs, state)
    costs = np.zeros(iterations+1, dtype=float)
    costs[0] = cost
    for iteration in range(iterations):
        if verbose:
            pc = 100*(iteration+1)/iterations
            print(f"{pc:.2f}% complete.", end="\r")
        acceptance_parameter = acceptance_parameter_generator(extra_inputs, iterations, iteration)
        candidate_state = candidate_generator(extra_inputs, state)
        candidate_cost = cost_function(extra_inputs, candidate_state)
        accept = acceptance_rule(cost, candidate_cost, acceptance_parameter)
        if accept:
            state, cost = candidate_state, candidate_cost
            if cost < best_cost:
                best_state, best_cost = state, cost
        costs[iteration+1] = cost
    return best_state, best_cost, costs

def simulated_annealing(extra_inputs, iterations, runs,
                        initial_state_generator,
                        cost_function,
                        candidate_generator,
                        acceptance_rule,
                        acceptance_parameter_generator,
                        verbose=False):
    if verbose:
        if runs == 1:
            outer_verbose, inner_verbose = False, True
        else:
            outer_verbose, inner_verbose = True, False
    else:
        outer_verbose, inner_verbose = False, False

    best_cost = float('inf')
    for run in range(runs):
        if outer_verbose:
            pc = 100*(run+1)/runs
            print(f"{pc:.2f}% complete.", end="\r")

        state, cost, costs = simulated_annealing_run(extra_inputs, iterations,
                             initial_state_generator,
                             cost_function,
                             candidate_generator,
                             acceptance_rule,
                             acceptance_parameter_generator,
                             verbose=inner_verbose)
        if cost < best_cost:
            best_state, best_cost, best_costs = state, cost, costs
    return best_state, best_cost, best_costs
