import numpy as np

def permute_array(a, perm):
    new_a = np.zeros_like(a)
    n = len(perm)
    for j in range(n):
        for k in range(n):
            new_a[j, k] = a[perm[j], perm[k]]
    return new_a

def permute_pairs(pairs, perm):
    inv_perm = invert_permutation(perm)
    new_pairs = []
    for pair in pairs:
        new_pairs.append((inv_perm[pair[0]], inv_perm[pair[1]]))
    return new_pairs

def permute_state(state, perm):
    # very slow, there must be a better way?
    N = state.shape[0]
    n = int(np.round(np.log2(N)))
    state_perm = np.zeros_like(state)
    j_perms = []
    for j in range(N):
        jbin = bin(j)[2:]
        jbin = ('0'*(n-len(jbin)))+jbin
        jbin = jbin[::-1]
        jbin = np.array(list([int(x) for x in jbin]))
        jbin_perm = np.ndarray(len(jbin), dtype=int)
        for k in range(n):
            jbin_perm[k] = jbin[perm[k]]
        jbin_perm = (''.join(list([str(x) for x in jbin_perm])))
        jbin_perm = jbin_perm[::-1]
        j_perm = int(jbin_perm, 2)
        j_perms.append(j_perm)
    j_perms = np.array(j_perms, dtype=int)
    state_perm[j_perms] = state[:]
    return state_perm

def invert_permutation(permutation):
    inverse_perm = np.zeros_like(permutation)
    for i, j in enumerate(permutation):
        inverse_perm[j] = i
    return inverse_perm
