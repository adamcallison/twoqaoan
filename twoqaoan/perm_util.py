import numpy as np

def permute_array(a, perm):
    new_a = np.zeros_like(a)
    n = len(perm)
    for j in range(n):
        for k in range(n):
            new_a[perm[j], perm[k]] = a[j, k]
    return new_a

def permute_pairs(pairs, perm):
    new_pairs = []
    for pair in pairs:
        new_pairs.append((perm[pair[0]], perm[pair[1]]))
    return new_pairs

def permute_state(state, perm):
    # very slow, there must be a better way?
    inv_perm = invert_permutation(perm)
    N = state.shape[0]
    n = int(np.round(np.log2(N)))
    state_unmapped = np.zeros_like(state)
    j_unmappeds = []
    for j in range(N):
        jbin = bin(j)[2:]
        jbin = ('0'*(n-len(jbin)))+jbin
        jbin = jbin[::-1]
        jbin = np.array(list([int(x) for x in jbin]))
        jbin_unmapped = np.ndarray(len(jbin), dtype=int)
        for k in range(n):
            jbin_unmapped[perm[k]] = jbin[k]
        jbin_unmapped = (''.join(list([str(x) for x in jbin_unmapped])))
        jbin_unmapped = jbin_unmapped[::-1]
        j_unmapped = int(jbin_unmapped, 2)
        j_unmappeds.append(j_unmapped)
    j_unmappeds = np.array(j_unmappeds, dtype=int)
    state_unmapped[j_unmappeds] = state[:]
    return state_unmapped

def invert_permutation(permutation):
    inverse_perm = np.zeros_like(permutation)
    for i, j in enumerate(permutation):
        inverse_perm[j] = i
    return inverse_perm
