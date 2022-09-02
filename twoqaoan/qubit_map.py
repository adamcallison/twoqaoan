import numpy as np

from twoqaoan.util import standardize_pairs, floyd_warshall, invert_permutation, permute_array

class QubitMap(object):
    def __init__(self, num_qubits, physical_coupled, logical_to_physical=None, full_init=True):
        self._num_qubits = num_qubits
        self._physical_coupled = standardize_pairs([tuple(x) for x in physical_coupled], symmetrize=False)
        if logical_to_physical is None:
            self._logical_to_physical = np.arange(num_qubits)
        else:
            self._logical_to_physical = np.array(logical_to_physical)
        
        if full_init:
            self._physical_distances = floyd_warshall(self.num_qubits, self.physical_coupled, standardize=False, symmetrize=True)
            self._compute_logical_coupled()
            if logical_to_physical is None:
                self._logical_distances = self._physical_distances.copy()
            else:
                #self._logical_distances = permute_array(self.physical_distances, self.physical_to_logical)
                self._logical_distances = permute_array(self.physical_distances, self.logical_to_physical)
        
    @property
    def num_qubits(self):
        return self._num_qubits
        
    @property
    def logical_to_physical(self):
        return self._logical_to_physical.copy()
    
    @property
    def physical_to_logical(self):
        return invert_permutation(self._logical_to_physical)
    
    @property
    def logical_coupled(self):
        return list(self._logical_coupled)
    
    @property
    def physical_coupled(self):
        return list(self._physical_coupled)
    
    def _compute_logical_coupled(self):
        physical_coupled = self.physical_coupled
        physical_to_logical = self.physical_to_logical
        tmp = [(physical_to_logical[x[0]], physical_to_logical[x[1]]) for x in physical_coupled]
        #logical_to_physical = self.logical_to_physical
        #tmp = [(logical_to_physical[x[0]], logical_to_physical[x[1]]) for x in physical_coupled]
        tmp = standardize_pairs(tmp, symmetrize=False)
        self._logical_coupled = tmp
        return tmp
    
    @property
    def physical_distances(self):
        return self._physical_distances.copy()
    
    @property
    def logical_distances(self):
        return self._logical_distances.copy()
    
    def swap(self, q1, q2, physical_indices=True):
        # q1 and q2 are PHYSICAL indices by default
        
        #if not physical_indices:
            #logical_to_physical = self.logical_to_physical
            #q1_use, q2_use = logical_to_physical[q1], logical_to_physical[q2]
        #else:
            #q1_use, q2_use = q1, q2
            
        if not physical_indices:
            q1_use, q2_use = q1, q2

        else:
            physical_to_logical = self.physical_to_logical
            q1_use, q2_use = physical_to_logical[q1], physical_to_logical[q2]
        
        self._logical_to_physical[q1_use], self._logical_to_physical[q2_use] = self._logical_to_physical[q2_use], self._logical_to_physical[q1_use]
        self._compute_logical_coupled()
        #self._logical_distances = permute_array(self.physical_distances, self.physical_to_logical)
        self._logical_distances = permute_array(self.physical_distances, self.logical_to_physical)
        return self

    def copy(self):
        other = QubitMap(self.num_qubits, self.physical_coupled, self.logical_to_physical, full_init=False)
        other._logical_coupled, other._physical_distances, other._logical_distances = self._logical_coupled.copy(), self._physical_distances.copy(), self._logical_distances.copy()
        return other