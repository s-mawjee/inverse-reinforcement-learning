import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef np.float_t FTYPE_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

cdef np.ndarray[FTYPE_t, ndim=1] one_step_lookahead_cython(int state_, np.ndarray[FTYPE_t, ndim=1] V, np.ndarray[FTYPE_t, ndim=3] transition_probs, np.ndarray[FTYPE_t, ndim=1] rewards, float discount_factor, int number_of_actions):
    cdef int state = state_
    cdef np.ndarray[FTYPE_t, ndim=1] A = np.zeros(number_of_actions)
    
    cdef int a = 0
    cdef int number_of_states = transition_probs.shape[2]
    cdef int next_state = 0
    cdef float prob = 0.0
    
    
    for a in range(number_of_actions):
        for next_state in range(number_of_states):            
            prob = transition_probs[state,a,next_state]
            A[a] =  A[a]  + prob * (rewards[state] + discount_factor * V[next_state])
    return A


def value_iteration_cython(np.ndarray[FTYPE_t, ndim=3] transition_probs, np.ndarray[FTYPE_t, ndim=1] rewards,  float theta=0.001,  float discount_factor=0.9):
    
    
    cdef int number_of_states = transition_probs.shape[0]
    cdef int number_of_actions = transition_probs.shape[1]
    cdef np.ndarray[FTYPE_t, ndim=1] V = np.zeros(number_of_states)
    
    
    cdef int s = 0
    cdef float delta = 0.0
    cdef float best_action_value = 0.0
    cdef np.ndarray[FTYPE_t, ndim=1] A = np.zeros(number_of_actions)
    
    
    while True:
        delta = 0.0
        for s in range(number_of_states):
            A = one_step_lookahead_cython(s, V, transition_probs, rewards, discount_factor, number_of_actions)
            best_action_value = max(A)
            delta = max([delta, np.abs(best_action_value - V[s])])
            V[s] = best_action_value
        if delta < theta:
            break
    
    cdef np.ndarray[FTYPE_t, ndim=2] policy = np.zeros([number_of_states, number_of_actions])
    cdef int best_action = 0
    s = 0
    for s in range(number_of_states):
        A = one_step_lookahead_cython(s, V, transition_probs, rewards, discount_factor, number_of_actions)
        # A_norm = normalise(A)
        A_norm = np.exp(A)/sum(np.exp(A))
        policy[s] = A_norm

    return policy, V