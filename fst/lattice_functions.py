
import logging
import array
import math
from fst_math import *

def LatticeStateTimes(lat, times):
    # top sort lat
    # check top sort
    num_states = lat.NumStates()
    times = [ -1 for x in range(num_states) ]
    times[0] = 0
    max_time = 0
    for state in range(num_states):
        cur_time = times[state]
        if max_time < cur_time:
            max_time = cur_time
        for arc in lat.GetArcs(state):
            if arc._ilabel != 0:
                if times[arc._nextstate] == -1:
                    times[arc._nextstate] = cur_time + 1
                else:
                    assert times[arc._nextstate] == cur_time + 1
            else: # epsilon input label on arc. Same time instance
                if times[arc._nextstate] == -1:
                    times[arc._nextstate] = cur_time
                else:
                    assert times[arc._nextstate] == cur_time
        # end this state

    return max_time


def LatticeForwardBackward(lat):
    acoustic_like_sum = 0.0
    # Make sure the lattice is topologically sorted.
    assert lat.Start() == 0

    num_states = lat.NumStates()
    state_times = []
    max_time = LatticeStateTimes(lat, state_times)
    post = [ {} for x in range(max_time) ]

    alpha = array.array('d',[ kLogZero for x in range(num_states) ])
    beta = alpha # we re-use the same memory for
    # this, but it's semantically distinct so we name it differently.
    tot_forward_prob = kLogZero

    alpha[0] = 0.0
    # Propagate alphas forward.
    for s in range(num_states):
        this_alpha = alpha[s]
        for arc in lat.GetArcs(s):
            arc_like = - arc._weight.Value()
            alpha[arc._nextstate] = LogAdd(alpha[arc._nextstate], this_alpha + arc_like)
        final_weight = lat.Final(s)
        if final_weight.IsZero() is False:
            final_like = this_alpha - final_weight.Value()
            tot_forward_prob = LogAdd(tot_forward_prob, final_like)
            assert state_times[s] == max_time and "Lattice is inconsistent (final-prob not at max_time)"
    # end forward
    # backforward
    for s in range(num_states-1,-1,-1):
        final_weight = lat.Final(s)
        this_beta = -final_weight.Value()
        for arc in lat.GetArcs(s):
            arc_like = - arc._weight.Value()
            arc_beta = beta[arc._nextstate] + arc_like
            this_beta = LogAdd(this_beta, arc_beta)

            # The following "if" is an optimization to avoid un-needed exp()
            if arc._ilabel != 0 or acoustic_like_sum is not None:
                posterior = math.exp(alpha[s] + arc_beta - tot_forward_prob)
                if arc._ilabel != 0: # Arc has a transition-id on it [not epsilon]
                    pdf = arc._ilabel - 1
                    try:
                        post[state_times[s]][pdf] += posterior
                    except KeyError:
                        post[state_times[s]][pdf] = posterior
                    #post[state_times[s]].append((pdf, posterior))
                    if acoustic_like_sum is not None:
                        acoustic_like_sum -= posterior * arc._weight._value2
        # end this state
        if acoustic_like_sum is not None and final_weight.IsZero() is False:
            final_logprob = - final_weight.Value()
            posterior = math.exp(alpha[s] + final_logprob - tot_forward_prob)
            acoustic_like_sum -= posterior * final_weight._value2

        beta[s] = this_beta
    # end back forward
    tot_backward_prob = beta[0]
    if tot_forward_prob - tot_backward_prob > 1e-8 or tot_forward_prob - tot_backward_prob < -1e-8:
        logging.info('Total forward probability over lattice = %d, while total backward probability = %d' % (tot_forward_prob, tot_backward_prob))
    # Now combine any posteriors with the same transition-id.
    #for t in range(max_time):
        
    return tot_backward_prob, acoustic_like_sum, post

