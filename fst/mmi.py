
import logging
import numpy as np
from lattice_functions import *
from posterior import *

def LatticeAcousticRescore(nnet_out, state_times, lat):
    # lat must be top sort and ilabel = pdf+1
    # check
    for s in range(lat.NumStates()):
        state = lat.GetState(s)
        t = state_times[s]
        for arc in state.GetArcs():
            if arc._ilabel != 0:
                pdf = arc._ilabel - 1
                arc._weight._value2 -= nnet_out[t][pdf]
        # end state
    # end all states

def LatticeAcousticRescore3D(nnet_out, state_times, lat, sentence, time_major = True):
    # lat must be top sort and ilabel = pdf+1
    for s in range(lat.NumStates()):
        state = lat.GetState(s)
        t = state_times[s]
        for arc in state.GetArcs():
            if arc._ilabel != 0:
                pdf = arc._ilabel - 1
                if time_major is True:
                    arc._weight._value2 -= nnet_out[t][sentence][pdf]
                else:
                    arc._weight._value2 -= nnet_out[sentence][t][pdf]

def MMILoss2D(nnet_out, lat, ali, sentence, acoustic_scale, lm_scale, time_major = False):
    max_time, state_times = LatticeStateTimes()
    # rescore the latice
    LatticeAcousticRescore3D(nnet_out, state_times, lat, sentence, time_major)
    if acoustic_scale != 1.0 or lm_scale != 1.0:
        ScaleLattice(lat, lm_scale, acoustic_scale)

    # get the posteriors
    tot_backward_prob, acoustic_like_sum, post = LatticeForwardBackward(lat)

    # convert the Posterior to a matrix
    d1, d2, d3 = np.shape(nnet_out)
    if time_major is True:
        nnet_diff_h = np.zeros(d1 * d3, dtype=np.float32).reshape(d1,d3)
    else:
        nnet_diff_h = np.zeros(d2 * d3, dtype=np.float32).reshape(d2,d3)
    
    PosteriorToPdfMatrix(post, nnet_diff_h)

    # Calculate the MMI-objective function
    # Calculate the likelihood of correct path from acoustic score,
    # the denominator likelihood is the total likelihood of the lattice.
    path_ac_like = 0.0
    if time_major is True:
        for t in range(len(ali)):
            pdf = ali[t]
            path_ac_like += nnet_out[t][sentence][pdf]
    else:
        for t in range(len(ali)):
            pdf = ali[t]
            path_ac_like += nnet_out[sentence][t][pdf]

    path_ac_like *= acoustic_scale
    mmi_obj = path_ac_like - tot_backward_prob

    # Note: numerator likelihood does not include graph score,
    # while denominator likelihood contains graph scores.
    # The result is offset at the MMI-objective.
    # However the offset is constant for given alignment,
    # so it does not change accross epochs.

    # Sum the den-posteriors under the correct path,
    post_on_ali = 0.0
    for t in range(len(ali)):
        pdf = ali[t]
        post_on_ali += nnet_diff_h[t][pdf]

    # Report,
    logging.info("Lattice #" + str(sentence) + " processed ("  )

    logging.info("Utterance " + str(utt) + ": Average MMI obj. value = " +
            str(mmi_obj/num_frames) + " over " + str(num_frames) + " frames." +
            " (Avg. den-posterior on ali " + str(post_on_ali / num_frames) + ")")
    
    # Search for the frames with num/den mismatch,
    frm_drop = 0
    frm_drop_vec = []
    for t in range(len(ali)):
        pdf = ali[t]
        posterior = nnet_diff_h[t][pdf]
        if posterior < 1e-20:
            frm_drop += 1
            frm_drop_vec.append(t)

    # 8. subtract the pdf-Viterbi-path
    for t in range(len(ali)):
        pdf = ali[t]
        nnet_diff_h[t][pdf] -= 1.0


    # 9. Drop mismatched frames from the training by zeroing the derivative,
    if drop_frames is True:
        for i in frm_drop_vec:
            nnet_diff_h[i] = 0.0

    # Report the frame dropping
    if frm_drop > 0:






def MMILoss3D(nnet_out, lat, time_major = False):
    '''
       nnet_out : it's numpy matrix if time_major is False (sentences, times, dim),else (times, sentences, dim)
       lat      : lattice list
       time_major: nnet_out major axis
    '''
    if time_major is False:
        assert np.shape(nnet_out)[0] == len(lat)
    else:
        assert np.shape(nnet_out)[1] == len(lat)
    
    mmiloss = []
    for loss in range(len(lat)):

        
