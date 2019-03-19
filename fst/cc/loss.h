#ifndef __MMI_LOSS_H__
#define __MMI_LOSS_H__

#include "matrix.h"
#include "base-math.h"
#include "sparse-lattice-function.h"
namespace hubo
{

/*
 * Compute MMI loss.
 * indexs (input)   : fst cur_state and next_state. indexs must be 3 dimensional tensor,
 *                    which has dimension (n, arcs_nums, 2), where n is the minibatch index,
 *                    states_nums is lattice state number, 2 is lattice dim save [cur_state, next_state]
 * pdf_values       : pdf_values is 2 dimensional tensor, which has dimension (n, arcs_nums), 
 *                    practical application pdf = pdf_values[n] - 1          
 * lm_ws            : lm_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
 * am_ws            : am_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
 * statesinfo       : statesinfo is 3 dimensional tensor, which has dimension (n, states_num, 2)
 * num_states       : num_states is states number, which has dimension (n)
 * 
 * nnet_out         : here it's nnet forward and no soft max and logit. nnet_outmust be 3 dimensional tensor,
 *                    which has dimension (t, n, p), where t is the time index, n is the minibatch index,
 *                    and p is class numbers. it map (rows, batch_size, cols)
 * rows             : is the max time index
 * batch_size       : is the minibatch index
 * cols             : is class numbers
 *
 * labels           : here it's acoustic align, max(labels) < p. which has dimension (n, t)
 * sequence_length  : The number of time steps for each sequence in the batch. which has dimension (n)
 * acoustic_scale   : acoustic scale
 * gradient (outptu): it shape same as nnet_out
 * loss             : it loss . which has dimension (n)
 *
 * */
bool MMILoss(const int32 *indexs, const int32 *pdf_values,
		BaseFloat* lm_ws, BaseFloat* am_ws, 
		const int32 *statesinfo, const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat* nnet_out, 
		int32 rows, int32 batch_size, int32 cols,
		const int32 *labels,
		const int32 *sequence_length, 
		BaseFloat acoustic_scale, BaseFloat* gradient, 
		BaseFloat *loss, bool drop_frames = true);

/* lat (input)         :
 * nnet_out_h          :
 * labels              :
 *
 * nnet_diff_h(output) :
 * loss                : 
 * return              : loss
 * */
void MMIOneLoss(Lattice *lat, Matrix<const BaseFloat> *nnet_out_h, const int32 *labels,
		Matrix<BaseFloat> *nnet_diff_h, BaseFloat acoustic_scale,
	   	BaseFloat *loss, bool drop_frames = true);


} // namespace
#endif
