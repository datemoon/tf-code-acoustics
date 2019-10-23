
#include <fst/fstlib.h>
#include "base/kaldi-math.h"
#include "chain/chain-training.h"

namespace hubo
{

using namespace kaldi;
using namespace kaldi::chain;

class DenominatorGraphSaver
{
public:
	DenominatorGraphSaver(){ }
	void Init(const int32 *indexs, const int32 *in_labels,
			const int32 *out_labels, BaseFloat* weights, 
			const int32* statesinfo, int32 num_states, int32 num_pdfs, 
			bool delete_laststatesuperfinal = false, const int32 den_start_state = 0);

	DenominatorGraph *GetDenGraph()
	{
		return _den_graph;
	}

	~DenominatorGraphSaver()
	{
		delete _den_graph;
	}

	void Info()
	{
		;
	}
private:
	DenominatorGraph *_den_graph;
};

/*
* Compute Chain loss.
* indexs (input)   : fst cur_state and next_state. indexs must be 3 dimensional tensor,
*                    which has dimension (n, arcs_nums, 2), where n is the minibatch index,
*                    states_nums is lattice state number, 2 is lattice dim save [cur_state, next_state]
* in_labels        : in_labels is 2 dimensional tensor, which has dimension (n, arcs_nums),
*                    practical application pdf = in_labels[n] - 1
* out_labels       : the same as in_labels, it's not important.
* weights          : lm_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
* statesinfo       : statesinfo is 3 dimensional tensor, which has dimension (n, states_num, 2)
* num_states       : num_states is states number, which has dimension (n)
* max_num_arcs     : all fst the maximum arc number.
* max_num_states   : all fst the maximum state number.
* deriv_weights    : which has dimension (rows, n) time_major
* supervision_weights             :
* supervision_num_sequences       :
* supervision_frames_per_sequence :
* supervision_label_dim           :
* nnet_out         : here it's nnet forward and no soft max and logit. nnet_outmust be 3 dimensional tensor,
*                    which has dimension (t, n, p), where t is the time index, n is the minibatch index,
*                    and p is class numbers. it map (rows, batch_size, cols)
* rows             : is the max time index
* batch_size       : is the minibatch index
* cols             : is class numbers
*
* //labels           : here it's acoustic align, max(labels) < p. which has dimension (n, t)
* acoustic_scale   : acoustic scale
* gradient (outptu): it shape same as nnet_out
* objf             : it loss . which has dimension (3)
*
* */

bool ChainLoss(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat* deriv_weights,
		const BaseFloat supervision_weights, const int32 supervision_num_sequences, 
		const int32 supervision_frames_per_sequence, const int32 supervision_label_dim, 
		const BaseFloat* nnet_out,
		int32 rows, int32 batch_size, int32 cols,
		// denominator fst
		const int32 *den_indexs, const int32 *den_in_labels, 
		const int32 *den_out_labels, BaseFloat* den_weights, 
		const int32* den_statesinfo, const int32 start_state,
		const int32 den_num_states,
		// output
		BaseFloat* gradient,
		BaseFloat* objf,
		float l2_regularize, float leaky_hmm_coefficient, float xent_regularize,
		fst::VectorFst<fst::StdArc> *den_fst_test = NULL, DenominatorGraph * den_graph_test = NULL, chain::Supervision *merge_supervision_test = NULL);


/*
 *
 *
 *
 * */
bool ChainLossDen(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat* deriv_weights,
		const BaseFloat supervision_weights, const int32 supervision_num_sequences, 
		const int32 supervision_frames_per_sequence, const int32 supervision_label_dim, 
		const BaseFloat* nnet_out,
		int32 rows, int32 batch_size, int32 cols,
		// denominator fst
		DenominatorGraphSaver &den_graph,
		BaseFloat* gradient,
		BaseFloat* objf,
		float l2_regularize, float leaky_hmm_coefficient, float xent_regularize);

/*
* Compute Chain loss and xent loss.
* indexs (input)   : fst cur_state and next_state. indexs must be 3 dimensional tensor,
*                    which has dimension (n, arcs_nums, 2), where n is the minibatch index,
*                    states_nums is lattice state number, 2 is lattice dim save [cur_state, next_state]
* in_labels        : in_labels is 2 dimensional tensor, which has dimension (n, arcs_nums),
*                    practical application pdf = in_labels[n] - 1
* out_labels       :
* weights          : lm_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
* statesinfo       : statesinfo is 3 dimensional tensor, which has dimension (n, states_num, 2)
* num_states       : num_states is states number, which has dimension (n)
* max_num_arcs     : all fst the maximum arc number.
* max_num_states   : all fst the maximum state number.
* deriv_weights    : which has dimension (rows, n) time_major
* supervision_weights             :
* supervision_num_sequences       :
* supervision_frames_per_sequence :
* supervision_label_dim           :
*
* nnet_out         : here it's nnet forward and no soft max and logit. nnet_out must be 3 dimensional tensor,
*                    which has dimension (t, n, p), where t is the time index, n is the minibatch index,
*                    and p is class numbers. it map (rows, batch_size, cols)
* xent_nnet_out    : here it's nnet forward and soft max and logit. xent_nnet_out must be 3 dimensional tensor,
*                    which has dimension (t, n, p), where t is the time index, n is the minibatch index,
*                    and p is class numbers. it map (rows, batch_size, cols)
* rows             : is the max time index
* batch_size       : is the minibatch index
* cols             : is class numbers
*
* //labels         : here it's acoustic align, max(labels) < p. which has dimension (n, t)
* acoustic_scale   : acoustic scale
* gradient (output): it shape same as nnet_out
* gradient_xent    : it shape same as nnet_out
* objf             : it loss . which has dimension (4)
*
* */
bool ChainXentLossDen(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat* deriv_weights,
		const BaseFloat supervision_weights, const int32 supervision_num_sequences, 
		const int32 supervision_frames_per_sequence, const int32 supervision_label_dim, 
		const BaseFloat* nnet_out,
		const BaseFloat* xent_nnet_out,
		int32 rows, int32 batch_size, int32 cols,
		// denominator fst
		DenominatorGraphSaver &den_graph_saver,
		// output
		BaseFloat* gradient,
		BaseFloat* gradient_xent,
		BaseFloat* objf,
		float l2_regularize, float leaky_hmm_coefficient, float xent_regularize);
}
