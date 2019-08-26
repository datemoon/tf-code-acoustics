
#include <vector>
#include <thread>
#include <cassert>
#include <iostream>
#include <sys/time.h>

#include "base/kaldi-math.h"
#include "chain/chain-training.h"
#include "fst-convert-openfst.h"


using namespace kaldi;
using namespace kaldi::chain;

namespace hubo
{

	/*
* Compute Chain loss.
* indexs (input)   : fst cur_state and next_state. indexs must be 3 dimensional tensor,
*                    which has dimension (n, arcs_nums, 2), where n is the minibatch index,
*                    states_nums is lattice state number, 2 is lattice dim save [cur_state, next_state]
* in_labels        : in_labels is 2 dimensional tensor, which has dimension (n, arcs_nums),
*                    practical application pdf = in_labels[n] - 1
* out_labels       :
* weights          : lm_ws is 2 dimensional tensor, which has dimension (n, arcs_nums)
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
* //labels           : here it's acoustic align, max(labels) < p. which has dimension (n, t)
* sequence_length  : The number of time steps for each sequence in the batch. which has dimension (n)
* acoustic_scale   : acoustic scale
* gradient (outptu): it shape same as nnet_out
* loss             : it loss . which has dimension (n)
*
* */

bool ChainLoss(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat *supervision_weights, const int32 *supervision_num_sequences, 
		const int32 *supervision_frames_per_sequence, const int32 *supervision_label_dim, 
		const int32 *sequence_length,
		const BaseFloat* nnet_out,
		int32 rows, int32 batch_size, int32 cols,
		// denominator fst
		const int32 *den_indexs, const int32 *den_in_labels, const int32 *den_out_labels, 
		BaseFloat* den_weights, const int32* den_statesinfo, const int32 den_num_states,
		BaseFloat* gradient,
		float l2_regularize, float leaky_hmm_coefficient, float xent_regularize)
{
#ifdef DEBUG_SPEED
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
#endif
	// convert fst
	std::vector<fst::VectorFst<fst::StdArc> > fst_v;
	// first process lat_v
	for(int32 i=0; i < batch_size; i++)
	{
		// first get state number
		int32 cur_num_states = num_states[i];
		const int32 *cur_statesinfo = statesinfo + i * max_num_states * 2;
		// second get arc number
		// int32 cur_num_arcs = cur_statesinfo[(cur_num_states-1)*2] + cur_statesinfo[(cur_num_states-1)*2+1];
		const int32 *cur_indexs = indexs + i * max_num_arcs * 2;
		const int32 *cur_in_labels = in_labels + i * max_num_arcs;
		const int32 *cur_out_labels = out_labels + i * max_num_arcs;
		BaseFloat* cur_weights = weights + i * max_num_arcs;
		fst::VectorFst<fst::StdArc> fst;
		bool ret = fst::ConvertSparseFstToOpenFst(cur_indexs, cur_in_labels, 
				cur_out_labels, cur_weights, cur_statesinfo, cur_num_states, &fst);
		if (ret != true)
		{
			return false;
		}
		fst_v.push_back(fst);
	} // fst ok
	// supervision merge
	std::vector<const chain::Supervision*> input_supervision;
	for(int32 i=0; i < batch_size; i++)
	{
		chain::Supervision supervision;
		supervision.weight = supervision_weights[i];
		supervision.num_sequences = supervision_num_sequences[i];
		supervision.frames_per_sequence = supervision_frames_per_sequence[i];
		supervision.label_dim = supervision_label_dim[i];
		supervision.fst = fst_v[i];
	}
	chain::Supervision output_supervision;
	MergeSupervision(input_supervision,
			&output_supervision);

	chain::Supervision merge_supervision;
	merge_supervision.Swap(&output_supervision);
	// supervision ok
#ifdef DEBUG_SPEED
	gettimeofday(&end, NULL);
	std::cout << "DEBUG_SPEED : " << __FILE__ << " : convert fst time:"
		<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
#endif
	// denominator graph create
	int32 num_pdf = merge_supervision.label_dim;
	static DenominatorGraph *den_graph = NULL;
	if (den_graph == NULL)
	{
		fst::VectorFst<fst::StdArc> den_fst;
		bool ret = fst::ConvertSparseFstToOpenFst(den_indexs, den_in_labels, den_out_labels, den_weights, den_statesinfo,
				den_num_states, &den_fst);
		if(ret != true)
			return ret;
		den_graph = new DenominatorGraph(den_fst, num_pdf);
	}

	ChainTrainingOptions opts;
	
	opts.l2_regularize = l2_regularize;
	opts.leaky_hmm_coefficient = leaky_hmm_coefficient;
	opts.xent_regularize = xent_regularize;
	
	Supervision supervision;
	// copy nnet_out to nnet_output
	CuSubMatrix<BaseFloat> nnet_output(nnet_out, rows * batch_size, cols, cols);

	CuSubMatrix<BaseFloat> nnet_output_deriv(gradient,
			nnet_output.NumRows(),
			nnet_output.NumCols(),
			nnet_output.NumCols());

	bool use_xent = (opts.xent_regularize != 0.0);
	//std::string xent_name = "output" + "-xent";  // typically "output-xent".
	CuMatrix<BaseFloat> xent_deriv;
	if (use_xent)
		xent_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
				kUndefined);
	
	BaseFloat tot_objf, tot_l2_term, tot_weight;

	ComputeChainObjfAndDeriv(opts, *den_graph, merge_supervision, nnet_output, 
			&tot_objf, &tot_l2_term, &tot_weight,
			&nnet_output_deriv, 
			(use_xent ? &xent_deriv : NULL));

	// loss nnet_output_deriv

	return true;
}

} // namespace hubo
