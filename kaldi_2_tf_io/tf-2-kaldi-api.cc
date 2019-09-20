
#include <vector>
#include <thread>
#include <cassert>
#include <iostream>
#include <sys/time.h>

#include "base/kaldi-math.h"
#include "chain/chain-training.h"
#include "fst-convert-openfst.h"
#include "tf-2-kaldi-api.h"

using namespace kaldi;
using namespace kaldi::chain;

namespace hubo
{

void DenominatorGraphSaver::Init(const int32 *indexs, const int32 *in_labels,
		const int32 *out_labels, BaseFloat* weights, 
		const int32* statesinfo, int32 num_states, int32 num_pdfs, 
		bool delete_laststatesuperfinal, const int32 den_start_state)
{
	fst::VectorFst<fst::StdArc> den_fst;
	fst::ConvertSparseFstToOpenFst(indexs, in_labels,
			out_labels, weights, statesinfo, num_states, &den_fst, 
			delete_laststatesuperfinal, den_start_state );
	//std::cout << "---delete_laststatesuperfinal:" <<delete_laststatesuperfinal << std::endl;
	//fst::PrintStandardFst(den_fst);
#if HAVE_CUDA==1
	CuDevice::Instantiate().SelectGpuId("yes");
	CuDevice::Instantiate().AllowMultithreading();
#endif
	_den_graph = new DenominatorGraph(den_fst, num_pdfs);
	//std::cout << "DenominatorGraphSaver ok" << std::endl;
}

bool EqualDenGraph(DenominatorGraph &den_graph1, DenominatorGraph &den_graph2)
{
	int32 num_state1 = den_graph1.NumStates();
	int32 num_state2 = den_graph2.NumStates();
	if(num_state1 != num_state2)
		return false;
	const Int32Pair *fw1trans = den_graph1.ForwardTransitions();
	const Int32Pair *fw2trans = den_graph2.ForwardTransitions();
	const Int32Pair *bw1trans = den_graph1.BackwardTransitions();
	const Int32Pair *bw2trans = den_graph2.BackwardTransitions();

	for(int s=0;s<num_state1; s++)
	{
		if(fw1trans[s].first != fw2trans[s].first ||
				fw1trans[s].second != fw2trans[s].second)
			return false;
		if(bw1trans[s].first != bw2trans[s].first ||
				bw1trans[s].second != bw2trans[s].second)
			return false;
	}
	const DenominatorGraphTransition *den1_trans = den_graph1.Transitions();
	const DenominatorGraphTransition *den2_trans = den_graph2.Transitions();
	for(int s=0;s<182460;s++)
	{
		if(den1_trans[s].transition_prob != den2_trans[s].transition_prob ||
				den1_trans[s].pdf_id != den2_trans[s].pdf_id || 
				den1_trans[s].hmm_state != den2_trans[s].hmm_state)
			return false;
	}
	//const CuVector<BaseFloat> &init_prob1 = den_graph1.InitialProbs();
	//const CuVector<BaseFloat> &init_prob2 = den_graph2.InitialProbs();
	//for(int i =0 ; i < init_prob1.Dim();i++)
	//{
	//	if(init_prob1(i) != init_prob2(i))
	//		return false;
	//}
	return true;
}

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
* acoustic_scale   : acoustic scale
* gradient (outptu): it shape same as nnet_out
* loss             : it loss . which has dimension (n)
*
* */

bool ChainLoss(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat supervision_weights, const int32 supervision_num_sequences, 
		const int32 supervision_frames_per_sequence, const int32 supervision_label_dim, 
		const BaseFloat* nnet_out,
		int32 rows, int32 batch_size, int32 cols,
		// denominator fst
		const int32 *den_indexs, const int32 *den_in_labels, 
		const int32 *den_out_labels, BaseFloat* den_weights, 
		const int32* den_statesinfo, const int32 den_start_state,
		const int32 den_num_states,
		BaseFloat* gradient,
		BaseFloat* objf,
		float l2_regularize, float leaky_hmm_coefficient, float xent_regularize,
		fst::VectorFst<fst::StdArc> *den_fst_test, DenominatorGraph * den_graph_test, chain::Supervision *merge_supervision_test)
{
#ifdef DEBUG_SPEED
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
#endif

	bool delete_laststatesuperfinal = true;
	DenominatorGraphSaver den_graph_saver;
	den_graph_saver.Init(den_indexs, den_in_labels, den_out_labels, 
			den_weights, den_statesinfo, 
			den_num_states, supervision_label_dim,
			delete_laststatesuperfinal, den_start_state);


	bool ret = ChainLossDen(indexs, in_labels, out_labels, weights, statesinfo, num_states,
			max_num_arcs, max_num_states,
			supervision_weights, supervision_num_sequences, supervision_frames_per_sequence, supervision_label_dim,
			nnet_out, rows, batch_size, cols,
			den_graph_saver, 
			gradient,
			objf,
			l2_regularize, leaky_hmm_coefficient, xent_regularize);

#ifdef DEBUG_SPEED
	gettimeofday(&end, NULL);
	std::cout << "DEBUG_SPEED : " << __FILE__ << " : ChainLossDen :"
		<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
#endif
	return ret;
}

bool BatchFst(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states, const int32 max_num_arcs, const int32 max_num_states, 
		const int32 batch_size,
		std::vector<fst::VectorFst<fst::StdArc> > *fst_v)
{
	// first process fst_v
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
		const BaseFloat* cur_weights = weights + i * max_num_arcs;
		fst::VectorFst<fst::StdArc> fst;
		bool ret = fst::ConvertSparseFstToOpenFst(cur_indexs, cur_in_labels, 
				cur_out_labels, cur_weights, cur_statesinfo, cur_num_states, &fst, true, 0);
		if (ret != true)
		{
			return false;
		}
		fst_v->push_back(fst);
	} // fst ok
	return true;
}

bool ChainLossDen(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		const int32 *num_states,
		const int32 max_num_arcs, const int32 max_num_states,
		const BaseFloat supervision_weights, const int32 supervision_num_sequences, 
		const int32 supervision_frames_per_sequence, const int32 supervision_label_dim, 
		const BaseFloat* nnet_out,
		int32 rows, int32 batch_size, int32 cols,
		// denominator fst
		DenominatorGraphSaver &den_graph_saver,
		// output
		BaseFloat* gradient,
		BaseFloat* objf,
		float l2_regularize, float leaky_hmm_coefficient, float xent_regularize)
{
#ifdef DEBUG_SPEED
	struct timeval start;
	struct timeval end;
	gettimeofday(&start, NULL);
	std::cout << "l2_regularize: " << l2_regularize << " leaky_hmm_coefficient: " << leaky_hmm_coefficient
		<< " xent_regularize: " << xent_regularize << std::endl;
	std::cout << "---start ChainLossDen calculate" << std::endl;
#endif
//#if HAVE_CUDA==1
//	CuDevice::Instantiate().SelectGpuId("yes");
//	//CuDevice::Instantiate().AllowMultithreading();
//#endif
	DenominatorGraph *den_graph = den_graph_saver.GetDenGraph();
	// convert fst
	std::vector<fst::VectorFst<fst::StdArc> > fst_v;
	bool ret = BatchFst(indexs, in_labels, out_labels, weights, statesinfo, num_states, 
			max_num_arcs, max_num_states, batch_size, &fst_v);
	//std::cout << "---BatchFst ok" << std::endl;
	if(ret == false)
	{
		std::cerr << "batch fst failed." << std::endl;
		return ret;
	}
	// supervision merge
	std::vector<const chain::Supervision*> input_supervision_point;
	input_supervision_point.resize(batch_size);
	std::vector<chain::Supervision> input_supervision;
	input_supervision.resize(batch_size);
	for(int32 i=0; i < batch_size; i++)
	{
		chain::Supervision &supervision = input_supervision[i];
		supervision.weight = supervision_weights;
		supervision.num_sequences = supervision_num_sequences;
		supervision.frames_per_sequence = supervision_frames_per_sequence;
		supervision.label_dim = supervision_label_dim;
		supervision.fst = fst_v[i];
		input_supervision_point[i] = &input_supervision[i];
		//std::cout << "Supervision info:"
		//	<< "\nweight              :" << supervision.weight
		//	<< "\nnum_sequences       :" << supervision.num_sequences
		//	<< "\nframes_per_sequence :" << supervision.frames_per_sequence
		//	<< "\nlabel_dim           :" << supervision.label_dim << std::endl;
		//std::cout << i << " fst: " << std::endl;
		//fst::PrintStandardFst(supervision.fst);
	}
	chain::Supervision output_supervision;
	MergeSupervision(input_supervision_point,
			&output_supervision);

	chain::Supervision merge_supervision;
	merge_supervision.Swap(&output_supervision);
	//std::cout << "---MergeSupervision ok" << std::endl;
	// remove eps
	//fst::RmEpsilon(&merge_supervision.fst);
	//fst::TopSort(&merge_supervision.fst);

	// supervision ok
#ifdef DEBUG_SPEED
	gettimeofday(&end, NULL);
	std::cout << "DEBUG_SPEED : " << __FILE__ << " : convert fst time:"
		<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
	gettimeofday(&start, NULL);
#endif
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
	//if (use_xent)
	//	xent_deriv.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
	//			kUndefined);
	
	BaseFloat *tot_objf = objf, 
			  *tot_l2_term = objf+1, 
			  *tot_weight =objf+2;
#ifdef DEBUG_SPEED
	gettimeofday(&end, NULL);
	std::cout << "DEBUG_SPEED : " << __FILE__ << " : nnet time:"
		<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
	gettimeofday(&start, NULL);
#endif
	//std::cout << "---Data prepare ok" << std::endl;
	//std::cout << "supervisoin info:" 
	//	<< "\nweight:             "<< merge_supervision.weight 
	//	<< "\nnum_sequences:      " << merge_supervision.num_sequences 
	//	<< "\nframes_per_sequence:" << merge_supervision.frames_per_sequence 
	//	<< "\nlabel_dim:          " << merge_supervision.label_dim <<  std::endl;
	//std::cout << "nnet_output info:(row,col)" << nnet_output.NumRows() << "," 
	//	<< nnet_output.NumCols() << std::endl;
	ComputeChainObjfAndDeriv(opts, *den_graph, merge_supervision, nnet_output, 
			tot_objf, tot_l2_term, tot_weight,
			&nnet_output_deriv, 
			(use_xent ? &xent_deriv : NULL));
	*tot_objf = *tot_objf / (*tot_weight);
	// loss nnet_output_deriv
#ifdef DEBUG_SPEED
	gettimeofday(&end, NULL);
	std::cout << "DEBUG_SPEED : " << __FILE__ << " : chain loss time:"
		<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
	std::cout << "tot_objf:" << *tot_objf
		<< "\ntot_l2_term:" << *tot_l2_term
		<< "\ntot_weight:" << *tot_weight << std::endl;
	gettimeofday(&start, NULL);
#endif
	if (use_xent) 
	{
		// this block computes the cross-entropy objective.
		CuMatrix<BaseFloat> xent_output;
		xent_output.Resize(nnet_output.NumRows(), nnet_output.NumCols(),
				kUndefined);
		xent_output.LogSoftMaxPerRow(nnet_output);

		// at this point, xent_deriv is posteriors derived from the numerator
		// computation.  note, xent_objf has a factor of '.supervision.weight'
		BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
		//std::cout << "xent_objf:" << xent_objf << std::endl;
	}
	//if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0)
	//{
	//}
	if (use_xent) 
	{
		xent_deriv.Scale(opts.xent_regularize);
		// if use xent add loss
		nnet_output_deriv.AddMat(1.0, xent_deriv);
	}
	nnet_output_deriv.Scale(-1.0);

#ifdef DEBUG_SPEED
	gettimeofday(&end, NULL);
	std::cout << "DEBUG_SPEED : " << __FILE__ << " : xent_deriv time:"
		<< (end.tv_sec - start.tv_sec)+(end.tv_usec-start.tv_usec)*1.0/1e6<< std::endl;
	std::cout << "end ChainLossDen calculate" << std::endl;
#endif
	return true;
}

} // namespace hubo
