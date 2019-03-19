
#include <vector>
#include <iostream>
#include <thread>
#include <cassert>
#include "sparse-lattice-function.h"
#include "matrix.h"
#include "loss.h"

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
		BaseFloat *loss, bool drop_frames)
{
	//int32 num_classes_raw = cols;
	std::vector<Lattice> lat_v;
	std::vector<Matrix<const BaseFloat> > nnet_out_h_v;
	std::vector<Matrix<BaseFloat> > nnet_diff_h_v;
	std::vector<const int32*> labels_v;
	std::vector<std::thread> threads;
	// first process lat_v and nnet_out_h_v
	for(int32 i=0; i < batch_size; i++)
	{
		// first get state number
		int32 cur_num_states = num_states[i];
		const int32 *cur_statesinfo = statesinfo + i * max_num_states * 2;
		// second get arc number
		//int32 cur_num_arcs = cur_statesinfo[(cur_num_states-1)*2] + cur_statesinfo[(cur_num_states-1)*2+1];
		const int32 *cur_indexs = indexs + i * max_num_arcs * 2;
		const int32 *cur_pdf_values = pdf_values + i * max_num_arcs;
		BaseFloat* cur_lm_ws = lm_ws + i * max_num_arcs;
		BaseFloat* cur_am_ws = am_ws + i * max_num_arcs;

		lat_v.push_back(Lattice(cur_indexs, cur_pdf_values, cur_lm_ws, cur_am_ws, cur_statesinfo, cur_num_states));

		// process nnet_out
		nnet_out_h_v.push_back(Matrix<const BaseFloat>(nnet_out + i * cols * rows, sequence_length[i], cols, batch_size * cols));
		nnet_diff_h_v.push_back(Matrix<BaseFloat>(gradient + i * cols * rows, sequence_length[i], cols, batch_size * cols));

		labels_v.push_back(labels + i * rows);
		
		// calculate mmi gradient.
		// threading calculate.
		threads.push_back(std::thread(MMIOneLoss, &lat_v[i], &nnet_out_h_v[i], labels_v[i], &nnet_diff_h_v[i], acoustic_scale, &loss[i], drop_frames));

	}

	// pauses until all thread finish.
	for(int32 i=0; i < batch_size; i++)
	{
		threads[i].join();
	}
	return true;
}

/* lat (input)         :
 * nnet_out_h          :
 * labels              :
 *
 * nnet_diff_h(output) :
 * loss                : 
 * return              : loss
 * */
void MMIOneLoss(Lattice *lat, Matrix<const BaseFloat> *nnet_out_h, const int32 *labels,
		Matrix<BaseFloat> *nnet_diff_h, BaseFloat acoustic_scale, BaseFloat *loss, bool drop_frames)
{
	//Lattice lat(indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states);
	//Matrix<BaseFloat> nnet_out_h(nnet_out, length, cols);
	
	std::vector<int32> state_times;
	int32 max_time = LatticeStateTimes(*lat, &state_times);

	int32 num_frames = nnet_out_h->NumRows();
	assert(max_time == num_frames);

	// rescore the latice
	LatticeAcousticRescore(*nnet_out_h, state_times, *lat);
	if (acoustic_scale != 1.0)
		lat->ScaleAmWeight(acoustic_scale);

	// get the posteriors 
	//Matrix<BaseFloat> nnet_diff_h(gradient, length, cols);
	BaseFloat acoustic_like_sum = 0.0;
	BaseFloat lat_like = LatticeForwardBackward(*lat,
			&acoustic_like_sum, *nnet_diff_h);

	// Calculate the MMI-objective function,
	// it's not must be calculate.
	// Calculate the likelihood of correct path from acoustic score,
	// the denominator likelihood is the total likelihood of the lattice.
	double path_ac_like = 0.0;
	for (int32 t = 0; t < num_frames; t++)
	{
		int32 pdf = labels[t];
		path_ac_like += (*nnet_out_h)(t, pdf);
	}
	path_ac_like *= acoustic_scale;
	double mmi_obj = path_ac_like - lat_like;

	// Note: numerator likelihood does not include graph score,
	// while denominator likelihood contains graph scores.
	// The result is offset at the MMI-objective.
	// However the offset is constant for given alignment,
	// so it does not change accross epochs.
	double post_on_ali = 0.0;
	for (int32 t = 0; t < num_frames; t++)
	{
		int32 pdf = labels[t];
		double posterior = (*nnet_diff_h)(t, pdf);
		post_on_ali += posterior;
	}
	// Report
	std::cout << "Utterance : Average MMI obj. value = "
		<< (mmi_obj/num_frames) << " over " << num_frames << " frames."
		<< " (Avg. den-posterior on ali " << post_on_ali / num_frames << ")" << std::endl;

	// subtract the pdf-Viterbi-path,
	for (int32 t = 0; t < nnet_diff_h->NumRows(); t++)
	{
		int32 pdf = labels[t];
		(*nnet_diff_h)(t, pdf) -= 1.0;
	}

	// Search for the frames with num/den mismatch,
	int32 frm_drop = 0;
	std::vector<int32> frm_drop_vec;
	for(int32 t=0; t<num_frames; t++)
	{
		int32 pdf = labels[t];
		double posterior = (*nnet_diff_h)(t, pdf);
		if (posterior < 1e-20)
		{
			frm_drop++;
			frm_drop_vec.push_back(t);
		}
	}
	int32 num_frm_drop = 0;
	// Drop mismatched frames from the training by zeroing the derivative,
	if (drop_frames)
	{
		for (int32 i = 0; i < frm_drop_vec.size(); i++)
		{
			nnet_diff_h->SetRowZero(frm_drop_vec[i]);
		}
		num_frm_drop += frm_drop;
	}
	*loss = mmi_obj/num_frames;
}




} // namespace
