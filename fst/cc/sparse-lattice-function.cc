#include <vector>
#include <iostream>
#include <algorithm>
#include "sparse-lattice-function.h"
#include "base-math.h"
#include "matrix.h"
namespace hubo {

using namespace std;

/*
 * only calculate one sentence LatticeForwardBackwardMpeVariants
 * */
BaseFloat LatticeForwardBackwardMpeVariants(Lattice &lat,
		const std::vector<int32> &silence_phones,
		Matrix<const int32> &trans,
		const int32 *num_ali,
		std::string criterion,
		bool one_silence_class,
		Matrix<float> &nnet_diff_h,
		BaseFloat &min, BaseFloat &max)
{
	int32 max_time = nnet_diff_h.NumRows();
	assert(criterion == "mpfe" || criterion == "smbr");
	bool is_mpfe = (criterion == "mpfe");
	// Input lattice must be topologically sorted.
	//assert(lat.Start() == 0);
	StateId num_states = lat.NumStates();
	nnet_diff_h.SetZero();

	vector<int32> state_times;
	state_times.resize(num_states, -1);

	vector<double> alpha(num_states, kLogZeroDouble),
		alpha_smbr(num_states, 0), //forward variable for sMBR
		beta(num_states, kLogZeroDouble),
		beta_smbr(num_states, 0); //backward variable for sMBR	

	double tot_forward_prob = kLogZeroDouble;
	double tot_forward_score = 0;

	alpha[0] = 0.0;
	state_times[0] = 0;

	// First Pass Forward,
	for (StateId s = 0; s < num_states; s++)
	{
		int32 cur_time = state_times[s];
		double this_alpha = alpha[s];
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			double arc_like = - arc.Value();
			alpha[arc._nextstate] = LogAdd(alpha[arc._nextstate], this_alpha + arc_like);
			n++;
			if (arc._pdf != 0)
			{
				if(state_times[arc._nextstate] == -1)
					state_times[arc._nextstate] = cur_time + 1;
				else
					assert(state_times[arc._nextstate] == cur_time + 1);
			}
			else
			{
				if(state_times[arc._nextstate] == -1)
					state_times[arc._nextstate] = cur_time ;
				else
					 assert(state_times[arc._nextstate] == cur_time );
			}
		}
		if(lat.IsFinal(s) == true)
		{
			double final_like = this_alpha;
			tot_forward_prob = LogAdd(tot_forward_prob, final_like);
			assert(state_times[s] == max_time &&
					"Lattice is inconsistent (final-prob not at max_time)");
		}
	}
	// First Pass Backward,
	for (StateId s = num_states-1; s >= 0; s--)
	{
		BaseFloat f = lat.Final(s);
		double this_beta = - f;
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			double arc_like = - arc.Value(),
				   arc_beta  = beta[arc._nextstate] + arc_like;
			this_beta = LogAdd(this_beta, arc_beta);
			n++;
		}
		beta[s] = this_beta;
	}
	// First Pass Forward-Backward Check
	double tot_backward_prob = beta[0];
	// may loose the condition somehow here 1e-6 (was 1e-8)
	if(!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-6))
	{
		cout << "Total forward probability over lattice = " << tot_forward_prob
		   << ", while total backward probability = " << tot_backward_prob
	   	   << endl;
	}

	alpha_smbr[0] = 0.0;
	// Second Pass Forward, calculate forward for MPFE/SMBR
	for (StateId s = 0; s < num_states; s++)
	{
		double this_alpha = alpha[s];
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			double arc_like = - arc.Value();
			double frame_acc = 0.0;
			if (arc._pdf != 0)
			{
				int32 cur_time = state_times[s];
				int32 phone = trans(arc._pdf, 2),
					  ref_phone = trans(num_ali[cur_time], 2);
				bool phone_is_sil = std::binary_search(silence_phones.begin(),
						silence_phones.end(),
						phone),
					 ref_phone_is_sil = std::binary_search(silence_phones.begin(),
							 silence_phones.end(),
							 ref_phone),
					 both_sil = phone_is_sil && ref_phone_is_sil;
				if (!is_mpfe) 
				{ // smbr.
					int32 pdf = trans(arc._pdf, 1),
						  ref_pdf = trans(num_ali[cur_time], 1);
					if (!one_silence_class)  // old behavior
						frame_acc = (pdf == ref_pdf && !phone_is_sil) ? 1.0 : 0.0;
					else
						frame_acc = (pdf == ref_pdf || both_sil) ? 1.0 : 0.0;
				}
				else
				{
					if (!one_silence_class)  // old behavior
						frame_acc = (phone == ref_phone && !phone_is_sil) ? 1.0 : 0.0;
					else
						frame_acc = (phone == ref_phone || both_sil) ? 1.0 : 0.0;
				}
			}
			double arc_scale = Exp(alpha[s] + arc_like - alpha[arc._nextstate]);
			alpha_smbr[arc._nextstate] += arc_scale * (alpha_smbr[s] + frame_acc);
		} // all arc
		if(lat.IsFinal(s) == true)
		{
			double final_like = this_alpha;
			double arc_scale = Exp(final_like - tot_forward_prob);
			tot_forward_score += arc_scale * alpha_smbr[s];
		}
	} // all state

	// Second Pass Backward, collect Mpe style posteriors
	for (StateId s = num_states-1; s >= 0; s--)
	{
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			double arc_like = - arc.Value(),
				   arc_beta  = beta[arc._nextstate] + arc_like;
			double frame_acc = 0.0;
			int32 transition_id = arc._pdf;
			if(transition_id != 0)
			{
				int32 cur_time = state_times[s];
				int32 phone = trans(transition_id, 2),
					  ref_phone = trans(num_ali[cur_time], 2);
				bool phone_is_sil = std::binary_search(silence_phones.begin(),
						silence_phones.end(), phone),
					 ref_phone_is_sil = std::binary_search(silence_phones.begin(),
							 silence_phones.end(),
							 ref_phone),
					 both_sil = phone_is_sil && ref_phone_is_sil;
				if (!is_mpfe) 
				{ // smbr.
					int32 pdf = trans(transition_id, 1),
						  ref_pdf = trans(num_ali[cur_time], 1);
					if(!one_silence_class) // old behavior
						frame_acc = (pdf == ref_pdf && !phone_is_sil) ? 1.0 : 0.0;
					else
						frame_acc = (pdf == ref_pdf || both_sil) ? 1.0 : 0.0;
				}
				else
				{
					if (!one_silence_class)  // old behavior
						frame_acc = (phone == ref_phone && !phone_is_sil) ? 1.0 : 0.0;
					else
						frame_acc = (phone == ref_phone || both_sil) ? 1.0 : 0.0;					
				}
			}
			double arc_scale = Exp(beta[arc._nextstate] + arc_like - beta[s]);
			// check arc_scale NAN,
			// this is to prevent partial paths in Lattices
			// i.e., paths don't survive to the final state
			if (KALDI_ISNAN(arc_scale)) 
				arc_scale = 0;
			beta_smbr[s] += arc_scale * (beta_smbr[arc._nextstate] + frame_acc);

			if (transition_id != 0) 
			{ // Arc has a transition-id on it [not epsilon]
				double posterior = Exp(alpha[s] + arc_beta - tot_forward_prob);
				double acc_diff = alpha_smbr[s] + frame_acc + beta_smbr[arc._nextstate]
					- tot_forward_score;
				double posterior_smbr = posterior * acc_diff;

				int32 pdf_id = trans(transition_id ,1);
				nnet_diff_h(state_times[s], pdf_id-1) -= posterior_smbr;
				// save max and min loss
				if(-posterior_smbr > max)
					max = -posterior_smbr;
				if(-posterior_smbr < min)
					min = -posterior_smbr;
//				(*post)[state_times[s]].push_back(std::make_pair(transition_id,
//							static_cast<BaseFloat>(posterior_smbr)));
			}
		} // all arc
	} // all state

	// Second Pass Forward Backward check
	double tot_backward_score = beta_smbr[0];  // Initial state id == 0
	// may loose the condition somehow here 1e-5/1e-4
	if (!ApproxEqual(tot_forward_score, tot_backward_score, 1e-4)) 
	{
		std::cout << "Total forward score over lattice = " << tot_forward_score
			<< ", while total backward score = " << tot_backward_score << std::endl;
	}

	// Output the computed posteriors
	return tot_forward_score;
}


/*
 * only calculate one sentence LatticeForwardBackward
 * */
BaseFloat LatticeForwardBackward(Lattice &lat,
		BaseFloat* acoustic_like_sum , Matrix<float> &nnet_diff_h)
{
	*acoustic_like_sum = 0.0;
	nnet_diff_h.SetZero();
	StateId num_states = lat.NumStates();

	// Make sure the lattice is topologically sorted.
	vector<int32> state_times;
	state_times.resize(num_states, -1);
	vector<double> alpha(num_states, kLogZeroDouble);
	vector<double> &beta(alpha); // we re-use the same memory for
	// this, but it's semantically distinct so we name it differently.
	double tot_forward_prob = kLogZeroDouble;

	alpha[0] = 0.0;
	state_times[0] = 0;
	// Propagate alphas forward.
	for (StateId s = 0; s < num_states; s++) 
	{
		int32 cur_time = state_times[s];
		double this_alpha = alpha[s];
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			double arc_like = - arc.Value();
			alpha[arc._nextstate] = LogAdd(alpha[arc._nextstate], this_alpha + arc_like);
			n++;
			if (arc._pdf != 0)
			{
				if(state_times[arc._nextstate] == -1)
					state_times[arc._nextstate] = cur_time + 1;
				else
					assert(state_times[arc._nextstate] == cur_time + 1);
			}
			else
			{
				if(state_times[arc._nextstate] == -1)
					state_times[arc._nextstate] = cur_time ;
				else
					assert(state_times[arc._nextstate] == cur_time );
			}
		}
		if(lat.IsFinal(s) == true)
		{
			double final_like = this_alpha;
			tot_forward_prob = LogAdd(tot_forward_prob, final_like);
		}
	}
	// backforward
	for (StateId s = num_states-1; s >= 0; s--)
	{
		BaseFloat f = lat.Final(s);
		double this_beta = - f;
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			double arc_like = - arc.Value(),
				   arc_beta  = beta[arc._nextstate] + arc_like;
			this_beta = LogAdd(this_beta, arc_beta);
			int32 pdf_id = arc._pdf;

			// The following "if" is an optimization to avoid un-needed exp().
			if (pdf_id != 0 || acoustic_like_sum != NULL)
			{
				double posterior = Exp(alpha[s] + arc_beta - tot_forward_prob);
				if (pdf_id != 0) // Arc has a transition-id on it [not epsilon]
					// Now combine any posteriors with the same pdf-id.
					nnet_diff_h(state_times[s], pdf_id-1) += posterior;
				if (acoustic_like_sum != NULL)
					*acoustic_like_sum -= posterior * arc._am_weight;
			}
			n++;
		}
		if (acoustic_like_sum != NULL && lat.IsFinal(s) == true)
		{
			double final_logprob = 0,
				   posterior = Exp(alpha[s] + final_logprob - tot_forward_prob);
			*acoustic_like_sum -= posterior * 0;
		}
		beta[s] = this_beta;
	}
	double tot_backward_prob = beta[0];
	if (!ApproxEqual(tot_forward_prob, tot_backward_prob, 1e-8))
	{
		cout << "Total forward probability over lattice = " << tot_forward_prob
		   << ", while total backward probability = " << tot_backward_prob
	   	   << endl;
	}
	return tot_backward_prob;
}

int32 LatticeStateTimes(Lattice &lat, vector<int32> *times)
{
	int32 num_states = lat.NumStates();
	times->clear();
	times->resize(num_states, -1);
	(*times)[0] = 0;
	int32 max_times = -1;
	for (int32 s = 0; s < num_states; s++)
	{
		int32 cur_time = (*times)[s];
		if(cur_time > max_times)
			max_times = cur_time;
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			if(arc._pdf != 0)
			{ // Non-epsilon input label on arc
				if ((*times)[arc._nextstate] == -1) 
				{
					(*times)[arc._nextstate] = cur_time + 1;
				}
				else
				{
					assert((*times)[arc._nextstate] == cur_time + 1);
				}
			}
			else
			{
				if ((*times)[arc._nextstate] == -1) 
				{
					(*times)[arc._nextstate] = cur_time ;
				}
				else
				{
					assert((*times)[arc._nextstate] == cur_time );
				}
			}
			n++;
		}
	}
	return max_times;
}

void LatticeAcousticRescore(const Matrix<const float> &log_like,
		const vector<int32> &state_times,
		Lattice &lat)
{
	int32 num_states = lat.NumStates();
	for (int32 s = 0; s < num_states; s++)
	{
		int32 n = 0;
		Arc arc;
		while(lat.GetArc(s, n, &arc) == true)
		{
			if(arc._pdf != 0)
			{
				int32 t = state_times[s];
				int32 pdf = arc._pdf - 1;
				arc._am_weight -= log_like(t, pdf);
				lat.SetArcAmValue(s, n, arc._am_weight);
			}
			n++;
		}
		//std::cout << s << " "  << n << std::endl;
	}
}



} // namespace hubo
