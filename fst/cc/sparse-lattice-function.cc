#include <vector>
#include <iostream>
#include "sparse-lattice-function.h"
#include "base-math.h"
#include "matrix.h"
namespace hubo {

using namespace std;



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

void LatticeAcousticRescore(const Matrix<float> &log_like,
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
				n++;
			}
		}
	}
}



} // namespace hubo
