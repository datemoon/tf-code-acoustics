#ifndef __SPARSE_LATTICE_FUNCTION_H__
#define __SPARSE_LATTICE_FUNCTION_H__
#include <vector>
#include <limits>
#include <cassert>
#include <sys/types.h>
#include <stdio.h>
#include "base-math.h"
#include "matrix.h"

namespace hubo {

typedef int StateId;
//const double kLogZeroDouble = -std::numeric_limits<double>::infinity();

struct Arc
{
	int32   _pdf;
	StateId _nextstate;
	BaseFloat _lm_weight;
	BaseFloat _am_weight;
	
	Arc() { }
	Arc(int32 pdf, StateId nextstate, BaseFloat lm_weight, BaseFloat am_weight):
		_pdf(pdf), _nextstate(nextstate), 
		_lm_weight(lm_weight), _am_weight(am_weight) { }

	void SetArc(int32 pdf, StateId nextstate, BaseFloat lm_weight, BaseFloat am_weight)
	{
		_pdf = pdf;
		_nextstate = nextstate;
		_lm_weight = lm_weight;
		_am_weight = am_weight;
	}

	BaseFloat Value()
	{
		return _lm_weight+_am_weight;
	}
};

class Lattice
{
public:
	Lattice(int32 *indexs,int32 *pdf_values,
			BaseFloat* lm_ws, BaseFloat* am_ws, int32* statesinfo, 
			int32 num_states):
		_indexs(indexs), _pdf_values(pdf_values),_lm_ws(lm_ws), 
		_am_ws(am_ws), _statesinfo(statesinfo), _num_states(num_states){ }
	Lattice():
		_indexs(NULL), _pdf_values(NULL),_lm_ws(NULL),
		_am_ws(NULL), _statesinfo(NULL), _num_states(0){ }
//	{
//		_indexs = NULL;
//		_pdf_values = NULL;
//	   	_lm_ws = NULL;
//		_am_ws = NULL;
//		_statesinfo = NULL; 
//		_num_states = NULL;
//	}

	StateId NumStates()
	{
		return _num_states;
	}

	void SetArcAmValue(StateId s, int32 arcid, BaseFloat am_value)
	{
		int32 offset = _statesinfo[s*2];
		assert(_indexs[offset*2] == s);
		_am_ws[offset] = am_value;
	}

	bool GetArc(StateId s, int32 arcid, Arc *arc)
	{
		if(arcid >= GetStateArcNums(s))
			return false;
		else
		{
			int32 offset = _statesinfo[s*2] + arcid;
			assert(_indexs[offset*2] == s);
			StateId nextstate = _indexs[offset*2+1];
			BaseFloat lm_weight = _lm_ws[offset];
			BaseFloat am_weight = _am_ws[offset];
			int32 pdf = _pdf_values[offset];
			arc->SetArc(pdf, nextstate, lm_weight, am_weight);
			return true;
		}
	}
	
	int32 GetStateArcNums(StateId s)
	{
		return _statesinfo[s*2+1];
	}

	bool IsFinal(StateId s)
	{
		if(GetStateArcNums(s) == 0)
			return true;
		else
			return false;
	}

	BaseFloat Final(StateId s)
	{
		if(GetStateArcNums(s) == 0)
			return 0;
		else
			return -kLogZeroFloat;
	}

	void Print()
	{
		for(int s=0;s<_num_states;++s)
		{
			int32 n = 0;
			Arc arc;
			while(GetArc(s, n, &arc) == true)
			{
				// instate tostate ilabel olabel lm_ws am_ws
				printf("%d %d %d %f,%f\n",
						s, arc._nextstate,
						arc._pdf, arc._lm_weight, arc._am_weight);
				n++;
			}
			/*int offset = _statesinfo[s*2];
			for(int n = offset ; n < offset+_statesinfo[s*2+1]; ++n)
			{
				// instate tostate ilabel olabel lm_ws am_ws
				printf("%d %d %d %f,%f\n",
						_indexs[2*n], _indexs[2*n+1],
						_pdf_values[n], _lm_ws[n], _am_ws[n]);
			}*/
		}
	}
private:
	int32 *_indexs; // fst cur_state and next_state
	int32 *_pdf_values; // map [cur_state, next_state]
	BaseFloat* _lm_ws;  // map [cur_state, next_state]
	BaseFloat* _am_ws;  // map [cur_state, next_state]
	int32* _statesinfo; // save every state save arcs number [ state_startoffset, number_arc
	int32 _num_states;   // state number
};

/// This function iterates over the states of a topologically sorted Sparse lattice and
/// counts the time instance corresponding to each state. The times are returned
/// in a vector of integers 'times' which is resized to have a size equal to the
/// number of states in the Sparse lattice. The function also returns the maximum time
/// in the Sparse lattice (this will equal the number of frames in the file).
int32 LatticeStateTimes(Lattice &lat, std::vector<int32> *times);

void LatticeAcousticRescore(const Matrix<float> &log_like,
		const std::vector<int32> &state_times,
		Lattice &lat);

BaseFloat LatticeForwardBackward(Lattice &lat,
		BaseFloat* acoustic_like_sum , Matrix<float> &nnet_diff_h);


} // namespace hubo
#endif
