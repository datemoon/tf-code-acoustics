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

/*
 * Lattice have only one final and all state have not weight.
 * It must be supper lattice final.
 * */
class Lattice
{
public:
	// indexs     : the same as arc number,recode [instate, tostate]
	// pdf_values : the same as arc number,recode [in_label]
	// lm_ws      : the same as arc number
	// am_ws      : the same as arc number
	// statesinfo : length is num_states * 2, recode [state_start_offset, narcs]
	// num_states : state number
	Lattice(const int32 *indexs, const int32 *pdf_values,
			BaseFloat* lm_ws, BaseFloat* am_ws, const int32* statesinfo, 
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
		_am_ws[offset+arcid] = am_value;
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

	void ScaleAmWeight(BaseFloat acoustic_scale)
	{
		for(int i = 0; 
				i < _statesinfo[(_num_states-1)*2]+_statesinfo[(_num_states-1)*2+1];i++)
			_am_ws[i] *= acoustic_scale;
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
	void PrintInfo()
	{
		int num_arcs = _statesinfo[2*(_num_states-1)] + _statesinfo[2*(_num_states-1)+1];
		std::cout << "indexs:" << num_arcs << std::endl;
		for(int a=0;a<num_arcs;a++)
		{
			std::cout << _indexs[2*a] << " " << _indexs[2*a+1] << std::endl;
		}
		std::cout << "pdf_values:" << 1 << std::endl;
		for(int a=0;a<num_arcs;a++)
		{
			std::cout << _pdf_values[a] << " ";
		}
		std::cout << std::endl;
		std::cout << "lm_ws:" << 1 << std::endl;
		for(int a=0;a<num_arcs;a++)
		{
			std::cout << _lm_ws[a] << " ";
		}
		std::cout << std::endl;

		std::cout << "am_ws:" << 1 << std::endl;
		for(int a=0;a<num_arcs;a++)
		{
			std::cout << _am_ws[a] << " ";
		}
		std::cout << std::endl;

		std::cout << "statesinfo:" << _num_states << std::endl;
		for(int s=0;s<_num_states;s++)
		{
			std::cout << _statesinfo[2*s] << " " << _statesinfo[2*s+1] << std::endl;
		}
	}
private:
	const int32 *_indexs; // fst cur_state and next_state
	const int32 *_pdf_values; // map [cur_state, next_state]
	BaseFloat* _lm_ws;  // map [cur_state, next_state]
	BaseFloat* _am_ws;  // map [cur_state, next_state]
	const int32* _statesinfo; // save every state save arcs number [ state_startoffset, number_arc]
	int32 _num_states;   // state number
};

/// This function iterates over the states of a topologically sorted Sparse lattice and
/// counts the time instance corresponding to each state. The times are returned
/// in a vector of integers 'times' which is resized to have a size equal to the
/// number of states in the Sparse lattice. The function also returns the maximum time
/// in the Sparse lattice (this will equal the number of frames in the file).
int32 LatticeStateTimes(Lattice &lat, std::vector<int32> *times);

void LatticeAcousticRescore(const Matrix<const float> &log_like,
		const std::vector<int32> &state_times,
		Lattice &lat);

BaseFloat LatticeForwardBackward(Lattice &lat,
		BaseFloat* acoustic_like_sum , Matrix<float> &nnet_diff_h);

/*
 * lat               (input)  : sparse lattice and it must be only one final and top sort.
 * silence_phones    (input)  : silence phone list
 * pdf_to_phone      (input)  : pdf map phone, it's 2D,(pdf_id, 2)[pdf_id, phone_id]
 * num_pdf           (input)  : pdf_id list , it's 1D
 * criterion         (input)  : "smbr" or "mpe"
 * one_silence_class (input)  : true or false
 * nnet_diff_h       (output) : output loss
 * */
BaseFloat LatticeForwardBackwardMpeVariants(Lattice &lat,
		const std::vector<int32> &silence_phones,
		Matrix<const int32> &pdf_to_phone,
		const int32 *num_pdf,
		std::string criterion,
		bool one_silence_class,
		Matrix<float> &nnet_diff_h);


} // namespace hubo
#endif
