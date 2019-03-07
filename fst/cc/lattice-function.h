#ifndef __LATTICE_FUNCTION_H__
#define __LATTICE_FUNCTION_H__
#include <vector>
#include <limits>
#include <cassert>


typedef float BaseFloat;
typedef int int32;
typedef int StateId;
const double kLogZeroDouble = -std::numeric_limits<double>::infinity();

struct Arc
{
	int32   _pdf;
	StateId _nextstate;
	BaseFloat _lm_weight;
	BaseFloat _am_weight;
	
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
	Lattice(BaseFloat *indexs,int32 *pdf_values,
			BaseFloat* lm_ws, BaseFloat* am_ws, int32* statesinfo, 
			int32 num_states):
		_indexs(indexs), _pdf_values(pdf_values),_lm_ws(lm_ws), 
		_am_ws(am_ws), _statesinfo(statesinfo), _num_states(num_states){ }
	Lattice():
		_indexs(NULL), _pdf_values(NULL),_lm_ws(NULL),
		_am_ws(NULL), _statesinfo(NULL), _num_states(NULL){ }

	StateId NumStates()
	{
		return _num_states;
	}

	bool GetArc(StateId s, int32 arcid, Arc *arc)
	{
		if(arcid >= GetStateArcNums(s))
			return false;
		else
		{
			int32 offset = statesinfo[s*2];
			assert(indexs[offset*2] == s)
			StateId nextstate = indexs[offset*2+1];
			BaseFloat lm_weight = _lm_ws[offset];
			BaseFloat am_weight = _am_ws[offset];
			int32 pdf = _pdf_values[offset]
			arc->SetArc(pdf, nextstate, lm_weight, am_weight);
			return true;
		}
	}
	
	int32 GetStateArcNums(StateId s)
	{
		return statesinfo[s*2+1];
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
private:
	BaseFloat *_indexs; // fst cur_state and next_state
	int32 *_pdf_values; // map [cur_state, next_state]
	BaseFloat* _lm_ws;  // map [cur_state, next_state]
	BaseFloat* _am_ws;  // map [cur_state, next_state]
	int32* _statesinfo; // save every state save arcs number [ state_startoffset, number_arc
	int32 num_states;   // state number
};

#endif
