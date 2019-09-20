
#include <fst/fstlib.h>
#include "fst-convert-openfst.h"

namespace fst
{

// using StdVectorFst = VectorFst<StdArc>;
// indexs     : the same as arc number,recode [instate, tostate]
// in_labels  : the same as arc number,recode [in_label]
// weights    : the same as arc number,recode [weight]
// statesinfo : length is num_states * 2, recode [state_start_offset, narcs]
// num_states : state number
template <class Arc>
bool ConvertSparseFstToOpenFst(const int32 *indexs, const int32 *in_labels, 
		const int32 *out_labels, const BaseFloat* weights, const int32* statesinfo, 
		int32 num_states,
		VectorFst<Arc> *fst,
		bool delete_laststatesuperfinal, int32 start_state)
//		StdVectorFst *fst)
{
	typedef typename Arc::StateId StateId;
	typedef typename Arc::Weight Weight;
	typedef typename Arc::Label Label;

	fst->DeleteStates();
	fst->AddState();

	StateId laststatesuperfinal = 0;
	if(delete_laststatesuperfinal == true)
		laststatesuperfinal = num_states-1;
	for(int s=0; s<num_states; s++)
	{
		int s_start = statesinfo[2*s+0];
		int narcs = statesinfo[2*s+1];
		for(int a = 0; a < narcs ; a++)
		{
			int cur_offset = a+s_start;
			int instate = indexs[cur_offset*2 + 0];
			int tostate = indexs[cur_offset*2 + 1];
			while(instate >= fst->NumStates())
			{
				fst->AddState();
				//printf("it shouldn't happen,instate > s");
			}
			float weight = weights[cur_offset];
			int in_label = in_labels[cur_offset];
			int out_label = out_labels[cur_offset];
			Weight w = (Weight)(weight);
			if(delete_laststatesuperfinal == true && 
					tostate == laststatesuperfinal)
			{
				fst->SetFinal(s, w);
			}
			else
			{
				fst->AddArc(instate, Arc(in_label, out_label, w, tostate));
			}
		}
		if(delete_laststatesuperfinal != true && s == num_states-1)
		{
			if(s >= fst->NumStates())
				fst->AddState();
			fst->SetFinal(s, Weight::One());
		}
	}
	fst->SetStart(start_state);
	return true;
}

template 
bool ConvertSparseFstToOpenFst<StdArc>(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		const BaseFloat* weights, const int32* statesinfo,
		int32 num_states,
		VectorFst<StdArc> *fst,
		bool delete_laststatesuperfinal, int32 start_state);

// using StdVectorFst = VectorFst<StdArc>;
// indexs     : the same as arc number,recode [instate, tostate]
// in_labels  : the same as arc number,recode [in_label]
// weights    : the same as arc number,recode [weight]
// statesinfo : length is num_states * 2, recode [state_start_offset, narcs]
// num_states : state number
VectorFst<StdArc> ConvertSparseFstToOpenFst(const int32 *indexs, const int32 *in_labels, 
		const int32 *out_labels, const BaseFloat* weights, const int32* statesinfo, 
		int32 num_states,
		bool delete_laststatesuperfinal, int32 start_state)
//		StdVectorFst *fst)
{
	typedef typename StdArc::StateId StateId;
	typedef typename StdArc::Weight Weight;
	typedef typename StdArc::Label Label;

	VectorFst<StdArc> fst;
	fst.DeleteStates();
	fst.AddState();

	StateId laststatesuperfinal = 0;
	bool delete_superfinal = false;
	if(delete_laststatesuperfinal == true)
	{
		delete_superfinal = true;
		laststatesuperfinal = num_states-1;
	}
	for(int s=0; s<num_states; s++)
	{
		int s_start = statesinfo[2*s+0];
		int narcs = statesinfo[2*s+1];
		for(int a = 0; a < narcs ; a++)
		{
			int cur_offset = a+s_start;
			int instate = indexs[cur_offset*2 + 0];
			int tostate = indexs[cur_offset*2 + 1];
			while(instate >= fst.NumStates())
			{
				fst.AddState();
				//printf("it shouldn't happen,instate > s");
			}
			float weight = weights[cur_offset];
			int in_label = in_labels[cur_offset];
			int out_label = out_labels[cur_offset];
			Weight w = (Weight)(weight);
			if(delete_laststatesuperfinal == true 
					&& tostate == laststatesuperfinal 
					&& in_label != 0
					&& out_label != 0)
			{ // delete this arc and delete superfinal
				fst.SetFinal(s, w);
			}
			else
			{ // add src
				fst.AddArc(instate, StdArc(in_label, out_label, w, tostate));
				if(tostate == laststatesuperfinal)
					delete_superfinal = false;
			}
		}
		if(s == num_states-1)
		{
			if(delete_laststatesuperfinal != true || delete_superfinal == false)
			{
				if(s >= fst.NumStates())
					fst.AddState();
				fst.SetFinal(s, Weight::One());
			}
		}
	}
	fst.SetStart(start_state);
	return fst;
}

bool MallocSparseFst(int num_states, int num_arcs,
		int32 **indexs,
		int32 **in_labels,
		int32 **out_labels,
		BaseFloat **weights,
		int32 **stateinfo)
{
	*indexs = new int32[num_arcs * 2];
	memset(*indexs, 0x00, num_arcs * 2 * sizeof(int32));

	*in_labels = new int32[num_arcs];
	memset(*in_labels, 0x00, num_arcs * sizeof(int32));

	*out_labels = new int32[num_arcs];
	memset(*out_labels, 0x00, num_arcs * sizeof(int32));

	*weights = new BaseFloat[num_arcs];
	memset(*weights, 0x00, num_arcs * sizeof(BaseFloat));

	*stateinfo = new int32[num_states * 2];
	memset(*stateinfo , 0x00, num_states * 2 * sizeof(int32));

	return true;
}


template <class Arc>
void CreateLastStateSuperFinal(VectorFst<Arc> *fst)
{
	typedef typename Arc::StateId StateId;
	typedef typename Arc::Weight Weight;
	typedef typename Arc::Label Label;
	int32 num_states = fst->NumStates();
	if(num_states <= 0)
		return;
	StateId last_superfinal_state = fst->AddState();
	fst->SetFinal(last_superfinal_state, Weight::One());
	for(int s=0; s<num_states; s++)
	{
		if(fst->Final(s) != Weight::Zero())
		{
			fst->AddArc(s, Arc(0, 0, fst->Final(s), last_superfinal_state));
		}
	}
	return ;
}
/*
 *
 * return : number states
 **/
template <class Arc>
int ConvertKaldiLatticeToSparseLattice(VectorFst<Arc> &const_fst,
		int32 **indexs,
		int32 **in_labels,
		int32 **out_labels,
		BaseFloat **weights,
		int32 **stateinfo,
		int32 *start_state)
{
	typedef typename Arc::Weight Weight;
	VectorFst<Arc> fst = const_fst;
	CreateLastStateSuperFinal(&fst);
	// check fst have only one final and the stateid is the last state.
	//TopSort(&fst);
	// inlat must be topsort and it's super final lattice(have only final state and final-probs are One()).
	int32 num_states = fst.NumStates();
	int32 num_final = 0 ;
	for(int s=0; s<num_states; s++)
	{
		if(fst.Final(s) != Weight::Zero())
			num_final ++;
	}
	int32 num_arcs = 0;
	for(int s=0; s<num_states; s++)
	{
		num_arcs += fst.NumArcs(s);
	}
	// malloc space for sparse lattice
	if(*indexs == NULL)
		MallocSparseFst(num_states, num_arcs, indexs, in_labels, out_labels, weights, stateinfo);

	int state_offset = 0;
	for(int s=0; s<num_states; s++)
	{
		int state_arcs = 0;
		for (ArcIterator<VectorFst<Arc> > aiter(fst, s); !aiter.Done(); aiter.Next())
		{
			int cur_offset = state_offset + state_arcs;
			const Arc &arc = aiter.Value();
			float weight = arc.weight.Value();
			int in_label = arc.ilabel;
			int out_label = arc.olabel;
			(*indexs)[cur_offset*2+0] = s;
			(*indexs)[cur_offset*2+1] = arc.nextstate;
			(*in_labels)[cur_offset] = in_label;
			(*out_labels)[cur_offset] = out_label;
			(*weights)[cur_offset] = weight;
			state_arcs++;
		}
		(*stateinfo)[2*s+0] = state_offset;
		(*stateinfo)[2*s+1] = state_arcs;
		state_offset += state_arcs;
	}
	*start_state = fst.Start();
	return num_states;

}

template 
int ConvertKaldiLatticeToSparseLattice<StdArc>(VectorFst<StdArc> &fst,
		int32 **indexs,
		int32 **in_labels,
		int32 **out_labels,
		BaseFloat **weights,
		int32 **stateinfo,
		int32 *start_state);

void PrintStandardFst(VectorFst<StdArc> &fst)
{
	typedef typename StdArc::Weight Weight;
	int32 num_states = fst.NumStates();
	for(int s=0; s<num_states; s++)
	{
		for (ArcIterator<VectorFst<StdArc> > aiter(fst, s); !aiter.Done(); aiter.Next())
		{
			const StdArc &arc = aiter.Value();
			std::cout << s << " " << arc.nextstate << " " << arc.ilabel << " " 
				<< arc.olabel << " " << arc.weight.Value() << std::endl;
		}
		if(fst.Final(s) != Weight::Zero())
		{
			std::cout << s << " " << fst.Final(s) << std::endl;
		}
	}
}

bool EqualSrc(const StdArc &arc1, const StdArc &arc2)
{
	if(arc1.nextstate == arc2.nextstate && 
			arc1.ilabel == arc2.ilabel &&
			arc1.olabel == arc2.olabel &&
			arc1.weight.Value() == arc2.weight.Value())
		return true;
	return false;
}

bool EqualFst(VectorFst<StdArc> &fst1, VectorFst<StdArc> &fst2)
{
	int32 num_states1 = fst1.NumStates();
	int32 num_states2 = fst2.NumStates();
	if(num_states1 != num_states2)
		return false;
	if(fst1.Start() != fst2.Start())
		return false;
	for(int s=0; s<num_states1; s++)
	{
		ArcIterator<VectorFst<StdArc> > aiter2(fst2, s);
		for (ArcIterator<VectorFst<StdArc> > aiter1(fst1, s);
				!aiter1.Done(); aiter1.Next())
		{
			if(aiter2.Done())
				return false;
			const StdArc &arc1 = aiter1.Value();
			const StdArc &arc2 = aiter2.Value();
			if(!EqualSrc(arc1, arc2))
				return false;
			aiter2.Next();
		}
		if(!aiter2.Done())
			return false;
		if(fst1.Final(s) != fst2.Final(s))
			return false;
	}
	return true;
}


}// namespace fst
