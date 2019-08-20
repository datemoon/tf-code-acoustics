
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
bool ConvertSparseFstToOpenFst(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		BaseFloat* weights, const int32* statesinfo,
		int32 num_states,
		VectorFst<Arc> *fst)
//		StdVectorFst *fst)
{
	typedef typename Arc::StateId StateId;
	typedef typename Arc::Weight Weight;
	typedef typename Arc::Label Label;

	fst->DeleteStates();
	StateId start_state = fst->AddState();
	fst->SetStart(start_state);

	for(int s=0; s<num_states; s++)
	{
		int s_start = statesinfo[2*s+0];
		int narcs = statesinfo[2*s+1];
		for(int a = 0; a < narcs ; a++)
		{
			int cur_offset = a+s_start;
			int instate = indexs[cur_offset*2 + 0];
			int tostate = indexs[cur_offset*2 + 1];
			while(instate > fst->NumStates())
			{
				fst->AddState();
				//printf("it shouldn't happen,instate > s");
			}
			float weight = weights[cur_offset];
			int in_label = in_labels[cur_offset];
			int out_label = out_labels[cur_offset];
			Weight w = (Weight)(weight);
			fst->AddArc(instate, Arc(in_label, out_label, w, tostate));
		}
		if(s == num_states-1)
			fst->SetFinal(s, Weight::One());;
	}

	return true;
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

/*
 *
 * return : number states
 **/
template <class Arc>
int ConvertKaldiLatticeToSparseLattice(VectorFst<Arc> &fst,
		int32 **indexs,
		int32 **in_labels,
		int32 **out_labels,
		BaseFloat **weights,
		int32 **stateinfo)
{
	CreateSuperFinal(&fst);
	TopSort(&fst);
	// inlat must be topsort and it's super final lattice(have only final state and final-probs are One()).
	int32 num_states = fst.NumStates();
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
	return num_states;

}



}// namespace fst
