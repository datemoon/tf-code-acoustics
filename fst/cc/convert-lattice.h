#ifndef __CONVERT_LATTICE_H__
#define __CONVERT_LATTICE_H__

// kaldi io
#include <stdlib.h>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "lat/kaldi-lattice.h"

namespace kaldi
{
bool MallocSparseLattice(int num_states, int num_arcs,
		BaseFloat **indexs,
		int32 **pdf_values,
		BaseFloat **lm_ws,
		BaseFloat **am_ws,
		int32 **stateinfo)
{
	*indexs = (BaseFloat *)malloc(num_arcs * 2 * sizeof(BaseFloat));
	*pdf_values = (int32 *)malloc(num_arcs * sizeof(int32));

	*lm_ws = (BaseFloat *)malloc(num_arcs  * sizeof(BaseFloat));
	*am_ws = (BaseFloat *)malloc(num_arcs  * sizeof(BaseFloat));
	*stateinfo = (int32 *)malloc(num_states * 2 * sizeof(int32));

	return true;
}
/*
 *
 * return : number states
 **/
int ConvertKaldiLatticeToSparseLattice(Lattice &inlat, 
		BaseFloat **indexs,
		int32 **pdf_values,
		BaseFloat **lm_ws,
		BaseFloat **am_ws,
		int32 **stateinfo)
{
	using namespace kaldi;

	typedef Lattice::Arc Arc;

	// inlat must be topsort and it's super final lattice(have only final state and final-probs are One()).
	fst::CreateSuperFinal(&inlat);
	int32 num_states = inlat.NumStates();
	int32 num_arcs = 0;
	for(int s=0; s<num_states; s++)
	{
		num_arcs += inlat.NumArcs(s);
	}
	// malloc space for sparse lattice
	if(*indexs == NULL)
		MallocSparseLattice(num_states, num_arcs, indexs, pdf_values, lm_ws, am_ws, stateinfo);

	int state_offset = 0;
	for(int s=0; s<num_states; s++)
	{
		int state_arcs = 0;
		for (fst::ArcIterator<Lattice> aiter(inlat, s); !aiter.Done(); aiter.Next())
		{
			int cur_offset = state_offset + state_arcs;
			const Arc &arc = aiter.Value();
			float lm_w = arc.weight.Value1();
			float am_w = arc.weight.Value2();
			int pdf = arc.ilabel;
			(*indexs)[cur_offset*2+0] = s;
			(*indexs)[cur_offset*2+1] = arc.nextstate;
			(*pdf_values)[cur_offset] = pdf;
			(*lm_ws)[cur_offset] = lm_w;
			(*am_ws)[cur_offset] = am_w;
			state_arcs++;
		}
		(*stateinfo)[2*s+0] = state_offset;
		(*stateinfo)[2*s+1] = state_arcs;
		state_offset += state_arcs;
	}
	return num_states;
}

} // end namespace
#endif
