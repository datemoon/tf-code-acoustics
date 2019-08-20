
#ifndef __FST_CONVERT_OPENFST_H__
#define __FST_CONVERT_OPENFST_H__


#include <fst/fstlib.h>

namespace fst
{
typedef int int32;
typedef float BaseFloat;


template <class Arc>
bool ConvertSparseFstToOpenFst(const int32 *indexs, const int32 *in_labels, const int32 *out_labels,
		BaseFloat* weights, const int32* statesinfo,
		int32 num_states,
		VectorFst<Arc> *fst);


bool MallocSparseFst(int num_states, int num_arcs,
		int32 **indexs,
		int32 **in_labels,
		int32 **out_labels,
		BaseFloat **weights,
		int32 **stateinfo);

template <class Arc>
int ConvertKaldiLatticeToSparseLattice(VectorFst<Arc> &fst,
		int32 **indexs,
		int32 **in_labels,
		int32 **out_labels,
		BaseFloat **weights,
		int32 **stateinfo);

} // namespace fst

#endif
