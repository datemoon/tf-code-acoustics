#ifndef __CONVERT_LATTICE_H__
#define __CONVERT_LATTICE_H__

// kaldi io
#include <stdlib.h>
#include <cstring>

#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "lat/kaldi-lattice.h"
#include "hmm/transition-model.h"

namespace kaldi
{
bool MallocSparseLattice(int num_states, int num_arcs,
		int32 **indexs,
		int32 **pdf_values,
		BaseFloat **lm_ws,
		BaseFloat **am_ws,
		int32 **stateinfo)
{
	*indexs = new int32[num_arcs * 2];
	//*indexs = (int32 *)malloc(num_arcs * 2 * sizeof(int32));
	memset(*indexs, 0x00, num_arcs * 2 * sizeof(int32));
	*pdf_values = new int32[num_arcs];
	//*pdf_values = (int32 *)malloc(num_arcs * sizeof(int32));
	memset(*pdf_values, 0x00, num_arcs * sizeof(int32));
	
	*lm_ws = new BaseFloat[num_arcs];
	//*lm_ws = (BaseFloat *)malloc(num_arcs  * sizeof(BaseFloat));
	memset(*lm_ws, 0x00, num_arcs * sizeof(BaseFloat));
	*am_ws = new BaseFloat[num_arcs];
	//*am_ws = (BaseFloat *)malloc(num_arcs  * sizeof(BaseFloat));
	memset(*am_ws, 0x00, num_arcs * sizeof(BaseFloat));
	*stateinfo = new int32[num_states * 2];
	//*stateinfo = (int32 *)malloc(num_states * 2 * sizeof(int32));
	memset(*stateinfo , 0x00, num_states * 2 * sizeof(int32));

	return true;
}
/*
 *
 * return : number states
 **/
int ConvertKaldiLatticeToSparseLattice(Lattice &inlat, 
		int32 **indexs,
		int32 **pdf_values,
		BaseFloat **lm_ws,
		BaseFloat **am_ws,
		int32 **stateinfo)
{
	using namespace kaldi;

	typedef Lattice::Arc Arc;

	fst::CreateSuperFinal(&inlat);
	TopSort(&inlat);
	// inlat must be topsort and it's super final lattice(have only final state and final-probs are One()).
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

void AliToPdfOffset(int *ali, int n, TransitionModel &trans_model, int offset=0)
{
	for(int i=0; i<n; ++i)
	{
		if(ali[i] == 0)
			continue;
		ali[i] = trans_model.TransitionIdToPdf(ali[i]) + offset;
	}
}

template<typename T>
T* BatchIn(T* input, int size, int batch)
{
	T *tmp = new T[size * batch];
	for(int i=0;i<batch;i++)
	{
		memcpy((void*)(tmp+size*i),(void*)input, sizeof(T) * size);
	}
	delete []input;
	return tmp;
}
template<typename T>
T *BatchInT(T* input, int rows, int cols, int batch)
{
	T *tmp = new T[rows * cols * batch];
	for(int r=0;r<rows;r++)
	{
		for(int i=0;i<batch;i++)
		{
			memcpy((void*)(tmp + r * batch * cols + cols *i),(void*)(input + cols * r), sizeof(T) * cols);
		}
	}
	delete[]input;
	return tmp;
}

void BatchAllData(int32 **indexs,
		int32 **pdf_values,
		BaseFloat **lm_ws,
		BaseFloat **am_ws,
		int32 **statesinfo,
		BaseFloat **batch_loss,
		int32 **batch_num_states,
		int32 **batch_pdf_ali,
		int32 **batch_sequence_length,
		float **h_nnet_out_h,
		float **gradient,
		int32 rows,
		int32 cols,
		int32 batch_size,
		int32 max_num_arcs,
		int32 max_num_states)
{
	if(batch_size <= 1)
		return;
	*indexs = BatchIn(*indexs, max_num_arcs * 2, batch_size);
	*pdf_values = BatchIn(*pdf_values, max_num_arcs, batch_size);
	*lm_ws = BatchIn(*lm_ws, max_num_arcs, batch_size);
	*am_ws = BatchIn(*am_ws, max_num_arcs, batch_size);
	*statesinfo = BatchIn(*statesinfo, max_num_states * 2 , batch_size);
	
	*batch_num_states = BatchIn(*batch_num_states, 1, batch_size);
	
	*h_nnet_out_h = BatchInT(*h_nnet_out_h, rows , cols, batch_size);
	
	*batch_pdf_ali = BatchIn(*batch_pdf_ali, cols, batch_size);
	*batch_sequence_length = BatchIn(*batch_sequence_length, 1, batch_size);
	
	*gradient = BatchIn(*gradient, rows * cols, batch_size);
	*batch_loss = BatchIn(*batch_loss, 1, batch_size);
}

void CutLineToStr(std::string line,std::vector<std::string> &str)//, char ch)
{
	str.clear();
	size_t i=0,j=0;
	size_t len = line.length();
	char *s = (char*)line.c_str();
	int k = 0;
	for(i=0,j=0;i<len;++i)
	{
		switch(s[i])
		{
			case ' ':
			case '\t':
			case '\n':
			case '\r':
			case '\f':
			case '\v':
				s[i] = '\0';
				if(j==0)
					break;
                str.push_back(s+i-j);
                ++k;
                j=0;
                break;
            default:
                ++j;
                break;
        }
    }
	// add the last word
	if(0 != j)
	{
		str.push_back(s+i-j);
	}
}


int32 *ReadMap(std::string &ali_map_filename, int *col, int row=3)
{
	FILE *fp = fopen(ali_map_filename.c_str(),"r");
	if(fp == NULL)
	{
		return NULL;
	}
	char line[1024*1024];
	memset(line,0x00,sizeof(line));
	int32 *map_list = NULL;
	int offset = 0;
	while(fgets(line,sizeof(line),fp) != NULL)
	{
		std::vector<std::string> str;
		CutLineToStr(line, str);
		if(map_list == NULL)
		{
			*col = static_cast<int32>(str.size());
			map_list = new int32[*col * row];
		}

		for(int i=0; i < *col; ++i)
		{
			map_list[i*row + offset] = atoi(str[i].c_str());
		}
		offset++;
	}
	fclose(fp);
	return map_list;
}

int32 *MapConvert(const int32 *ali_map,int row, int col=3)
{
	int const_col = 2;
	int32 max_pdf = 0;
	for(int r=0;r<row;r++)
	{
		if(ali_map[r*col+1] > max_pdf)
			max_pdf = ali_map[r*col+1];
	}
	int32 *pdf_to_phone = new int32[(max_pdf+1)*const_col];
	memset(pdf_to_phone, 0x00, (max_pdf+1)* const_col * sizeof(int32));
	for(int r=0;r<row;r++)
	{
		int32 pdf = ali_map[r*col+1];
		if(pdf < 0)
			continue;
		int32 phone = ali_map[r*col+2];
		if(pdf != 0 && pdf_to_phone[ pdf * const_col+0] != 0)
		{
			assert(pdf == pdf_to_phone[ pdf * const_col+0]);
			assert(pdf_to_phone[ pdf * const_col+1] == phone);
			continue;
		}
		else
		{
			pdf_to_phone[ pdf * const_col+0] = pdf;
			pdf_to_phone[ pdf * const_col+1] = phone;
		}
	}
	return pdf_to_phone;
} // end MapConvert

} // end namespace
#endif
