
#include <stdlib.h>

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
