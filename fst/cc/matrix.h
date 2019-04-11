#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cassert>
#include <iostream>
#include <cstring>

namespace hubo
{

template <typename Real>
class Matrix
{
public:
	Matrix(Real *data, int rows, int cols, int stride=0):
		_data(data), _num_rows(rows), _num_cols(cols)
	{
		if(stride == 0)
			_stride = cols;
		else
			_stride = stride;
	}

	Matrix():_data(NULL),_num_rows(0), _num_cols(0), _stride(0) { }

	void SetZero()
	{
		;
	}
	inline Real operator()(int r, int c) const
	{
		assert(r < _num_rows && c < _num_cols);
		return *(_data + r * _stride + c);
	}
	inline Real& operator()(int r, int c) 
	{
		assert(r < _num_rows && c < _num_cols);
		return *(_data + r * _stride + c);
	}

	inline void SetRowZero(int r)
	{
		memset(_data + r*_stride, 0x00, sizeof(Real) * _num_cols);
	}
	int NumRows() { return _num_rows; }
	int NumCols() { return _num_cols; }

	void Print()
	{
		std::cout << _num_rows << " " << _num_cols << std::endl;
		for(int r=0;r<_num_rows;++r)
		{
			for(int c=0;c<_num_cols;++c)
			{
				std::cout << _data[r*_stride+c] << " ";
			}
			std::cout << std::endl;
		}
	}

	int GetMaxVal(int r)
	{
		int max_id = 0;
		Real *max = _data + r * _stride + 0;
		for(int c=1;c<_num_cols;++c)
		{	
			if(*(_data+ r * _stride + c) > *max)
			{
				max = _data+ r * _stride + c;
				max_id = c;
			}
		}
		return max_id;
	}
private:
	Real *_data;
	int _num_rows;
	int _num_cols;
	int _stride;
};
} // namespace




#endif
