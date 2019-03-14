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
	Matrix(Real *data, int cols, int rows, int stride=0):
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
	int NumRows() { return _num_rows; }
	int NumCols() { return _num_cols; }
private:
	Real *_data;
	int _num_rows;
	int _num_cols;
	int _stride;
};
} // namespace




#endif
