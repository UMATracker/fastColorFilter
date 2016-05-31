#ifdef _OEPNMP
#include <omp.h>
#endif

#define BOOST_PYTHON_STATIC_LIB
#include <Python.h>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <boost/shared_array.hpp>

#include <algorithm>
#include <iostream>

#include <cmath>
#include <cstdint>

namespace py = boost::python;
namespace np = boost::numpy;

void exec(
	boost::shared_array<boost::shared_array<uint8_t*>> array,
	const uint8_t* rgb,
	const int dist,
	const int rows,
	const int cols
	) {
	auto r0 = rgb[0];
	auto g0 = rgb[1];
	auto b0 = rgb[2];

	#pragma omp parallel for
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			auto r1 = array[i][j][0];
			auto g1 = array[i][j][1];
			auto b1 = array[i][j][2];

			int r = std::max(r0, r1) - std::min(r0, r1);
			int g = std::max(g0, g1) - std::min(g0, g1);
			int b = std::max(b0, b1) - std::min(b0, b1);

			auto d = sqrt(r*r + g*g + b*b);
			if (dist < d) {
				array[i][j][0] = 0;
				array[i][j][1] = 0;
				array[i][j][2] = 0;
			}
		}
	}
}

void exec_wrap(
	np::ndarray array,
	np::ndarray rgb,
	int dist
	) {
	if (array.get_dtype() != np::dtype::get_builtin<uint8_t>()) {
		PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
		py::throw_error_already_set();
	}
	if (array.get_nd() != 3) {
		PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
		py::throw_error_already_set();
	}
	if (array.shape(2) != 3) {
		PyErr_SetString(PyExc_TypeError, "Incorrect shape. np.ndarray.shape!=(row, col, 3).");
		py::throw_error_already_set();
	}
	if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
		PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous");
		py::throw_error_already_set();
	}

	auto iter = reinterpret_cast<uint8_t*>(array.get_data());
	int rows = array.shape(0);
	int cols = array.shape(1);

	boost::shared_array<boost::shared_array<uint8_t*>> ptrs_row(new boost::shared_array<uint8_t*>[rows]);
	for (int i = 0; i < rows; ++i, iter += cols*3) {
		boost::shared_array<uint8_t*> ptrs_col(new uint8_t*[cols]);
		ptrs_row[i] = ptrs_col;

		auto iter_tmp = iter;
		for (int j = 0; j < cols; ++j, iter_tmp += 3) {
			ptrs_col[j] = iter_tmp;
		}
	}

	exec(
		ptrs_row,
		reinterpret_cast<uint8_t*>(rgb.get_data()),
		dist,
		rows,
		cols
		);
}

BOOST_PYTHON_MODULE(fastColorFilter) {
	Py_Initialize();
	np::initialize();
	py::def("exec", exec_wrap);
}