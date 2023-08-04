#include "Types.h"
#include "ParticlePatch.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>


namespace py = pybind11;

template<typename T>
Vector3_t<T> array_to_vector(py::array_t<T> a) {
    py::buffer_info buf1 = a.request();
    if (buf1.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    if (buf1.size < 3 || buf1.size > 4)
        throw std::runtime_error("Size of array must be 3 or 4");
   
    T *ptr = static_cast<T *>(buf1.ptr);
    if (buf1.size == 3)
	return Vector3_t<T>(ptr[0],ptr[1],ptr[2]);
    else
	return Vector3_t<T>(ptr[0],ptr[1],ptr[2],ptr[3]);
}

// basic types
template<typename T>
void declare_vector(py::module &m, const std::string &typestr) {
    using Class = Vector3_t<T>;
    std::string pyclass_name = std::string("Vector3_t_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
	.def(py::init<>())
	.def(py::init<T>())
	.def(py::init<T, T, T>())
	.def(py::init([](py::array_t<T> a) { return array_to_vector<T>(a); }))
	// .def(py::init<T, T, T, T>(),[] lambda )
	// .def(py::init<T*>())
	.def(py::self + py::self)
	.def(py::self * py::self)
        // .def(py::self += py::self)
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self * float())
        .def(-py::self)
        .def("__repr__", &Class::to_string);
}

PYBIND11_MODULE(pyarbd, m) {
    declare_vector<int>(m, "int");
    declare_vector<float>(m, "float");
    declare_vector<double>(m, "double");
    m.attr("Vector3") = m.attr("Vector3_t_float");

/*
    py::class_<Vector_t<T>>(m, "Vector3")
        .def(py::init<float, float, float>())
        .def(py::init<float, float, float>())

        py::class_<Patch>(m, "Patch")
        .def(py::init<float, float, float>())
	.def(py::self + py::self)
	.def(py::self * py::self)
        // .def(py::self += py::self)
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self * float())
        .def(-py::self)
        .def("__repr__", &Vector3::to_string);
*/
}

