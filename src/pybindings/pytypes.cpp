#include "../Types.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

namespace py = pybind11;

/// Convert a NumPy array to a Vector3_t object.
///
/// This function converts a 1D NumPy array of size 3 or 4 into a Vector3_t object.
///
/// \tparam T - The data type of the elements in the NumPy array.
/// \param a - The NumPy array to convert.
/// \return A Vector3_t object created from the array.
template<typename T>
Vector3_t<T> array_to_vector(py::array_t<T> a) {
    py::buffer_info info = a.request();
    if (info.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    if (info.size < 3 || info.size > 4)
        throw std::runtime_error("Size of array must be 3 or 4");
   
    T *ptr = static_cast<T *>(info.ptr);
    if (info.size == 3)
	return Vector3_t<T>(ptr[0],ptr[1],ptr[2]);
    else
	return Vector3_t<T>(ptr[0],ptr[1],ptr[2],ptr[3]);
}

/// Convert a NumPy array to a Vector3_t object.
///
/// This function converts a 1D NumPy array of size 3 or 4 into a Vector3_t object.
///
/// \tparam T - The data type of the elements in the NumPy array.
/// \param a - The NumPy array to convert.
/// \return A Array<Vector3_t<T>> object created from the array.
template<typename T>
Array<Vector3_t<T>> array_to_vector_arr(py::array_t<T> a) {
    py::buffer_info info = a.request();
    if (info.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");
    if (info.shape[1] < 3 || info.shape[1] > 4)
        throw std::runtime_error("Second dimension of numpy array must contain 3 or 4 elements");
   
    T *ptr = static_cast<T *>(info.ptr);

    Array<Vector3_t<T>> arr(static_cast<size_t>(info.shape[0]));
    assert( info.strides[0] == info.shape[1] );
    if (info.shape[1] == 3) {
	for (size_t i = 0; i < info.shape[0]; ++i) {
	    size_t j = i*info.strides[0];
	    arr[i] = (Vector3_t<T>( ptr[j], ptr[j+1], ptr[j+2] ));
	}
    } else {
	for (size_t i = 0; i < info.shape[0]; ++i) {
	    size_t j = i*info.shape[1];
	    arr[i] = (Vector3_t<T>( ptr[j], ptr[j+1], ptr[j+2], ptr[j+3] ));
	}
    }
    return arr;
}


/// Declare Python bindings for Vector3_t<some_type>
///
/// This function defines Python bindings for Vector3_t class with the specified data type.
///
/// \tparam T - The data type of the Vector3_t elements, typically float.
/// \param m - The Python module in which to declare the bindings.
/// \param typestr - The string representation of the data type for naming the Python class.
template<typename T>
void declare_vector(py::module &m, const std::string &typestr) {
    using Class = Vector3_t<T>;
    std::string pyclass_name = std::string("Vector3_t_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
	// Constructors
	.def(py::init<>())
	.def(py::init<T>())
	.def(py::init<T, T, T>())
	.def(py::init([](py::array_t<T> a) { return array_to_vector<T>(a); }))
	// .def(py::init<T, T, T, T>(),[] lambda )
	// .def(py::init<T*>())

	// Operators
	.def(py::self + py::self)
	.def(py::self * py::self)
        // .def(py::self += py::self)
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self * float())
        .def(-py::self)
	
	// Conversions
        .def("__repr__", &Class::to_string);
}

/// Declare Python bindings for Vector3_t<some_type>
///
/// This function defines Python bindings for Vector3_t class with the specified data type.
///
/// \tparam T - The data type of the Vector3_t elements, typically float.
/// \param m - The Python module in which to declare the bindings.
/// \param typestr - The string representation of the data type for naming the Python class.
template<typename T>
void declare_vector_array(py::module &m, const std::string &typestr) {
    using Class = Array<Vector3_t<T>>;
    std::string pyclass_name = std::string("Vector3_Array_t_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
	// Constructors
	.def(py::init<>())
	.def(py::init([](py::array_t<T> a) { return array_to_vector_arr<T>(a); }))

	// // Operators
	// .def(py::self + py::self)
	// .def(py::self * py::self)
        // // .def(py::self += py::self)
        // .def(py::self *= float())
        // .def(float() * py::self)
        // .def(py::self * float())
        // .def(-py::self)
	
	// Conversions
        // .def("__repr__", &Class::to_string)
	;
}


/// Initialize Python bindings for basic types.
///
/// \param m - The Python module in which to declare the bindings.
void init_pytypes(py::module_ &m) {
    declare_vector<int>(m, "int");
    declare_vector<float>(m, "float");
    declare_vector<double>(m, "double");
    m.attr("Vector3") = m.attr("Vector3_t_float"); // alias Vector3 to Vector3_t_float 
    
    declare_vector_array<int>(m, "int");
    declare_vector_array<float>(m, "float");
    declare_vector_array<double>(m, "double");

    m.attr("VectorArr") = m.attr("Vector3_Array_t_float"); // alias Vector3 to Vector3_t_float 
}
