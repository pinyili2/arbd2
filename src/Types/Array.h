/*********************************************************************
 * @file  Array.h
 * 
 * @brief Declaration of templated Array class.
 *********************************************************************/
#pragma once
#include <memory>
#include <type_traits> // for std::common_type<T,U>
#include <sstream>

// Simple templated array object without resizing capabilities 
template<typename T>
class Array {
public:
    HOST inline Array<T>() : num(0), values(nullptr) {} // printf("Creating Array1 %x\n",this);
    HOST inline Array<T>(size_t num) : num(num), values(nullptr) {
	// printf("Constructing Array<%s> %x with values %x\n", type_name<T>().c_str(), this, values);
	host_allocate();
	// printf("Array<%s> %x with values %x\n", type_name<T>().c_str(), this, values);
    }
    HOST inline Array<T>(size_t num, const T* inp ) : num(num), values(nullptr) {
	// printf("Constructing Array<%s> %x with values %x\n", type_name<T>().c_str(), this, values);
	host_allocate();
	for (size_t i = 0; i < num; ++i) {
	    values[i] = inp[i];
	}
	// printf("Created Array3 %x with values %x\n",this, values);
    }
    HOST inline Array<T>(const Array<T>& a) { // copy constructor
	// printf("Copy-constructing Array<T> %x from %x with values %x\n",this, &a, a.values);
	num = a.num;
	host_allocate();
	for (size_t i = 0; i < num; ++i) {
	    values[i] = a[i];
	}
	// printf("Copy-constructed Array<T> %x with values %x\n",this, values);
    }
    HOST inline Array<T>(Array<T>&& a) { // move constructor
	// printf("Move-constructing Array<T> from %x with values %x\n", &a, a.values);
	num = a.num;
	values = a.values;
	a.values = nullptr;
	a.num = 0;		// not needed?
	// printf("Move-constructed Array<T> with values %x\n",  values);
    }
    HOST inline Array<T>& operator=(const Array<T>& a) { // copy assignment operator
	num = a.num;
	host_allocate();
	for (size_t i = 0; i < num; ++i) {
	    values[i] = a[i];
	}
	printf("Copy-operator for Array<T> %x with values %x\n",this, values);
	return *this;
    }
    HOST inline Array<T>& operator=(Array<T>&& a) { // move assignment operator
	host_deallocate();
	num = a.num;
	values = a.values;
	a.num = 0;
	a.values = nullptr;
	printf("Move-operator for Array<T> %x with values %x\n",this, values);
	return *this;
    }
    HOST DEVICE inline T& operator[](size_t i) {
	assert( i < num );
	return values[i];
    }
    HOST DEVICE inline const T& operator[](size_t i) const {
	assert( i < num );
	return values[i];
    }
    HOST inline ~Array<T>() {
	// printf("Destroying Array %x with values %x\n",this, values);
	host_deallocate();
    }
    
#ifdef USE_CUDA
    // This ugly template allows overloading copy_to_cuda, depending on whether T.copy_to_cuda exists using C++14-compatible SFINAE
    template <typename Dummy = void, typename std::enable_if_t<!has_copy_to_cuda<T>::value, Dummy>* = nullptr>
    HOST inline Array<T>* copy_to_cuda(Array<T>* dev_ptr = nullptr) const {
	if (dev_ptr == nullptr) { // allocate if needed
	    // printf("   cudaMalloc for array\n");
	    gpuErrchk(cudaMalloc(&dev_ptr, sizeof(Array<T>)));
	}

	// Allocate values_d
	T* values_d = nullptr;
	if (num > 0) {
	    // printf("   cudaMalloc for %d items\n", num);
	    size_t sz = sizeof(T) * num;
	    gpuErrchk(cudaMalloc(&values_d, sz));

	    // Copy values
	    gpuErrchk(cudaMemcpy(values_d, values, sz, cudaMemcpyHostToDevice));
	}
	    
	// Copy Array with pointers correctly assigned
	Array<T> tmp(0);
	tmp.num = num;
	tmp.values = values_d;
	gpuErrchk(cudaMemcpy(dev_ptr, &tmp, sizeof(Array<T>), cudaMemcpyHostToDevice));
	tmp.num = 0;
	tmp.values = nullptr;
	// printf("Copying Array<%s> %x with %d values %x to device at %x\n", type_name<T>().c_str(), this, num, values, dev_ptr);
	return dev_ptr;
    }

    template <typename Dummy = void, typename std::enable_if_t<has_copy_to_cuda<T>::value, Dummy>* = nullptr>
    HOST inline Array<T>* copy_to_cuda(Array<T>* dev_ptr = nullptr) const {
	// enable_if<!has_copy_to_cuda<T>::value, T>::type* = 0) const {
	if (dev_ptr == nullptr) { // allocate if needed
	    // printf("   cudaMalloc for array\n");
	    gpuErrchk(cudaMalloc(&dev_ptr, sizeof(Array<T>)));
	}

	// Allocate values_d
	T* values_d = nullptr;
	if (num > 0) { 
	    size_t sz = sizeof(T) * num;
	    // printf("   cudaMalloc for %d items\n", num);
	    gpuErrchk(cudaMalloc(&values_d, sz));

	    // Copy values
	    for (size_t i = 0; i < num; ++i) {
		values[i].copy_to_cuda(values_d + i);
	    }
	}

	// Copy Array with pointers correctly assigned
	Array<T> tmp(0);
	tmp.num = num;
	tmp.values = values_d;
	gpuErrchk(cudaMemcpy(dev_ptr, &tmp, sizeof(Array<T>), cudaMemcpyHostToDevice));
	tmp.num = 0;
	tmp.values = nullptr;
	// printf("Copying Array %x with values %x to device at %x\n",this, values, dev_ptr);
	return dev_ptr;
    }

    template <typename Dummy = void, typename std::enable_if_t<!has_copy_to_cuda<T>::value, Dummy>* = nullptr>
    HOST static Array<T> copy_from_cuda(Array<T>* dev_ptr) {
	// Create host object, copy raw device data over
	Array<T> tmp(0);
	if (dev_ptr != nullptr) {
	    gpuErrchk(cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));

	    if (tmp.num > 0) {
		T* values_d = tmp.values;
		tmp.values = new T[tmp.num];
	    	    
		// Copy values
		size_t sz = sizeof(T) * tmp.num;
		gpuErrchk(cudaMemcpy(tmp.values, values_d, sz, cudaMemcpyDeviceToHost));
	    } else {
		tmp.values = nullptr;
	    }
	}
	// printf("Copying device Array %x to host %x with values %x\n", dev_ptr, &tmp, tmp.values);
	return tmp;
    }

    template <typename Dummy = void, typename std::enable_if_t<has_copy_to_cuda<T>::value, Dummy>* = nullptr>
    HOST static Array<T> copy_from_cuda(Array<T>* dev_ptr) {
	// Create host object, copy raw device data over
	Array<T> tmp(0);

	if (dev_ptr != nullptr) {
	    gpuErrchk(cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));

	    if (tmp.num > 0) {
		T* values_d = tmp.values;
		tmp.values = new T[tmp.num];
	    	    
		// Copy values
		for (size_t i = 0; i < tmp.num; ++i) {
		    tmp.values[i] = T::copy_from_cuda(values_d + i);
		}
	    } else {
		tmp.values = nullptr;
	    }
	}
	// printf("Copying device Array %x to host %x with values %x\n", dev_ptr, &tmp, tmp.values);
	return tmp;
    }

    template <typename Dummy = void, typename std::enable_if_t<!has_copy_to_cuda<T>::value, Dummy>* = nullptr>
    HOST static void remove_from_cuda(Array<T>* dev_ptr, bool remove_self = true) {
	// printf("Removing device Array<%s> %x\n", typeid(T).name(), dev_ptr);
	if (dev_ptr == nullptr) return;
	Array<T> tmp(0);
	gpuErrchk(cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));
	if (tmp.num > 0) {
	    // Remove values
	    gpuErrchk(cudaFree(tmp.values));
	}
	tmp.values = nullptr;
	gpuErrchk(cudaMemset((void*) &(dev_ptr->values), 0, sizeof(T*))); // set nullptr on to device
	if (remove_self) {
	    gpuErrchk(cudaFree(dev_ptr));
	    dev_ptr = nullptr;
	}
	// printf("...done removing device Array<%s> %x\n", typeid(T).name(), dev_ptr);
    }

    template <typename Dummy = void, typename std::enable_if_t<has_copy_to_cuda<T>::value, Dummy>* = nullptr>
    HOST static void remove_from_cuda(Array<T>* dev_ptr, bool remove_self = true) {
	// printf("Removing device Array<%s> %x\n", typeid(T).name(), dev_ptr);
	if (dev_ptr == nullptr) return;
	Array<T> tmp(0);
	gpuErrchk(cudaMemcpy(&tmp, dev_ptr, sizeof(Array<T>), cudaMemcpyDeviceToHost));
	if (tmp.num > 0) {
	    // Remove values
	    for (size_t i = 0; i < tmp.num; ++i) {
		T::remove_from_cuda(tmp.values+i, false);
	    }
	}
	tmp.values = nullptr;
	gpuErrchk(cudaMemset((void*) &(dev_ptr->values), 0, sizeof(T*))); // set nullptr on device
	if (remove_self) {
	    gpuErrchk(cudaFree(dev_ptr));
	    dev_ptr = nullptr;
	}
	// printf("...done removing device Array<%s> %x\n", typeid(T).name(), dev_ptr);
    }
#endif
    HOST DEVICE size_t size() const { return num; }
    
private:
    HOST void host_allocate() {
	host_deallocate();
	if (num > 0) {
	    values = new T[num];
	} else {
	    values = nullptr;
	}
	// printf("Array<%s>.host_allocate() %d values at %x\n", typeid(T).name(), num, values);

    }
    HOST void host_deallocate() {
	// printf("Array<%s>.host_deallocate() %d values at %x\n", typeid(T).name(), num, values);
	if (values != nullptr) delete[] values;
	values = nullptr;
    }
    
    size_t num;
    T* values;
};
