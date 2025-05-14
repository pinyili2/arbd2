#pragma once
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <initializer_list>
#include <string>
#include <memory>
#include <algorithm>


class BitmaskBase {
	public:
		BitmaskBase(const size_t len) : len(len) {}
		virtual ~BitmaskBase() = default;
		
		HOST DEVICE
		virtual void set_mask(size_t i, bool value) = 0;
		
		HOST DEVICE
		virtual bool get_mask(size_t i) const = 0;
		
		size_t get_len() const { return len; }
		
		virtual void print() const {
			for (size_t i = 0; i < len; ++i) {
				printf("%d", static_cast<int>(get_mask(i)));
			}
			printf("\n");
		}
		
	protected:
		size_t len;
	};

// Don't use base because virtual member functions require device malloc
class Bitmask {
    typedef size_t idx_t;
    typedef unsigned int data_t;
public:
    Bitmask(const idx_t len) : len(len) {
	idx_t tmp = get_array_size();
	// printf("len %lld\n",len);
	// printf("tmp %d\n",tmp);
	assert(tmp * data_stride >= len);
	mask = (tmp > 0) ? new data_t[tmp] : nullptr;
	for (int i = 0; i < tmp; ++i) mask[i] = data_t(0);
    }
    ~Bitmask() { if (mask != nullptr) delete[] mask; }

    HOST DEVICE
    idx_t get_len() const { return len; }

    HOST DEVICE
    void set_mask(idx_t i, bool value) {
	// return;
	assert(i < len);
	idx_t ci = i/data_stride;
	data_t change_bit = (data_t(1) << (i-ci*data_stride));
#ifdef __CUDA_ARCH__
	if (value) {
	    atomicOr(  &mask[ci], change_bit );
	} else {
	    atomicAnd( &mask[ci], ~change_bit );
	}
#else
	if (value) {
	    mask[ci] = mask[ci] | change_bit;
	} else {
	    mask[ci] = mask[ci] & (~change_bit);
	}
#endif
    }
    
    HOST DEVICE
    bool get_mask(const idx_t i) const {
	// return false;
	assert(i < len);
	const idx_t ci = i/data_stride;
	return mask[ci] & (data_t(1) << (i-ci*data_stride));
    }

    HOST DEVICE inline bool operator==(Bitmask& b) const {
	// Inefficient but straightforward approach; directly comparing underlying data would be fine, but then we need to deal with data strides
	if (len != b.len) return false;
	for (idx_t i = 0; i < len; ++i) {
	    if (get_mask(i) != b.get_mask(i)) return false;
	}
	return true;
    }

#ifdef USE_CUDA
    HOST
    Bitmask* copy_to_cuda(Bitmask* tmp_obj_d = nullptr) const {
	Bitmask obj_tmp(0);
	data_t* mask_d = nullptr;
	size_t sz = sizeof(data_t) * get_array_size();
	if (tmp_obj_d == nullptr) {
	    gpuErrchk(cudaMalloc(&tmp_obj_d, sizeof(Bitmask)));
	}
	if (sz > 0) {
	    gpuErrchk(cudaMalloc(&mask_d, sz));
	    gpuErrchk(cudaMemcpy(mask_d, mask, sz, cudaMemcpyHostToDevice));
	}
	// printf("Bitmask::copy_to_cuda() len(%lld) mask(%x)\n", len, mask_d);
	obj_tmp.len = len;
	obj_tmp.mask = mask_d;
	gpuErrchk(cudaMemcpy(tmp_obj_d, &obj_tmp, sizeof(Bitmask), cudaMemcpyHostToDevice));
	obj_tmp.mask = nullptr;
	return tmp_obj_d;
    }

    HOST
    static Bitmask copy_from_cuda(Bitmask* obj_d) {
	Bitmask obj_tmp(0);
	gpuErrchk(cudaMemcpy(&obj_tmp, obj_d, sizeof(Bitmask), cudaMemcpyDeviceToHost));
	printf("TEST: %d\n", obj_tmp.len);
	if (obj_tmp.len > 0) {
	    size_t array_size = obj_tmp.get_array_size();
	    size_t sz = sizeof(data_t) * array_size;
	    data_t *data_addr = obj_tmp.mask;
	    obj_tmp.mask = new data_t[array_size];
	    gpuErrchk(cudaMemcpy(obj_tmp.mask, data_addr, sz, cudaMemcpyDeviceToHost));
	} else {
	    obj_tmp.mask = nullptr;
	}
	return obj_tmp;
    }

    HOST
    static void remove_from_cuda(Bitmask* obj_d) {
	Bitmask obj_tmp(0);
	gpuErrchk(cudaMemcpy(&obj_tmp, obj_d, sizeof(Bitmask), cudaMemcpyDeviceToHost));
	if (obj_tmp.len > 0) {
	    gpuErrchk(cudaFree(obj_tmp.mask));
	}
	obj_tmp.mask = nullptr;
	gpuErrchk(cudaMemset((void*) &(obj_d->mask), 0, sizeof(data_t*))); // set nullptr on to device
	gpuErrchk(cudaFree(obj_d));
	obj_d = nullptr;
    }
#endif

    HOST
    auto to_string() const {
	std::string s;
	s.reserve(len);
	for (size_t i = 0; i < len; ++i) {
	    s += get_mask(i) ? '1' : '0';
	}
	return s;
    }
    
    HOST DEVICE
    void print() {
	for (int i = 0; i < len; ++i) {
	    printf("%d", (int) get_mask(i));
	}
	printf("\n");
    }

private:
    idx_t get_array_size() const { return (len == 0) ? 1 : (len-1)/data_stride + 1; }
    idx_t len;
    const static idx_t data_stride = CHAR_BIT * sizeof(data_t)/sizeof(char);
    data_t* __restrict__ mask;
};
