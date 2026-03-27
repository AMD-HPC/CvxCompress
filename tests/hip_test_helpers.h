// Copyright (C) 2025 Advanced Micro Devices, Inc.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file or at https://opensource.org/licenses/MIT.

// HIP test helper functions - random init and error checking
// Copied from hipCVXCompressAI for CvxCompress vs HIP GPU comparison tests
// Header-only implementation

#ifndef HIP_TEST_HELPERS_H
#define HIP_TEST_HELPERS_H

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// HIPCHECK macro for error checking
#define HIPCHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// L2 Error kernel
template<typename T>
__global__ void l2ErrorKernel(T* x, T* y, T* result, size_t N) {
    extern __shared__ char sdata_raw[];
    T* sdata = reinterpret_cast<T*>(sdata_raw);
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;
    
    T thread_sum = static_cast<T>(0.0);
    size_t idx = static_cast<size_t>(bid * num_threads + tid);
    size_t stride = static_cast<size_t>(num_blocks * num_threads);
    
    while (idx < N) {
        T diff = x[idx] - y[idx];
        thread_sum += diff * diff;
        idx += stride;
    }
    
    sdata[tid] = thread_sum;
    __syncthreads();
    
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// L2 Error host wrapper
template<typename T>
T l2Error(T* d_x, T* d_y, size_t N, bool relative = false, bool sqrt_result = true) {
    const int threadsPerBlock = 512;
    
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    int numCUs = prop.multiProcessorCount;
    int blocksPerGrid = numCUs;
    
    T* d_result = nullptr;
    HIPCHECK(hipMalloc(&d_result, sizeof(T)));
    HIPCHECK(hipMemset(d_result, 0, sizeof(T)));
    
    size_t sharedMemSize = threadsPerBlock * sizeof(T);
    hipLaunchKernelGGL(l2ErrorKernel<T>,
                       dim3(blocksPerGrid),
                       dim3(threadsPerBlock),
                       sharedMemSize, 0,
                       d_x, d_y, d_result, N);
    
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipGetLastError());
    
    T h_result;
    HIPCHECK(hipMemcpy(&h_result, d_result, sizeof(T), hipMemcpyDeviceToHost));
    
    T error = sqrt_result ? std::sqrt(h_result) : h_result;
    
    HIPCHECK(hipFree(d_result));
    return error;
}

// Random initialization kernel
template<typename T>
__global__ void randomInitKernel(T* array, size_t size, unsigned long long seed) {
    size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx < size) {
        rocrand_state_xorwow state;
        rocrand_init(seed, static_cast<unsigned long long>(idx), 0, &state);
        
        // Generate random value in range [0.0, 1.0)
        T random_value = static_cast<T>(rocrand_uniform(&state));
        array[idx] = random_value;
    }
}

// Random initialization host wrapper
template<typename T>
void randomInit(T* d_array, size_t size, unsigned long long seed) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(randomInitKernel<T>, 
                       dim3(blocksPerGrid), 
                       dim3(threadsPerBlock), 
                       0, 0, 
                       d_array, size, seed);
    
    HIPCHECK(hipGetLastError());
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipGetLastError());
}

// Explicit template instantiations
template __global__ void l2ErrorKernel<float>(float* x, float* y, float* result, size_t N);
template __global__ void l2ErrorKernel<double>(double* x, double* y, double* result, size_t N);
template float l2Error<float>(float*, float*, size_t, bool, bool);
template double l2Error<double>(double*, double*, size_t, bool, bool);

template __global__ void randomInitKernel<float>(float* array, size_t size, unsigned long long seed);
template __global__ void randomInitKernel<double>(double* array, size_t size, unsigned long long seed);
template void randomInit<float>(float* d_array, size_t size, unsigned long long seed);
template void randomInit<double>(double* d_array, size_t size, unsigned long long seed);

#endif // HIP_TEST_HELPERS_H
