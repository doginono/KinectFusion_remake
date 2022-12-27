#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceMeasurement.cuh"
#include <stdio.h>
#include <iostream>

//The GPU does the work
__global__ void vectorAdd(const int*  a, const int*  b, int*  c) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < 8){
		// Addddd
		c[tid]= a[tid]+ b[tid];
		//printf("Call for value : %d\n", c[tid]);
	}
}
namespace CUDA{
	//Wrapper to call the kernel function on the GPU
	void my_cuda_func(std::vector<int>& A, std::vector<int>& B, int C[]) {

		int* d_a, * d_b, * d_c;
		constexpr int N = 8;

		constexpr size_t bytes = sizeof(int) * N;

		//allocate memory on the GPU of the size you want to change in our case sizeof(int) * N;
		cudaMalloc(&d_a, bytes);
		cudaMalloc(&d_b, bytes);
		cudaMalloc(&d_c, bytes);

		//copy the data to the GPU
		cudaMemcpy(d_a, A.data(), bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, B.data(), bytes, cudaMemcpyHostToDevice);

		//8 threads 1 block
		vectorAdd<<< 1, 8 >>>(d_a, d_b, d_c);

		//After the calculation copy the value back to the CPU 
		cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

		//cudaDeviceSynchronize();
		//Free the allocated memory on the GPU
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
	
		std::cout << "COMPLETED SUCCESSFULLY\n";

	}
}