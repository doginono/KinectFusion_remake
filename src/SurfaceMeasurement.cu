#include "SurfaceMeasurement.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>


__global__ void my_kernel(int A[], int B[], int C[]) {
	//(*C.elements)[] = A.elements[threadIdx] + B.elements[threadIdx];
	//int col = int(int(threadIdx.x )/ (3));
	int row = int(threadIdx.x);
	C[row]= A[row]* B[row];
}

void my_cuda_func(int A[], int B[], int C[]) {
	my_kernel<<<1, 9>>>(A, B, C);
	cudaDeviceSynchronize();
}
