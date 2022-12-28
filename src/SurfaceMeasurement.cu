#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceMeasurement.cuh"
#include <stdio.h>
#include <iostream>
#include <time.h>


//The GPU does the work
__global__ void vectorAdd(const int*  a, const int*  b, int*  c) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < 8){
		c[tid]= a[tid]+ b[tid];
	}
}
__global__ void initSensorFrame_kernel(const float* depthMap, const Matrix3f rotationInv, const Vector3f translationInv,
	 float* camparams, Vector3f* pointsTmp) {
	//this should be done on gpu meaning 640 as input
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;	
	if (tid < 640*480) { //640*480 being the height and width
		if (depthMap[tid] == MINF) {
			pointsTmp[tid] = Vector3f(MINF, MINF, MINF);
		}
		else {
			int u = (blockIdx.x * blockDim.x + threadIdx.x)%640;
			int v = int((blockIdx.x * blockDim.x + threadIdx.x) / 640);
			pointsTmp[tid] = rotationInv * Vector3f((u - camparams[2]) / camparams[0] * depthMap[tid], (v - camparams[3]) / camparams[1] * depthMap[tid], depthMap[tid]) + translationInv;
		}
	}
}
	
	
__global__ void example_kernel(float* depthmap,Vector3f* pointsTmp) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < 640 * 480)
	{
		
		float depth = depthmap[tid];
		pointsTmp[tid] = Vector3f(depth, depth, depth);
	}

}



namespace CUDA {
	//Wrapper to call the kernel function on the GPU

	void initSensorFrame(float depthMap[], Matrix3f& rotationInv, Vector3f& translationInv,
		std::vector<float>& camparams, std::vector<Vector3f>& pointsTmp) {

		//size_t bytes = sizeof(float) * 4+ sizeof(Vector3f) * (sizeof(pointsTmp) + 1)+ sizeof(Matrix3f);//+1 because translation

		//allocate memory on the GPU of the size you want to change in our case sizeof(int) * N;
		float* depthPointer;

		float* camparamPointer;
		Vector3f* pointsPointer;
		cudaMalloc(&depthPointer, sizeof(float) * 640 * 480);
		//4 variables in camparams
		//printf("CUDALI.. %d clicks (%f seconds).\n", t, ((float)t) / CLOCKS_PER_SEC);

		cudaMalloc(&camparamPointer, sizeof(float) * 4);
		cudaMalloc((void**)&pointsPointer, sizeof(Vector3f) * 640 * 480);
		
		//copy the data to the GPU
		cudaMemcpy(depthPointer, depthMap, sizeof(float) * 640 * 480, cudaMemcpyHostToDevice);
		cudaMemcpy(camparamPointer, camparams.data(), sizeof(float) * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(pointsPointer, pointsTmp.data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
		//8 threads 1 block
		initSensorFrame_kernel <<<4800, 64 >> > (depthPointer, rotationInv, translationInv, camparamPointer, pointsPointer);
		//After the calculation copy the value back to the CPU 
		cudaMemcpy(pointsTmp.data(), pointsPointer, sizeof(Vector3f) * 640 * 480, cudaMemcpyDeviceToHost);
		
		//cudaDeviceSynchronize();
		//Free the allocated memory on the GPU
		cudaFree(depthPointer);
		cudaFree(camparamPointer);
		cudaFree(pointsPointer);
		
	}

	//just example tried here to find the bug XD
	void example(float depthMap[], std::vector<Vector3f>& pointsTmp) {
		float* depthPointer;
		Vector3f* pointsPointer;

		cudaMalloc(&depthPointer, sizeof(float) * 640 * 480);
		cudaMalloc((void**)&pointsPointer, sizeof(Vector3f) * 640 * 480);

		cudaMemcpy(depthPointer, depthMap, sizeof(float) * 640 * 480, cudaMemcpyHostToDevice);
		cudaMemcpy(pointsPointer, pointsTmp.data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
		
		example_kernel <<< 4800, 64 >>>(depthPointer,pointsPointer);
		cudaMemcpy(pointsTmp.data(), pointsPointer, sizeof(Vector3f) * 640 * 480, cudaMemcpyDeviceToHost);

		cudaFree(pointsPointer);
		cudaFree(depthPointer);

	}


}