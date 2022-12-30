#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceMeasurement.cuh"
#include <stdio.h>
#include <iostream>
#include <time.h>
#include "PoseEstimation.cuh"

//The GPU does the work
__global__ void poseEstimation_kernel( float* camparams, Vector3f* verticesSource, Vector3f* verticesPrevious,
	Matrix4f currentCameraToWorld, Matrix4f worldToCamera, int* correspondencesArray)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < 640 * 480) { //640*480 being the height and width
		if (verticesSource[tid] != Vector3f(MINF, MINF, MINF)|| verticesPrevious[tid] != Vector3f(MINF, MINF, MINF) || 
			verticesPrevious[tid][2]!= MINF) {
			//we are camera coordinates at verticesSource and verticesPrevious 
			Matrix3f rotInv = worldToCamera.block<3, 3>(0, 0);
			Vector3f transInv = worldToCamera.block<3, 1>(0, 2);

			Vector3f cameraPrevglob		= rotInv * verticesPrevious[tid] + transInv;
			Vector3f cameraCurrentglob  = rotInv * verticesSource[tid] + transInv;

			Vector2i pixelCoord(int(cameraPrevglob[0] * camparams[0] / cameraPrevglob[2] + camparams[2]),
								int(cameraPrevglob[1] * camparams[1] / cameraPrevglob[2] + camparams[3]));


			if ((verticesSource[tid] - verticesPrevious[tid]).norm() < 0.1	&& pixelCoord[0] > 0 && pixelCoord[0] <640 && pixelCoord[1] > 0 && pixelCoord[1] < 480) {
				//rotInv* verticesPrevious[]
				//printf("%i %i  \n", pixelCoord[0], pixelCoord[1]);

				correspondencesArray[pixelCoord[0] * pixelCoord[1]] = int(pixelCoord[0] * pixelCoord[1]);
				//printf("%i \n", correspondencesArray[pixelCoord[0] * pixelCoord[1]]);
			}
			else {
				correspondencesArray[tid] = 0;
			}
			//if ((verticesSource[pixelCoord[0] * pixelCoord[1]] - verticesPrevious[pixelCoord[0] * pixelCoord[1]]).norm() < 0.1) {
			//	printf("%i %i \n", pixelCoord[0] , pixelCoord[1]);
			//}
		}
	}
}
	
	



namespace CUDA {
	//Wrapper to call the kernel function on the GPU
	//only going to update the variable currentCameratoWorld to find the transformation

	void poseEstimation(PointCloud& source, PointCloud& previous, std::vector<float>& camparams, 
		Matrix4f& currentCameraToWorld, Matrix4f& worldToCamera, std::vector<int>& correspondencesArray){
		// currentcamerapose.inverse()*target
		
		Vector3f* verticesSource;
		Vector3f* verticesPrevious;

		float* camparamPointer; //params of the source
		int* correspondencesPointer;

		//Mallocs
		//4 variables in camparams Look at exercise 5 for multiplication
		cudaMalloc(&camparamPointer, sizeof(float) * 4);
		cudaMalloc((void**)&verticesSource, sizeof(Vector3f) * 640 * 480);
		cudaMalloc((void**)&verticesPrevious, sizeof(Vector3f) * 640 * 480);
		cudaMalloc(&correspondencesPointer, sizeof(int) * 640 * 480);
		

		cudaMemcpy(verticesSource, source.getPoints().data(), sizeof(Vector3f) * 640 * 480 , cudaMemcpyHostToDevice);
		cudaMemcpy(verticesPrevious, previous.getPoints().data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
		cudaMemcpy(correspondencesPointer, correspondencesArray.data(), sizeof(int) * 640 * 480, cudaMemcpyHostToDevice);
		cudaMemcpy(camparamPointer, camparams.data(), sizeof(float) * 4, cudaMemcpyHostToDevice);

		poseEstimation_kernel <<<4800, 64 >>> (camparamPointer,	verticesSource, verticesPrevious, 
											currentCameraToWorld, worldToCamera, correspondencesPointer);
		cudaDeviceSynchronize();

		cudaMemcpy(correspondencesArray.data(), correspondencesPointer, sizeof(int) * 640 * 480, cudaMemcpyDeviceToHost);
		//printf("%i \n", correspondencesArray[236 * 348]);
		//printf("%i \n", correspondencesArray.at(0));
		//At last free them
		cudaFree(verticesSource);
		cudaFree(verticesPrevious);
		cudaFree(camparamPointer);
		cudaFree(correspondencesPointer);
		//delete[] correspondencesArray;
	}

}