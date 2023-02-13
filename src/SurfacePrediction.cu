#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceReconstruction.cuh"
#include "PointCloud.h"


//voxweights vox values, depthmap, camparams needs to be copied needs to be included
__global__ void surfacePredictionKernel(Vector3d min, Vector3d max, double* voxWeights, double* voxValues, Matrix4f currentCameraPose, Matrix4f transMatrixcur,
	Vector3f* points, Vector3f* normals, float* camparams, Vector3f voxelDistance)
{
	//Pixel coordinates
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int respectiveX = int(tid % 640); // Column in image pixels
	int respectiveY = int(tid / 640); // Row in image pixels
	//start with the position of the camera which is equal to the translation in currentCameraPose
	Vector2i pixelCoord = Vector2i(respectiveX, respectiveY);
	//Now to cam coordinates

	Vector3f pixelsinCamSpace = Vector3f((pixelCoord[0] - camparams[2]) / camparams[0],
										(pixelCoord[1] - camparams[3]) / camparams[1],
										1.0f);

	// A ray with direction in World Coordinates
	Vector3f raydirection = transMatrixcur.block<3, 3>(0, 0) * pixelsinCamSpace;
	//could not write it in one line do not know why the direction seems to be updating nicely, seems to be updating without return
	raydirection.normalize();

	float updateStepRay = voxelDistance[2];
	Vector3f raystart;
	Vector3f rayprev;
	//to world coordinates in grid
	raystart = (transMatrixcur.block<3, 1>(0, 3) + raydirection * updateStepRay )/ voxelDistance[2];
	rayprev = raystart;
	float previousTsdf = 1;
	float tsdf = 1;

	for (int i = 0; i < 512 ; i++) {
		//ray is in the grid coord
		raystart = (transMatrixcur.block<3, 1>(0, 3) + (raydirection * updateStepRay)) / voxelDistance[2];
		
		if(int(raystart.x())>0 && int(raystart.y())>0 && int(raystart.z())>0 &&
			(raystart.x()) < 512 && int(raystart.y()) < 512 && int(raystart.z()) <512 &&
			voxValues[int(raystart.x()) * 512 * 512 + int(raystart.y()) * 512 + int(raystart.z())]!=1 )
		{
			previousTsdf = tsdf;
			tsdf = voxValues[int(raystart.x()) * 512 * 512 + int(raystart.y()) * 512 + int(raystart.z())];
			
			if (tsdf <= 0 && previousTsdf > 0) {
				Vector3f pointFound;
				//grid to world
				//update step ray is the length of the raydirection
				//this seems to be working
				
				raystart = rayprev + raydirection * previousTsdf / (previousTsdf-tsdf);
				pointFound = Vector3f(min[0] + voxelDistance[0] * raystart[0],
									  min[1] + voxelDistance[1] * raystart[1],
									  min[2] + voxelDistance[2] * raystart[2]);
				
				//	printf(" PointFound, Raystart: %f %f %f , % i, % i, % i \n",  pointFound[0], pointFound[1], pointFound[2], int(raystart.x()), int(raystart.y()), int(raystart.z()));
				
				points[pixelCoord[1] * 640 + pixelCoord[0]] = currentCameraPose.block<3,3>(0,0)* pointFound + currentCameraPose.block<3,1>(0,3);

				//Now we need to find the normals
				/* 
				*	How the interpolation works :
				*	Look at voxels upper lower, right left, front behind, and find out the value 
				*	that normal should have by interpolating them
				*/
				break;

			}
		}
		rayprev = raystart;
		updateStepRay += voxelDistance[2];

	}
	//can never jump over a voxel this way
	//goingto update through z axis therefore I chose the distance along the z axis
	
}
namespace CUDA {
	//Also need the spacing to be able to project the voxels to ->World->cam->image plane, will be calculated on cuda but needs to be adressed for faster update
	//added min and max point of the voxel, min left lower corner, max= left uppercorner
	void SurfacePrediction(Vector3d& min, Vector3d& max, double* voxWeights, double* voxValues, Matrix4f& currentCameraPose, Matrix4f& transMatrixcur,
		std::vector<Vector3f>& points, std::vector<Vector3f>& normals, std::vector<float>& camparams, PointCloud& frame){
		double* voxWeightPointer;
		double* voxValuePointer;

		float* camparamPointer; //params of the source
		Vector3f* pointsPointer;
		Vector3f* normalsPointer;
		//Mallocs
		//Each has a value for one voxel.
		//cudaMalloc(&voxWeightPointer, sizeof(double) * 512 * 512 * 512);
		cudaMalloc(&voxValuePointer, sizeof(double) * 512 * 512 * 512);
		//4 variables in camparams Look at exercise 5 for multiplication
		cudaMalloc(&camparamPointer, sizeof(float) * 4);
		cudaMalloc((void**)&pointsPointer, sizeof(Vector3f) * 640 * 480);
		cudaMalloc((void**)&normalsPointer, sizeof(Vector3f) * 640 * 480);



		//CudaHostalloc should be used can be changed later, I dont want anything to get crashed
		//cudaMemcpy(voxWeightPointer, voxWeights, sizeof(double) * 512 * 512 * 512, cudaMemcpyHostToDevice);
		cudaMemcpy(voxValuePointer, voxValues, sizeof(double) * 512 * 512 * 512, cudaMemcpyHostToDevice);
		cudaMemcpy(camparamPointer, camparams.data(), sizeof(float) * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(pointsPointer, points.data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);
		cudaMemcpy(normalsPointer, normals.data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);

		//Now everything works with copying should create a grid, block and threads to be able to iterate over the values and weight in cuda
		// Same thing as the Pose Estimation, We have x512 y512 z512 weights and values. we are only going to update it by depth meaning
		//First start with for i in range(z): update where 512 512 x and y think of it as an image by 512 512. We are going to update by looking behind of the voxel
		//In the iteration. So 512=2^9 => 512*8, 512/8 => 4096, 64. This should be generalized and not be calcualted by hand!!!!
		Vector3f distanceBetweenVoxels((max[0] - min[0]) / 511,
										(max[1] - min[1]) / 511,
										(max[2] - min[2]) / 511);
		surfacePredictionKernel << <4800, 64 >> > (min, max, voxWeightPointer, voxValuePointer, currentCameraPose, transMatrixcur, pointsPointer, normalsPointer, camparamPointer,distanceBetweenVoxels);

		//cudaMemcpy(voxWeights, voxWeightPointer, sizeof(double) * 512 * 512 * 512, cudaMemcpyDeviceToHost);
		//cudaMemcpy(voxValues, voxValuePointer, sizeof(double) * 512 * 512 * 512, cudaMemcpyDeviceToHost);
		cudaMemcpy(frame.m_points.data(), pointsPointer, sizeof(Vector3f) * 640 * 480, cudaMemcpyDeviceToHost);
		cudaMemcpy(frame.m_normals.data(), normalsPointer, sizeof(Vector3f) * 640 * 480, cudaMemcpyDeviceToHost);


		//call the kernel here
		cudaDeviceSynchronize();
		//cudaFree(voxWeightPointer);
		cudaFree(voxValuePointer);
		cudaFree(camparamPointer);
		cudaFree(pointsPointer);
		cudaFree(normalsPointer);

		//surfaceReconstructionKernel << <4096, 64 >> > (min, max, voxWeightPointer, voxValuePointer, currentCameraPose, transMatrixcur, depthMapPointer, normalsPointer, camparamPointer);


	}
}
