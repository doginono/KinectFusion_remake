#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SurfaceReconstruction.cuh"

//voxweights vox values, depthmap, camparams needs to be copied needs to be included
__global__ void surfaceReconstructionKernel(Vector3d min, Vector3d max, double* voxWeights, double* voxValues, Matrix4f currentCameraPose, Matrix4f transMatrixcur,
	float* depthMap,Vector3f* normals, float* camparams)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// (max[0] - min[0]) / (dx - 1)
	//totaldistance with respect to x y z
	//this can be included in Cuda call
	if (tid<512*512) {
		Vector3f distanceBetweenVoxels((max[0] - min[0]) / 511,
										(max[1] - min[1]) / 511,
										(max[2] - min[2]) / 511);
		int respectiveX = int(tid / 512); // Row in image
		int respectiveY = int(tid % 512); // Column in image

		//Voxel in World coordinates
		Vector3f locationVoxelG;
		//Voxel in camSpace
		Vector3f locationVoxelC;
		//Voxel in pixels
		Vector2i pixelCoord;
		int counter = 0;

		//iterate over the depth, look behind of the voxel!!!!!!!
		for (int z = 0; z < 512; z++) {
			//find the location of the voxel in worldspace, Seems to be working fine
			locationVoxelG = Vector3f(min[0] + distanceBetweenVoxels[0] * respectiveX,
								      min[1] + distanceBetweenVoxels[1] * respectiveY,
									  min[2] + distanceBetweenVoxels[2] * z);
			//Now go to the camspace of the given PointCloud
			locationVoxelC = currentCameraPose.block<3, 3>(0, 0) * locationVoxelG + currentCameraPose.block<3, 1>(0, 3);
			//bring it to image Coordinates project it same as the pose estimation
			pixelCoord=Vector2i( int(locationVoxelC[0] * camparams[0] / locationVoxelC[2] + camparams[2]),
						int(locationVoxelC[1] * camparams[1] / locationVoxelC[2] + camparams[3]));
			
			if (pixelCoord[0] > 0 && pixelCoord[0] < 640 && pixelCoord[1] > 0 && pixelCoord[1] < 480) {
				//voxWeights[0] = 10000;
				//From the original paper Kinectfusion equations 6 7 8 9
				float lambda =(locationVoxelC / locationVoxelC[2]).norm();
				Vector3f translation= currentCameraPose.block<3, 1>(0, 3);
				float RawDepth = depthMap[pixelCoord[1] * 640 + pixelCoord[0]];
				float voxCurrentValue;
				if (RawDepth != MINF) {
					float sdf = (translation - locationVoxelG).norm() - RawDepth;
					//Voila
					//To measure free space
					Vector3f freeSpace = locationVoxelG / locationVoxelG.norm();
					/*From paper cos=> The associated weight WRk(p) is proportional
						to cos(q) / Rk(x), where q is the angle between the associated
						pixel ray directionand the surface normal measurement in the local
						frame.*/
					float cosangle = freeSpace.dot(normals[pixelCoord[1] * 640 + pixelCoord[0]])/ (freeSpace.norm() * normals[pixelCoord[1] * 640 + pixelCoord[0]].norm());
					if (sdf > 0) {
						voxCurrentValue = std::min(1.0f, sdf);
						//printf("sdf: %f", sdf);
						if (voxCurrentValue < 1) {
							//printf("sdf: %f", voxCurrentValue);
						}
					}
					else {
						voxCurrentValue = std::max(-1.0f, sdf);
					}
					//x* dy* dz + y * dz + z;
					//assuming equal weights we can update it later// running average added
					voxValues[respectiveX * 512 * 512 + respectiveY * 512 + z] = (voxWeights[respectiveX * 512 * 512 + respectiveY * 512 + z]*voxValues[respectiveX * 512 * 512 + respectiveY * 512 + z] 
																				+ voxCurrentValue)/(voxWeights[respectiveX * 512 * 512 + respectiveY * 512 + z]+1);
					//This should be updated
					voxWeights[respectiveX * 512 * 512 + respectiveY * 512 + z]+= 1;// voxWeights[respectiveX * 512 * 512 + respectiveY * 512 + z];
				}
			}
		}
		//test
		


	}
}
namespace CUDA{
	//Also need the spacing to be able to project the voxels to ->World->cam->image plane, will be calculated on cuda but needs to be adressed for faster update
	//added min and max point of the voxel, min left lower corner, max= left uppercorner
	void SurfaceReconstruction(Vector3d& min, Vector3d& max, double* voxWeights, double* voxValues, Matrix4f& currentCameraPose, Matrix4f& transMatrixcur,
		float* depthMap, std::vector<Vector3f>& normals, std::vector<float>& camparams) {
		//calling the kernel
		double* voxWeightPointer;
		double* voxValuePointer;

		float* camparamPointer; //params of the source
		float* depthMapPointer;
		Vector3f* normalsPointer;
		//Mallocs
		//Each has a value for one voxel.
		cudaMalloc(&voxWeightPointer, sizeof(double) * 512 * 512 * 512);
		cudaMalloc(&voxValuePointer, sizeof(double) * 512 * 512 * 512);
		//4 variables in camparams Look at exercise 5 for multiplication
		cudaMalloc(&camparamPointer, sizeof(float) * 4);
		cudaMalloc(&depthMapPointer, sizeof(float) * 640 * 480);
		cudaMalloc((void**)&normalsPointer, sizeof(Vector3f) * 640 * 480);



		//CudaHostalloc should be used can be changed later, I dont want anything to get crashed
		cudaMemcpy(voxWeightPointer, voxWeights, sizeof(double) * 512 * 512 * 512, cudaMemcpyHostToDevice);
		cudaMemcpy(voxValuePointer, voxValues, sizeof(double) * 512 * 512 * 512, cudaMemcpyHostToDevice);
		cudaMemcpy(camparamPointer, camparams.data(), sizeof(float) * 4, cudaMemcpyHostToDevice);
		cudaMemcpy(depthMapPointer, depthMap, sizeof(float) * 640*480, cudaMemcpyHostToDevice);
		cudaMemcpy(normalsPointer, normals.data(), sizeof(Vector3f) * 640 * 480, cudaMemcpyHostToDevice);

		//Now everything works with copying should create a grid, block and threads to be able to iterate over the values and weight in cuda
		// Same thing as the Pose Estimation, We have x512 y512 z512 weights and values. we are only going to update it by depth meaning
		//First start with for i in range(z): update where 512 512 x and y think of it as an image by 512 512. We are going to update by looking behind of the voxel
		//In the iteration. So 512=2^9 => 512*8, 512/8 => 4096, 64. This should be generalized and not be calcualted by hand!!!!
		surfaceReconstructionKernel << <4096, 64 >> > (min,max,voxWeightPointer, voxValuePointer, currentCameraPose, transMatrixcur, depthMapPointer, normalsPointer, camparamPointer);

		cudaMemcpy(voxWeights, voxWeightPointer, sizeof(double) * 512 * 512 * 512, cudaMemcpyDeviceToHost);
		cudaMemcpy(voxValues, voxValuePointer, sizeof(double) * 512 * 512 * 512, cudaMemcpyDeviceToHost);


		//call the kernel here
		cudaDeviceSynchronize();
		cudaFree(voxWeightPointer);
		cudaFree(voxValuePointer);
		cudaFree(camparamPointer);
		cudaFree(depthMapPointer);

	}
}
